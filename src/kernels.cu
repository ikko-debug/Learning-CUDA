#include <algorithm>
#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
#define BLOCK_SIZE 256

//1.Warp Reduce: 在 32 个线程内快速求和 (不需 Shared Memory)
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    //0xffffffff 表示 Warp 里所有 32 个线程都参与
    //每次折叠一半: 16 -> 8 -> 4 -> 2 -> 1
    //除2等同于右移1位
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        
        //"当前值" + "offset个偏移量位置"
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
//辅助函数：Warp 内求最大值
template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        T temp = __shfl_down_sync(0xffffffff, val, offset);
        if (temp > val) val = temp;
    }
    return val;
}
//2.Block Reduce: 在整个 Block 内求和
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    //静态分配共享内存，用来存放每个 Warp 的总和
    //一个 Block 32 个 Warps
    static __shared__ T shared[32]; 
    
    int lane = threadIdx.x % warpSize; //Warp内排第几 (0-31)
    int wid  = threadIdx.x / warpSize; //第几个 Warp

    //每个 Warp 内部先归并
    val = warpReduceSum(val);

    //Warp 0把结果写到 Shared Memory
    if (lane == 0) {
        shared[wid] = val;
    }

    //等待所有 Warp 写完
    __syncthreads();

    //由第一个 Warp (warp 0) 负责把 Shared Memory 里的数加起来
    //只有前 (blockDim.x / 32) 个线程需要读取数据
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    //第一个 Warp 再次做 Warp Reduce 
    //加上&& lane==0不对，所有wrap0所有线程都进入warpreducesum
    //在tracekernel里再判断
    if (wid == 0 ) {
        val = warpReduceSum(val);
    }

    return val;
}

//3.The Optimized Kernel
template <typename T>
__global__ void traceKernel(const T* __restrict__ input, size_t rows, size_t cols, size_t n, T* out) {
    //这里的 stride 是整个 Grid 一次能处理的数据量
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x; //一个grid的总线程数
    //local sum for each thread in register
    T local_sum = 0;

    //防止一个grid处理不完
    for (size_t i = idx; i < n; i += stride) {
        //i * cols + i 是对角线元素的物理索引
        local_sum += input[i * cols + i];
    }

    //Block 内部归并
    local_sum = blockReduceSum(local_sum);

    //只有 Block 的 Thread 0 负责写回 Global Memory
    //极大减少 atomicAdd 的竞争
    if (threadIdx.x == 0) {
        atomicAdd(out, local_sum);
    }
}

//类型转换: 支持 float/half
template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ half from_float<half>(float v) {
  return __float2half(v);
}

//native_attention_kernel)
template <typename T>
__global__ void native_attention_kernel(const T* q, const T* k, const T* v, T* o,int batch_size, int target_seq_len, int src_seq_len,int query_heads, int kv_heads, int head_dim, bool is_causal) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t o_elems = batch_size * target_seq_len * query_heads * head_dim;
  if (idx >= o_elems) {
    return;
  }

  //线性索引 -> (b, t, qh, d)
  int d = static_cast<int>(idx % head_dim);
  size_t tmp = idx / head_dim;
  int qh = static_cast<int>(tmp % query_heads);
  tmp /= query_heads;
  int t = static_cast<int>(tmp % target_seq_len);
  int b = static_cast<int>(tmp / target_seq_len);

  //GQA: query head -> kv head
  int kv_h = (qh * kv_heads) / query_heads;

  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

  //Step 1: 计算 max score (数值稳定)
  float max_score = -INFINITY;
  for (int sk = 0; sk < src_seq_len; ++sk) {
    if (is_causal && sk > t) {
      continue;
    }
    const T* q_ptr = q + (((b * target_seq_len + t) * query_heads + qh) * head_dim);
    const T* k_ptr = k + (((b * src_seq_len + sk) * kv_heads + kv_h) * head_dim);
    float dot = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      dot += to_float(q_ptr[i]) * to_float(k_ptr[i]);
    }
    float score = dot * scale;
    if (score > max_score) {
      max_score = score;
    }
  }

  //Step 2: 计算 softmax 分母
  float denom = 0.0f;
  for (int sk = 0; sk < src_seq_len; ++sk) {
    if (is_causal && sk > t) {
      continue;
    }
    const T* q_ptr = q + (((b * target_seq_len + t) * query_heads + qh) * head_dim);
    const T* k_ptr = k + (((b * src_seq_len + sk) * kv_heads + kv_h) * head_dim);
    float dot = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      dot += to_float(q_ptr[i]) * to_float(k_ptr[i]);
    }
    float score = dot * scale;
    denom += expf(score - max_score);
  }

  if (denom == 0.0f) {
    o[idx] = from_float<T>(0.0f);
    return;
  }

  //Step 3: 计算加权和 (输出元素)
  float out_val = 0.0f;
  for (int sk = 0; sk < src_seq_len; ++sk) {
    if (is_causal && sk > t) {
      continue;
    }
    const T* q_ptr = q + (((b * target_seq_len + t) * query_heads + qh) * head_dim);
    const T* k_ptr = k + (((b * src_seq_len + sk) * kv_heads + kv_h) * head_dim);
    const T* v_ptr = v + (((b * src_seq_len + sk) * kv_heads + kv_h) * head_dim);
    float dot = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      dot += to_float(q_ptr[i]) * to_float(k_ptr[i]);
    }
    float score = dot * scale;
    float w = expf(score - max_score) / denom;
    out_val += w * to_float(v_ptr[d]);
  }

  o[idx] = from_float<T>(out_val);
}

template <typename T>
__global__ void flash_attention_v1_kernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal, float scale) {
  //每个 Block 处理一个 Query 向量 (1 x head_dim)
  const int q_idx = blockIdx.x;
  const int tid = threadIdx.x; //线程处理 d 维中的一个分量

  const int q_vecs = batch_size * target_seq_len * query_heads;
  if (q_idx >= q_vecs || tid >= head_dim) {
    return;
  }

  //线性索引 -> (b, t, qh)
  int tmp = q_idx;
  int qh = tmp % query_heads;
  tmp /= query_heads;
  int t = tmp % target_seq_len;
  int b = tmp / target_seq_len;

  //GQA: query head -> kv head
  int kv_h = (qh * kv_heads) / query_heads;

  const T* q_ptr = Q + q_idx * head_dim;

  //将当前 Block 负责的 Q 向量加载到寄存器
  float q_val = to_float(q_ptr[tid]);

  //Online Softmax 统计量
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float o_i = 0.0f;

  //Shared Memory 存放 K 和 V 的块
  extern __shared__ float s_mem[];
  float* s_k = s_mem;                          //head_dim
  float* s_v = s_mem + head_dim;               //head_dim
  float* s_dot = s_mem + 2 * head_dim;         //1 float
  float* s_red = s_mem + 2 * head_dim + 1;     //head_dim

  for (int j = 0; j < src_seq_len; ++j) {
    if (is_causal && j > t) {
      continue;
    }

    const T* k_ptr = K + (((b * src_seq_len + j) * kv_heads + kv_h) * head_dim);
    const T* v_ptr = V + (((b * src_seq_len + j) * kv_heads + kv_h) * head_dim);

    //加载 K 和 V 到 Shared Memory
    s_k[tid] = to_float(k_ptr[tid]);
    s_v[tid] = to_float(v_ptr[tid]);
    __syncthreads();

    //计算点积 S = Q * K^T (通用 block 归并，支持非 32 对齐)
    float score = q_val * s_k[tid];
    s_red[tid] = score;
    __syncthreads();

    int n = head_dim;
    for (int stride = (n + 1) / 2; stride > 0; stride = (stride + 1) / 2) {
      if (tid < stride) {
        int other = tid + stride;
        if (other < n) {
          s_red[tid] += s_red[other];
        }
      }
      __syncthreads();
      if (stride == 1) {
        break;
      }
    }

    if (tid == 0) {
      s_dot[0] = s_red[0] * scale;
    }
    __syncthreads();
    float dot = s_dot[0];

    //Online Softmax 更新
    float m_prev = m_i;
    float l_prev = l_i;
    m_i = fmaxf(m_prev, dot);
    float p_prev = expf(m_prev - m_i);
    float p_curr = expf(dot - m_i);
    l_i = l_prev * p_prev + p_curr;
    o_i = (o_i * l_prev * p_prev + p_curr * s_v[tid]) / l_i;

    __syncthreads();
  }

  O[q_idx * head_dim + tid] = from_float<T>(o_i);
}



template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  if (rows == 0 || cols == 0) {
    return T(0);
  }
  size_t n = std::min(rows, cols);
  size_t total_elems = rows * cols;

  T* d_input = nullptr;
  T* d_out = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_input, total_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_out, sizeof(T)));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), total_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_out, 0, sizeof(T)));

  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  traceKernel<T><<<grid, block>>>(d_input, rows, cols, n, d_out);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  T h_out = T(0);
  RUNTIME_CHECK(cudaMemcpy(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_out));
  return h_out;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 || query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    h_o.clear();
    return;
  }

  const size_t q_elems = batch_size * target_seq_len * query_heads * head_dim;
  const size_t k_elems = batch_size * src_seq_len * kv_heads * head_dim;
  const size_t v_elems = k_elems;
  const size_t o_elems = q_elems;

  h_o.resize(o_elems);

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, q_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_k, k_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_v, v_elems * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(&d_o, o_elems * sizeof(T)));

  //Host -> Device: 拷贝 Q/K/V 到 GPU，并清空输出
  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_elems * sizeof(T), cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemset(d_o, 0, o_elems * sizeof(T)));

    //Device kernel launch: flash attention v1 (每个 block 处理一个 query 向量)
  dim3 block(head_dim); //一个block处理全部head_dim维度
  dim3 grid(batch_size * target_seq_len * query_heads);
  size_t shared_bytes = (3 * head_dim + 1) * sizeof(float);
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  flash_attention_v1_kernel<T><<<grid, block, shared_bytes>>>(
    d_q, d_k, d_v, d_o,
    batch_size, target_seq_len, src_seq_len,
    query_heads, kv_heads, head_dim, is_causal, scale);

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  //Device -> Host: 拷回输出结果
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_elems * sizeof(T), cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
}

//*********************************************************************
//Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
//DO NOT MODIFY THIS SECTION
//*********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
