#include <algorithm>
#include <cmath>
#include <vector>
#include <musa_fp16.h>

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
constexpr int Br = 16;
constexpr int Bc = 16;

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
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    float scale) {
  //线程布局: x 维对应列(Bc)，y 维对应行(Br)
  int tx = threadIdx.x; //Bc 维度
  int ty = threadIdx.y; //Br 维度

  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int q_block_idx = blockIdx.x;

  //当前 block 负责的 query 行区间
  int q_start_idx = q_block_idx * Br;
  int q_len_local = min(Br, target_seq_len - q_start_idx);

  int kv_head_idx = (head_idx * kv_heads) / query_heads;

  //共享内存布局：Q tile(Br*D), K tile(Bc*D), V tile(Bc*D), O accum(Br*D), m(Br), l(Br)
  extern __shared__ float smem[];
  float* s_Q = smem;                                 //Br * head_dim
  float* s_K = s_Q + Br * head_dim;                  //Bc * head_dim
  float* s_V = s_K + Bc * head_dim;                  //Bc * head_dim
  float* s_O = s_V + Bc * head_dim;                  //Br * head_dim
  float* s_m = s_O + Br * head_dim;                  //Br
  float* s_l = s_m + Br;                             //Br

  //1 载入 Q tile，初始化输出累加器与统计量
  for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < Br * head_dim; i += blockDim.x * blockDim.y) {
    int r = i / head_dim;
    int c = i % head_dim;
    int global_q = q_start_idx + r;
    if (r < q_len_local && global_q < target_seq_len) {
      size_t q_index = ((static_cast<size_t>(batch_idx) * target_seq_len + global_q) * query_heads + head_idx) * head_dim + c;
      s_Q[r * head_dim + c] = to_float(Q[q_index]);
    } else {
      s_Q[r * head_dim + c] = 0.0f;
    }
    s_O[r * head_dim + c] = 0.0f;
  }
  if (tx == 0 && ty < Br) {
    s_m[ty] = -1e20f;
    s_l[ty] = 0.0f;
  }
  __syncthreads();

  for (int j_base = 0; j_base < src_seq_len; j_base += Bc) {
    //2 逐块加载 K/V tile
    int kv_len_local = min(Bc, src_seq_len - j_base);

    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < Bc * head_dim; i += blockDim.x * blockDim.y) {
      int r = i / head_dim;
      int c = i % head_dim;
      int global_k = j_base + r;
      if (r < kv_len_local && global_k < src_seq_len) {
        size_t k_index = ((static_cast<size_t>(batch_idx) * src_seq_len + global_k) * kv_heads + kv_head_idx) * head_dim + c;
        s_K[r * head_dim + c] = to_float(K[k_index]);
        s_V[r * head_dim + c] = to_float(V[k_index]);
      } else {
        s_K[r * head_dim + c] = 0.0f;
        s_V[r * head_dim + c] = 0.0f;
      }
    }
    __syncthreads();

    //3 对每个 query 行做在线 softmax 更新并累加输出
    if (tx == 0 && ty < q_len_local) {
      for (int tk = 0; tk < kv_len_local; ++tk) {
        int global_q_idx = q_start_idx + ty;
        int global_k_idx = j_base + tk;
        if (is_causal && global_k_idx > global_q_idx) {
          continue;
        }

        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          dot += s_Q[ty * head_dim + d] * s_K[tk * head_dim + d];
        }
        dot *= scale;

        float m_prev = s_m[ty];
        float l_prev = s_l[ty];
        float m_new = fmaxf(m_prev, dot);
        float p_prev = expf(m_prev - m_new);
        float p_curr = expf(dot - m_new);
        float l_new = l_prev * p_prev + p_curr;

        for (int d = 0; d < head_dim; ++d) {
          s_O[ty * head_dim + d] = s_O[ty * head_dim + d] * p_prev + p_curr * s_V[tk * head_dim + d];
        }

        s_m[ty] = m_new;
        s_l[ty] = l_new;
      }
    }

    __syncthreads();
  }

  //4) 归一化并写回输出
  if (ty < q_len_local) {
    float denom = s_l[ty];
    float inv_l = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    for (int d = tx; d < head_dim; d += Bc) {
      float val = s_O[ty * head_dim + d] * inv_l;
      int global_q = q_start_idx + ty;
      size_t o_index = ((static_cast<size_t>(batch_idx) * target_seq_len + global_q) * query_heads + head_idx) * head_dim + d;
      O[o_index] = from_float<T>(val);
    }
  }
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
  RUNTIME_CHECK(musaMalloc(&d_input, total_elems * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_out, sizeof(T)));
  RUNTIME_CHECK(musaMemcpy(d_input, h_input.data(), total_elems * sizeof(T), musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemset(d_out, 0, sizeof(T)));

  dim3 block(BLOCK_SIZE);
  dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  traceKernel<T><<<grid, block>>>(d_input, rows, cols, n, d_out);
  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());

  T h_out = T(0);
  RUNTIME_CHECK(musaMemcpy(&h_out, d_out, sizeof(T), musaMemcpyDeviceToHost));

  RUNTIME_CHECK(musaFree(d_input));
  RUNTIME_CHECK(musaFree(d_out));
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
  RUNTIME_CHECK(musaMalloc(&d_q, q_elems * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_k, k_elems * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_v, v_elems * sizeof(T)));
  RUNTIME_CHECK(musaMalloc(&d_o, o_elems * sizeof(T)));

  //Host -> Device: 拷贝 Q/K/V 到 GPU，并清空输出
  RUNTIME_CHECK(musaMemcpy(d_q, h_q.data(), q_elems * sizeof(T), musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_k, h_k.data(), k_elems * sizeof(T), musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemcpy(d_v, h_v.data(), v_elems * sizeof(T), musaMemcpyHostToDevice));
  RUNTIME_CHECK(musaMemset(d_o, 0, o_elems * sizeof(T)));

  //线程块: (Bc, Br) 形成 [列, 行] 的 tile 计算
  dim3 block(Bc, Br);
  int grid_x = (target_seq_len + Br - 1) / Br;
  dim3 grid(grid_x, query_heads, batch_size);
  //共享内存大小与 kernel 中的 smem 布局保持一致
  size_t shared_bytes = (Br * head_dim + Bc * head_dim + Bc * head_dim + Br * head_dim + Br + Br) * sizeof(float);
  float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  flash_attention_v1_kernel<T><<<grid, block, shared_bytes>>>(
    d_q, d_k, d_v, d_o,
    batch_size, target_seq_len, src_seq_len,
    query_heads, kv_heads, head_dim, is_causal, scale);

  RUNTIME_CHECK(musaGetLastError());
  RUNTIME_CHECK(musaDeviceSynchronize());

  //Device -> Host: 拷回输出结果
  RUNTIME_CHECK(musaMemcpy(h_o.data(), d_o, o_elems * sizeof(T), musaMemcpyDeviceToHost));

  RUNTIME_CHECK(musaFree(d_q));
  RUNTIME_CHECK(musaFree(d_k));
  RUNTIME_CHECK(musaFree(d_v));
  RUNTIME_CHECK(musaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
