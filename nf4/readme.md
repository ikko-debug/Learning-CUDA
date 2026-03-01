
## NF4反量化
权重 = NF4表值 × (一级scale × 二级scale) + offset
scale = code2[ absmax_q[block] ] × absmax2[group]
w =
  NF4_TABLE[q]
× code2[ absmax_q[block] ]
× absmax2[group]
+ offset
关联关系：
Group (16384 weights)
 └── 256 Blocks
      └── 64 weights
           └── 2 per byte
## bitsandbytes参考实现
```shell
source /data/shared/miniconda3/etc/profile.d/conda.sh && conda activate cuda && python - <<'PY'
import torch
import bitsandbytes as bnb

rows, cols, blocksize = 4096, 4096, 64

# Generate data
x = torch.randn(rows, cols, device='cuda', dtype=torch.bfloat16)
packed, qstate = bnb.functional.quantize_4bit(
    x,
    blocksize=blocksize,
    quant_type='nf4',
    compress_statistics=True
)

# Warmup
for _ in range(5):
    y = bnb.functional.dequantize_4bit(packed, qstate, quant_type='nf4', blocksize=blocksize)

torch.cuda.synchronize()

# Timing
iters = 100
start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    y = bnb.functional.dequantize_4bit(packed, qstate, quant_type='nf4', blocksize=blocksize)
stop.record()
stop.synchronize()

ms = start.elapsed_time(stop) / iters

# Approx bandwidth
num_elements = rows * cols
size_packed = num_elements // 2
num_blocks = (num_elements + blocksize - 1) // blocksize
num_groups = (num_blocks + 255) // 256
size_absmax_q = num_blocks
size_absmax2 = num_groups * 2
size_code2 = 256 * 2
size_out = num_elements * 2

total_bytes = size_packed + size_absmax_q + size_absmax2 + size_code2 + size_out
bandwidth = total_bytes / (ms / 1000.0) / 1e9

print(f"bitsandbytes dequantize_4bit: {ms:.6f} ms")
print(f"Approx bandwidth: {bandwidth:.2f} GB/s")
PY
```
时间：0.083885 ms
近似带宽：503.16 GB/s
## 构造标准测试集





## 编写main.cu
### 反量化查表
来自github bitsandbytes/csrc/kernels.cu
```cpp
__device__ static float fp4_dequantization_lut[8] = {
    0.0f,            // 0b000
    0.005208333333f, // 0b001
    0.66666667f,     // 0b010
    1.0f,            // 0b011
    0.33333333f,     // 0b100
    0.5f,            // 0b101
    0.16666667f,     // 0b110
    0.25f            // 0b111
};

__device__ static float nf4_dequantization_lut[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};
```
bitsandbytes使用device，考虑是否使用constant
| 维度      | **device** static | **constant**      |
| ------- | ----------------- | ----------------- |
| 存储位置    | Global Memory     | Constant Memory   |
| 缓存      | L2 / L1           | 专用 Constant Cache |
| Warp 广播 | 无              |  有               |
| 访问延迟    | 高                 | 低                 |
| 适合      | 普通全局数据            | 查表 / 常量           |
使用constant先。
### host逻辑
1.输入解析，读取二进制文件
2.内存规划
3.数据加载，分配显存
4.启动kernel
5.记录性能，写入数据
### device逻辑
1. 全局一维线程索引
2. 读取这 1 个字节，并解包成两个 4-bit 索引
3. 计算当前字节属于哪一个量化 Block 和 Group
4. 暴力从全局内存 (Global Memory) 读取双重量化参数
5. 结合 NF4 查表，计算真实的浮点权重
6. 最朴素的分别写回 (没有使用 Union 向量化合并写入)

## nativekernel问题解决
精度远超阈值
### 解码索引覆盖不足
最初 grid 维度按 num_groups 计算，实际应按 packed byte 数计算，导致大部分元素没写。
```
Kernel Time: 0.121476 ms
Effective Bandwidth (approx): 347.457 GB/s
Output written to nf4/data/output.bin
```

### gendata，offset 处理错误
offset 并非恒为 0，而且应加在 absmax 上，不应加在最终权重上。
```
Kernel Time: 0.141412 ms
Effective Bandwidth (approx): 298.473 GB/s
Output written to nf4/data/output.bin
MAE (Mean Absolute Error): 0.000024
Max Error:                 0.031250
------------------------------
✅ PASS: MAE (0.000024) is within threshold (0.01)
```
### 为了可拓展性，使用 Grid-Stride Loops
```cpp
int blockSize = 256;
// 我们根据硬件 SM 数量来决定 grid，或者简单给一个足够大的数
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
int gridSize = 32 * numSMs; // 保证每个 SM 都有活干
```
修改代码
```cpp
int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int grid_x = sm_count * 4;
    int64_t max_grid = (total_bytes + blockDim.x - 1) / blockDim.x;
    if (grid_x > max_grid) {
        grid_x = static_cast<int>(max_grid);
    }
    if (grid_x < 1) {
        grid_x = 1;
    }
    dim3 gridDim(grid_x);
```
此时性能
```
Kernel Time: 0.121476 ms
Effective Bandwidth (approx): 347.457 GB/s
```
换成通用api计算
```cpp
int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    int grid_x = sm_count * 4;
    int64_t max_grid = (total_bytes + blockDim.x - 1) / blockDim.x;
    if (grid_x > max_grid) {
        grid_x = static_cast<int>(max_grid);
    }
    if (grid_x < 1) {
        grid_x = 1;
    }
    dim3 gridDim(grid_x);
```
此时性能
```
SM count: 108, max active blocks/SM: 8, grid_x: 864
Kernel Time: 0.124518 ms
Effective Bandwidth (approx): 338.969 GB/s
Output written to nf4/data/output.bin
```
## 优化大纲
1. 向量化访存优化
```cpp
output[tid * 2]     = __float2bfloat16(w1_fp32);
    output[tid * 2 + 1] = __float2bfloat16(w2_fp32);
```
这发起了两次独立的写入，gpu显存控制器可以一次性吃32-bit，64-bit，甚至128-bit的数据

2. 冗余计算与重复访存
blocksize = 64，那么 32 个线程（正好是一个 Warp）处理的 64 个权重，其实都属于同一个 Block。
3. 整数除法（位运算优化）
4. code2 码表的读取延迟
## 实际优化
### 向量化访存
向量化加载 packed_weights
```cpp
uint8_t packed = packed_weights[byte_idx];
```
单字节 load，效率差。GPU 更适合 4B / 8B 对齐访问。
需注意，主机端对 packed buffer 做 4 字节对齐补零，避免最后一个 uint32 读取越界。
```cpp
for (int b = 0; b < 4; ++b) {
            int64_t byte_idx = word_idx * 4 + b;
            if (byte_idx >= total_bytes) {
                continue;
            }

            uint8_t packed = static_cast<uint8_t>((packed_word >> (8 * b)) & 0xFF);
            uint8_t idx1 = (packed >> 4) & 0x0F; // 高 4 位对应 output[byte_idx * 2]
            uint8_t idx2 = packed & 0x0F;        // 低 4 位对应 output[byte_idx * 2 + 1]

            int bytes_per_block = blocksize / 2;
            int block_id = static_cast<int>(byte_idx / bytes_per_block);
            int group_id = block_id / group_size;

            float a2 = __half2float(absmax2[group_id]);
            uint8_t qa = absmax_q[block_id];
            float c2 = __half2float(code2[qa]);
            float real_absmax = c2 * a2 + offset;

            float w1_fp32 = NF4_LUT[idx1] * real_absmax;
            float w2_fp32 = NF4_LUT[idx2] * real_absmax;

            int64_t out_idx = byte_idx * 2;
            if (out_idx < num_elements) {
                output[out_idx] = __float2bfloat16(w1_fp32);
            }
            if (out_idx + 1 < num_elements) {
                output[out_idx + 1] = __float2bfloat16(w2_fp32);
            }
        }
// Kernel Time: 0.172722 ms
// Effective Bandwidth (approx): 244.368 GB/s
```
#### 读取方式改为 16B 向量化

kernel 参数从 const uint32_t* 改为 const uint4*
每线程处理 1 个 16B word（32 个权重），total_words = ceil(total_bytes / 16)
读取 16B 后用 #pragma unroll 在寄存器里拆 16 个 byte，再拆成 32 个 4-bit 索引
参数只读一次

对应这 32 个权重只计算一次 block_id/group_id
只读一次 absmax2/absmax_q/code2/offset，得到 real_absmax，后面 32 个权重复用
写回向量化

每个byte 生成 2 个 bf16，打包成 1 个 uint32
16 个 uint32 组成 4 个 float4，一次写回 8 个 bf16
尾部不足 32 权重时走标量写回，避免越界
主机侧对齐与网格调整

packed buffer padding 改为 16B 对齐
cudaMalloc/cudaMemcpy 使用对齐后的大小
grid 上限按 total_words 而不是 total_bytes 计算
#### 此外，精简写回：跳过 float4 强转
```cpp
out_f4[0] = *reinterpret_cast<float4*>(&v0);
```
既然已经拼接好了 uint4，直接用 uint4 类型的指针写回即可。GPU 并不在乎你存入的是 float 还是 uint，只要它是 128 位的。当告诉 GPU 要执行一个存储（Store）操作时，硬件只需要知道两件事：
起始地址（Starting Address）： 数据要写到哪？
位宽（Bit Width）： 这一趟搬多大的数据（32位、64位还是 128位）？
当定义一个 uint4 变量并写回时，CUDA 编译器（NVCC）会生成一条类似 STG.E.128 (Store Global 128-bit) 的汇编指令。这条指令不管这 128 位里装的是 8 个 bfloat16、4 个 float，还是 16 个 char。它只负责把寄存器里的 128 个比特流，原封不动地拍到对应的显存地址上。
### 在native上实现合并写回（无向量访存）
```cpp
if (out_idx + 1 < num_elements) {
            Bf16Bits lo; lo.bf = __float2bfloat16(w1_fp32);
            Bf16Bits hi; hi.bf = __float2bfloat16(w2_fp32);
            out_u32[out_idx / 2] = static_cast<uint32_t>(lo.u) | (static_cast<uint32_t>(hi.u) << 16);
        } else if (out_idx < num_elements) {
            output[out_idx] = __float2bfloat16(w1_fp32);
        }
Kernel Time: 0.102705 ms
Effective Bandwidth (approx): 410.962 GB/s
```
## 加载表constant为广播可能conflict
```cpp
__shared__ float s_LUT[16];
    if (threadIdx.x < 16) {
        s_LUT[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    __syncthreads();
```
搬到 SM（流处理器）内部极其昂贵、速度极快的 Shared Memory（共享内存）中。