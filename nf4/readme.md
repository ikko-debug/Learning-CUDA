
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