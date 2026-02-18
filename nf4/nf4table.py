import torch
from bitsandbytes.functional import create_normal_map
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# 生成 NF4 码表
# offset=0.0 是标准 NF4，如果是 0.5 则可能是为了对齐某些对称性
nf4_table = create_normal_map(offset=0.0)

logging.info("--- NF4 Lookup Table (Index 0-15) ---")
# 转换为 list 并取前 16 个（标准 NF4 索引范围）
table_values = nf4_table.tolist()[:16]

for i, val in enumerate(table_values):
    # 打印索引、十进制值和十六进制位表示（方便核对数据对齐）
    logging.info(f"Index {i:2d}: {val:12.8f}")

# 额外步骤：直接生成 C++ 数组格式
cpp_array = ", ".join([f"{v:.8f}f" for v in table_values])
logging.info("\n--- Copy-paste for your CUDA Kernel ---")
logging.info(f"__constant__ float nf4_archive[16] = {{ {cpp_array} }};")