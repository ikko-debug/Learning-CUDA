#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
struct Header {
    int64_t num_rows;
    int64_t num_cols;
    int32_t blocksize;
};
