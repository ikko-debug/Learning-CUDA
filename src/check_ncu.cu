#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

__global__ void saxpy(const float* x, const float* y, float* out, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a * x[idx] + y[idx];
    }
}

static void check_cuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << "\n";
        std::exit(1);
    }
}

int main() {
    const int n = 1 << 20;
    const float a = 2.5f;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* h_x = static_cast<float*>(std::malloc(bytes));
    float* h_y = static_cast<float*>(std::malloc(bytes));
    float* h_out = static_cast<float*>(std::malloc(bytes));
    if (!h_x || !h_y || !h_out) {
        std::cerr << "host malloc failed\n";
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i % 100);
        h_y[i] = static_cast<float>((i * 2) % 100);
    }

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_out = nullptr;
    check_cuda(cudaMalloc(&d_x, bytes), "cudaMalloc d_x failed");
    check_cuda(cudaMalloc(&d_y, bytes), "cudaMalloc d_y failed");
    check_cuda(cudaMalloc(&d_out, bytes), "cudaMalloc d_out failed");

    check_cuda(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_x failed");
    check_cuda(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_y failed");

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    saxpy<<<blocks, threads>>>(d_x, d_y, d_out, a, n);
    check_cuda(cudaGetLastError(), "kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "kernel sync failed");

    check_cuda(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy h_out failed");

    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float expected = a * h_x[i] + h_y[i];
        float err = std::fabs(h_out[i] - expected);
        if (err > max_err) {
            max_err = err;
        }
    }

    std::cout << "SAXPY max error: " << max_err << "\n";

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
    std::free(h_x);
    std::free(h_y);
    std::free(h_out);
    return 0;
}
