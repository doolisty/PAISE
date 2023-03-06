#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// SpMV = Sparse Matrix * Vector
// CSR = Compressed Sparse Row

void SpMV_CSR_sequential(const float *data, const float *X, float *Y,
                         int row_num, int *row_ptr, int *col_idx) {
    for (int i = 0; i < row_num; ++i) {
        float val = 0.0f;
        int row_start = row_ptr[i];
        int row_end = row_ptr[i+1];
        for (int j = row_start; j < row_end; ++j) {
            val += data[j] * X[col_idx[j]];
        }
        Y[i] += val; // 这里用 + 而不是 =，可能 Y 被初始化为 bias
    }
}

__global__ void SpMV_CSR_kernel(const float *data, const float *X, float *Y,
                                int row_num, const int *row_ptr, const int *col_idx) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (tid < row_num) {
        for (int i = row_ptr[tid]; i < row_ptr[tid+1]; ++i) {
            val += data[i] * X[col_idx[i]];
        }
        Y[tid] += val;
    }
}

// {data} 是原始矩阵的转置
__global__ void SpMV_ELL_kernel(const float *data, const float *X, const float *Y,
                                int row_num, int row_size, const int *col_idx) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < row_num) {
        float val = 0.0f;
        for (int i = tid; i < row_num * row_size; i += row_size) {
            val += data[i] * X[col_idx[i]];
        }
        Y[tid] = val;
    }
}

int main(int argc, char *argv[]) {
    return 0;
}