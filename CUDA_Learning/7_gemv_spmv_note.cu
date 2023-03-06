#include <cuda_runtime.h>

template <const int WARP_SIZE>
__device__ __forceinline__ float warpReduce(float sum) {
    if (WARP_SIZE >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
    if (WARP_SIZE >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);
    if (WARP_SIZE >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4);
    if (WARP_SIZE >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2);
    if (WARP_SIZE >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

// n > 32
template <const int ROW_PER_BLOCK>
__global__ void sgemv(float* __restrict__ A, float* __restrict__ x, float* __restrict__ y, const int M, const int N) {
    const int warp_size = 32;
    const int thread_lane = threadIdx.x % warp_size;
    const int curr_row = blockIdx.x * ROW_PER_BLOCK + threadIdx.x / warp_size;

    if (curr_row < M) {
        float tmp_sum = 0.0f;
        int k_iter = N / (warp_size * 4);
        if (k_iter == 0) k_iter = 1;

        for (int i = 0; i < k_iter; ++i) {
            int curr_col = i * warp_size + thread_lane;
            float4 A_curr_val = reinterpret_cast<float4*>(&A[curr_row * N])[curr_col];
            float4 x_curr_val = reinterpret_cast<float4*>(x)[curr_col];
            tmp_sum += A_curr_val.x * x_curr_val.x;
            tmp_sum += A_curr_val.y * x_curr_val.y;
            tmp_sum += A_curr_val.z * x_curr_val.z;
            tmp_sum += A_curr_val.w * x_curr_val.w;
        }
        #pragma unroll
        for (int i = warp_size >> 1; i > 0; i >>= 1) {
            tmp_sum += __shfl_down_sync(0xffffffff, tmp_sum, i);
        }
        if (thread_lane == 0) {
            y[curr_row] = tmp_sum;
        }
    }
}


// 开一维线程块
template <const int THREAD_PER_ROW, const int ROW_PER_BLOCK>
__global__ void spmv_csr(const int row_num, const int *A_row_offset, const int *A_col_idx, const float *A_val, const float *x, float *y) {
    const int thread_per_block = THREAD_PER_ROW * ROW_PER_BLOCK;
    const int tid = blockIdx.x * thread_per_block + threadIdx.x;  // thread 的全局偏移
    const int thread_lane = threadIdx.x & (THREAD_PER_ROW - 1);  // threadIdx.x % THREAD_PER_ROW
    const int row_id = tid / THREAD_PER_ROW;

    if (row_id < row_num) {
        const int row_start = A_row_offset[row_id];
        const int row_end = A_row_offset[row_id + 1];

        float tmp_sum = 0.0f;
        for (int i = row_start + thread_lane; i < row_end; i += THREAD_PER_ROW) {
            tmp_sum += A_value[i] * x[A_col_idx[i]];
        }

        tmp_sum = warpReduce<THREAD_PER_ROW>(tmp_sum);
        if (thread_lane == 0) {
            y[row_id] = tmp_sum;
        }
    }
}
