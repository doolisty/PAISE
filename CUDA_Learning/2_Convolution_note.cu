#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_MASK_WIDTH 10
__constant__ float M[MAX_MASK_WIDTH];


__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P,
                                            const int mask_width, const int width) {
    const int n = mask_width >> 1;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float Pval = 0;
    const int start = tid - n;
    for (int j = 0; j < mask_width; ++j) {
        if (start + j >= 0 && start + j < width) {
            Pval += N[start + j] * M[j];
        }
    }
    P[tid] = Pval;
}


// 每个 block 后 n 个线程从前一个 block 的位置取 left_halo
// 前 n 个线程从后一个 block 的位置取 right_halo，n 是 mask 长度的一半
template <int TILE_SIZE>
__global__ void convolution_1D_tiled_kernel(float *N, float *M, float *P,
                                            const int mask_width, const int width){
    int n = mask_width >> 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float N_shared[TILE_SIZE + MAX_MASK_WIDTH - 1];

    // load left halo elems
    const int left_halo_idx = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - n) { // 假设了一个 block 的大小要比 mask 一半的 n 大
        N_shared[threadIdx.x - (blockDim.x - n)]  // N_shared 顶头放，所以要减去当前线程在 block 中的偏移
            = (left_halo_idx < 0) ? 0 : N[left_halo_idx];
    }

    // load center elems
    N_shared[threadIdx.x + n] = N[tid];

    // load right halo elems
    const int right_halo_idx = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    if (threadIdx.x < n) { // 前 n 个 thread
        N_shared[n + blockDim.x + threadIdx.x] // n 是已经放入的左 halo
            = (right_halo_idx >= width) ? 0 : N[right_halo_idx];
    }

    __syncthreads();

    // calculate P[tid]
    float Pval = 0;
    for (int i = 0; i < mask_width; ++i) {
        Pval += N_shared[threadIdx.x + i] * M[i];
    }
    P[tid] = Pval;
}


template <int TILE_SIZE>
__global__ void convolution_1D_tiled_caching_kernel(float *N, float *M, float *P,
                                                    const int mask_width, const int width) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = mask_width >> 1;
    __shared__ float N_shared[TILE_SIZE];

    N_shared[threadIdx.x] = N[tid];

    __syncthreads();

    float Pval = 0;
    const int n_start = tid - n;
    const int tile_start = blockIdx.x * blockDim.x;
    const int tile_end = (blockIdx.x + 1) * blockDim.x;
    for (int i = 0; i < mask_width; ++i) {
        int n_index = n_start + i;
        if (n_index >= 0 && n_index < width) {
            if (n_index >= tile_start && n_index < tile_end) {
                Pval += N_shared[threadIdx.x - n + i] * M[i];
            } else {
                Pval += N[n_index] * M[i];
            }
        }
    }
    P[tid] = Pval;
}


//
template <int O_TILE_WIDTH>
__global__ void convolution_2D_tiled_kernel(float *N, float *P, const int height, const int width,
                                            const int pitch, const int mask_width,
                                            const float __restrict__ *M) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_o = O_TILE_WIDTH * blockIdx.y + ty;
    const int col_o = O_TILE_WIDTH * blockIdx.x + tx;
    const int n = mask_width >> 1;

    // 在位置上与 tx, ty 一致（但是全局偏移
    const int row_i = row_o - n;
    const int col_i = col_o - n;

    __shared__ float N_tile_shared[O_TILE_WIDTH + MAX_MASK_WIDTH - 1][O_TILE_WIDTH + MAX_MASK_WIDTH - 1];
    
    // [STEP 1] load input tile into shared memory
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        N_tile_shared[ty][tx] = N[row_i * pitch + col_i];
    } else {
        N_tile_shared[ty][tx] = 0.0f;
    }

    __syncthreads();

    // [STEP 2] 用前 (O_TILE_WIDTH * O_TILE_WIDTH) 个线程计算 output_tile
    // （一个 block 里有 (O_TILE_WIDTH + MAX_MASK_WIDTH - 1)^2 个线程，与 N_tile_shared 大小相同）
    float Pval = 0.0f;
    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
        for (int i = 0; i < mask_width; ++i) {
            for (int j = 0; j < mask_width; ++j) {
                // (ty, tx) 相当于当前线程计算的 out_pixel 的卷积核开始的位置（左上角
                Pval += M[i * mask_width + j] * N_tile_shared[ty+i][tx+j];
            }
        }
        // 有必要吗？
        if (row_o < height && col_o < width) {
            P[row_o * pitch + col_o] = Pval;
        }
    }
}

template <int O_TILE_WIDTH, int CHANNELS>
__global__ void convolution_2D_tiled_channels_kernel(float *N, float *P, const int height, const int width,
                                                     const int pitch, const int mask_width,
                                                     const float __restrict__ *M) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row_o = O_TILE_WIDTH * blockIdx.y + ty;
    const int col_o = O_TILE_WIDTH * blockIdx.x + tx;
    const int n = mask_width >> 1;

    // 在位置上与 tx, ty 一致（但是全局偏移
    const int row_i = row_o - n;
    const int col_i = col_o - n;

    __shared__ float N_tile_shared[O_TILE_WIDTH + MAX_MASK_WIDTH - 1][O_TILE_WIDTH + MAX_MASK_WIDTH - 1];
    
#pragma unroll
    for (int k = 0; k < CHANNELS; ++k) {
        if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
            N_tile_shared[ty][tx] = N[(row_i * pitch + col_i) * CHANNELS + k];
        } else {
            N_tiled_shared[ty][tx] = 0.0f;
        }

        __syncthreads();

        float Pval = 0.0f;
        if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
            for (int i = 0; i < mask_width; ++i) {
                for (int j = 0; j < mask_width; ++j) {
                    Pval += M[i * mask_width + j] * N_tile_shared[ty+i][tx+j];
                }
            }
        }
        if (row_o < height && col_o < width) {
            P[(row_o * pitch + col_o) * CHANNELS + k] = Pval;
        }
        
    }
}

int main(int argc, char *argv[]) {
    return 0;
}