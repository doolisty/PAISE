#include <cuda_runtime.h>

template <const int TILE_SIZE>
__global__ void reduce0(const float *d_in, float *d_out, const int data_num) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    __shared__ float tile_shared[TILE_SIZE];

    if (idx < data_num) {
        tile_shared[tid] = d_in[idx];
    }
    __syncthreads();

    for (int stride = 1; stride < TILE_SIZE; stride *= 2) {
        if (tid % (stride * 2) == 0) {
            tile_shared[tid] += tile_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = tile_shared[0];
    }
}

// 优化 1：消除 warp divergence
template<const int TILE_SIZE>
__global__ void reduce1(const float *d_in, float *d_out, const int data_num) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * TILE_SIZE + threadIdx.x;
    __shared__ float tile_shared[TILE_SIZE];

    if (idx < data_num) {
        tile_shared[tid] = d_in[idx];
    }
    __syncthreads();

    for (int stride = 1; stride < TILE_SIZE; stride *= 2) {
        int index = 2 * stride * tid;  // 用最前面的几个线程计算
        if (index < TILE_SIZE) {
            tile_shared[tid] += tile_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = tile_shared[tid];
    }
}

// 优化 2：消除 bank conflict
template <const int TILE_SIZE>
__global__ void reduce2(const float *d_in, float *d_out, const int data_num) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float tile_shared[TILE_SIZE];
    if (idx < data_num) {
        tile_shared[tid] = d_in[idx];
    }
    __syncthreads();

    for (int stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tile_shared[tid] += tile_shared[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_out[blockIdx.x] = tile_shared[tid];
    }
}
