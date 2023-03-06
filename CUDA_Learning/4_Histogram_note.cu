#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <int BLOCK_SIZE, int TILE_SIZE>
__global__ void histogram_block_kernel(const unsigned char *buffer, const long size, unsigned int *histo) {
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int tid = threadIdx.x;
// 书里一开始没用 shared memory
//     __shared__ unsigned char buf_shared[BLOCK_SIZE * TILE_SIZE];

// #pragma unroll
//     for (int i = 0; i < TILE_SZIE; ++i) {
//         if (idx * TILE_SIZE + i < size) {
//             buf_shared[tid * TILE_SIZE + i] = buffer[idx * TILE_SIZE + i];
//         }
//     }

//     __syncthreads();

#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        // 不要忘记检查越界，读数据和计算的时候都记得检查
        if (idx * TILE_SIZE + i < size) {
            // 书里一开始没用 shared memory
            // int offset = buf_shared[tid * TILE_SIZE + i] - 'a';
            int offset = buffer[idx * TILE_SIZE + i] - 'a';
            if (offset >= 0 && offset < 26) {
                atomicAdd(&(histo[offset >> 2]), 1);
            }
        }
    }
}


// 没有 TILE_SIZE，这里一个线程处理一个数据
__global__ void histogram_interleaved_kernel(const unsigned char *buffer, const long size, unsigned int *histo) {
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // stride = blockDim.x * gridDim.x，即每次迭代所有线程都取连续的一个大块，有利于 cache 命中
    for (unsigned int i = tid; i < size; i += blockDim.x * gridDim.x) {
        int offset = buffer[i] - 'a';
        if (offset >= 0 && offset < 26) {
            atomicAdd(&(histo[offset >> 2]), 1);
        }
    }
}


// num_bins：histogram 的区间数
__global__ void histogram_privatized_kernel(const unsigned char *buffer, const long size, unsigned int *histo,
                                            const int num_bins) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 书上这里的 histo_shared 是在 launch kernel 时定义的，但总觉得怪怪的
    __shared__ unsigned int *histo_shared = new int[num_bins];
    for (int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        histo_shared[binIdx] = 0u;
    }

    __syncthreads();

    for (unsigned int i = tid; i < size; i += blockDim.x * gridDim.x) {
        int offset = buffer[i] - 'a';
        if (offset >= 0 && offset < 26) {
            atomicAdd(&(histo_shared[offset >> 2]), 1);
        }
    }

    __syncthreads();

    // write back
    for (int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        atomicAdd(&(histo[binIdx]), histo_shared[binIdx]);
    }
}


__global__ void histogram_privatized_aggregation_kernel(const unsigned char *buffer, const long size, unsigned int *histo,
                                                        const int num_bin) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ unsigned int *histo_shared = new int[num_bin];
    for (int binIdx = threadIdx.x; binIdx < num_bin; binIdx += blockDim.x) {
        histo_shared[binIdx] = 0u;
    }

    syncthreads();

    unsigned int prev_bin_idx = -1;
    unsigned int accumulator = 0;

    for (unsigned int i = tid; i < size; i += blockDim.x * gridDim.x) {
        int offset = buffer[i] - 'a';
        if (offset >= 0 && offset < 26) {
            unsigned int curr_bin_idx = offset >> 2;
            if (curr_bin_idx != prev_bin_idx) {
                if (accumulator > 0) {
                    atomicAdd(&(histo_shared[prev_bin_idx]), accumulator);
                }
                accumulator = 1;
                prev_bin_idx = curr_bin_idx;
            } else {
                ++accumulator;
            }
        }
    }
    if (accumulator > 0) {
        atomicAdd(&(histo_shared[prev_bin_idx]), accumulator);
    }

    __syncthreads();

    for (int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
        atomicAdd(&(histo[binIdx]), histo_shared[binIdx]]);
    }
}


int main(int argc, char *argv[]) {
    return 0;
}