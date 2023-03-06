#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <int SECTION_SIZE, bool USE_INCLUSIVE>
__global__ void kogge_stone_scan_kernel(float *X, float *Y, const int input_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    __shared__ float X_shared[SECTION_SIZE];

    if (idx < input_size) {
        if (USE_INCLUSIVE) {
            X_shared[tid] = X[idx];
        } else {
            // exclusive
            if (tid > 0) {
                X_shared[tid] = X[idx - 1];
            } else {
                X_shared[tid] = 0;
            }
        }
    }

    for (unsigned int stride = 1; stride < SECTION_SIZE; stride *= 2) {
        __syncthreads();
        if (tid >= stride) {
            X_shared[tid] += X_shared[tid - stride];
        }
    }
    Y[idx] = X_shared[tid];
}

template <int SECTION_SIZE>
__global__ void brent_kung_scan_kernel(float *X, float *Y, const int input_size) {
    // 因为每个 block 最多用 SECTION_SIZE 一半大小的线程，于是每个 block 取两倍于线程数的数据
    const int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    __shared__ float X_shared[SECTION_SIZE];
    if (idx < input_size) {
        X_shared[tid] = X[idx];
    }
    if (idx + blockDim.x < input_size) {
        X_shared[tid + blockDim.x] = X[idx + blockDim.x];
    }

    // [STEP 1]：对 2 * stride - 1 位置上的元素做 reduce
    for (unsigned int stride = 1; stride < SECTION_SIZE; stride <<= 1) {
        __syncthreads();
#if 0
        // naive 写法，会出现更多的 control divergence
        if ((tid + 1) % (2 * stride) == 0) {
            X_shared[tid] += X_shared[tid - stride];
        }
#else
        // 从前往后顺序启动线程来做 reduce，divergence 会更少
        int index = (tid + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE) {
            X_shared[index] += X_shared[index - stride];
        }
#endif
    }

    // [STEP 2]：从整个序列的正中间开始，向后以递减 stride 作 distribution
    for (unsigned int stride = SECTION_SIZE >> 2; stride > 0; stride >>= 1) {
        __syncthreads();
        int index = (tid + 1) * 2 * stride - 1;
        if (index + stride < SECTION_SIZE) {
            X_shared[index + stride] += X_shared[index];
        }
    }

    __syncthreads();

    if (idx < input_size) {
        Y[idx] = X_shared[tid];
    }
    if (idx + blockDim.x < input_size) {
        Y[idx + blockDim.x] = X_shared[tid + blockDim.x];
    }
}

template <int SECTION_SIZE, int TILE_SIZE>
__global__ void three_phase_scan_kernel(float *X, float *Y, const int input_size) {
    const int block_start_idx = blockIdx.x * SECTION_SIZE;
    const int tid = threadIdx.x;
    __shared__ float X_shared[SECTION_SIZE];

#pragma unroll
    for (int i = 0; i < SECTION_SIZE; i += TILE_SIZE) {
        X_shared[tid + i] = X[block_start_idx + i];
    }

    __syncthreads();

    // [PHASE 1] 在 tile 内部做线性 scan
#pragma unroll
    for (int i = 1; i < TILE_SIZE; ++i) {
        int index = tid * TILE_SIZE + i;
        X_shared[index] += X_shared[index - 1];
    }

    // [PHASE 2] 在 tile 内部做 scan
#pragma unroll
    for (unsigned int stride = TILE_SIZE; stride < SECTION_SIZE; stride <<= 1) {
        __syncthreads();
        if (tid * TILE_SIZE >= stride) {
            int index = (tid + 1) * TILE_SIZE - 1;
            X_shared[index] += X_shared[index - stride];
        }
    }

    __syncthreads();

    // [PHASE 3] 把上一个 tile 的最后一个元素加到这个 tile 除了最后元素以外的其他元素上
    if (tid > 0) {
        float factor = X_shared[tid * TILE_SIZE - 1];
#pragma unroll
        for (int i = 0; i < TILE_SIZE - 1; ++i) {
            X_shared[tid * TILE_SIZE + i] += factor;
        }
    }

    __syncthreads();

    // write back
#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
        int index = tid * TILE_SIZE + i;
        if (index < input_size) {
            Y[block_start_idx + index] = X_shared[index];
        }
    }
}

/* ==== hierarchical scan starts ==== */

template <int SECTION_SIZE>
__global__ hierarchical_scan_step_1(float *X, float *Y, float *S, const int input_size) {
    const int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    __shared__ float X_shared[SECTION_SIZE];

    if (idx < input_size) {
        X_shared[tid] = X[idx];
    }
    if (idx + blockDim.x < input_size) {
        X_shared[tid + blockDim.x] = X[idx + blockDim.x];
    }

    for (unsigned int stride = 1; stride < SECTION_SIZE; stride <<= 1) {
        __syncthreads();
        int index = (tid + 1) * 2 * stride - 1;
        if (index < SECTION_SIZE) {
            X_shared[index] += X_shared[index - stride];
        }
    }
    for (unsigned int stride = SECTION_SIZE >> 2; stride > 0; stride >>= 1) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index + stride < SECTION_SIZE) {
            X_shared[index + stride] = X_shared[index];
        }
    }

    __syncthreads();

    if (tid == blockDim.x - 1) {
        S[blockIdx.x] = X_shared[SECTION_SIZE - 1];
    }
    if (idx < input_size) {
        Y[idx] = X_shared[tid];
    }
    if (idx + blockDim.x < input_size) {
        Y[idx + blockDim.x] = X_shared[tid + blockDim.x];
    }
}

// step 2 和普通的 scan kernel 一样，不写了

// thread_num = SECTION_SIZE
template <int SECTION_SIZE>
__global__ hierarchical_scan_step_3(float *S, float *Y, const int input_size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;

    if (blockIdx.x > 0) {
        __shared__ float factor = S[blockIdx.x - 1];
        Y[idx] += factor;
    }
}

/* ==== hierarchical scan ends ==== */

int main(int argc, char *argv[]) {
    return 0;
}