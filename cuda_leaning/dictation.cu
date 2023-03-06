#include <cuda_runtime.h>

// 1. sgemm
#define FETCH_FLOAT4(arr) (reinterpret_cast<float4*>(&(arr))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void sgemm(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, const int M, const int K, const int N) {
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int x_thread_per_block = BN / TN;
    const int y_thread_per_block = BM / TM;
    const int thread_per_block = x_thread_per_block * y_thread_per_block;
    const int tid = ty * x_thread_per_block + tx;
    
    __shared__ float A_shared[BK][BM];
    __shared__ float B_shared[BK][BN];
    float A_reg[TM];
    float B_reg[TN];
    float C_tmp_res[TM][TN] = {0.0f};

    // 搬运 A 时的参数配置
    const int A_thread_per_row = BK / 4;
    const int A_tile_row = tid / A_thread_per_row;
    const int A_tile_col = (tid % A_thread_per_row) * 4;
    const int A_tile_stride_row = thread_per_block / A_thread_per_row;
    int A_ldg_round = BM / A_tile_stride_row;
    float A_ldg_reg[4 * A_ldg_round];

    // 搬运 B 时的参数配置
    const int B_thread_per_row = BN / 4;
    const int B_tile_row = tid / B_thread_per_row;
    const int B_tile_col = (tid % B_thread_per_row) * 4;
    const int B_tile_stride_row = thread_per_block / B_thread_per_row;

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        if (bx == 0) {
            #pragma unroll
            for (int i = 0; i < BM; i += A_tile_stride_row) {
                int ldg_idx = (i / A_tile_stride_row) * 4;
                FETCH_FLOAT4(A_ldg_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
                    by * BM + i + A_tile_row,
                    k + A_tile_col,
                    K
                )]);
                A_shared[A_tile_col][i + A_tile_row] = A_ldg_reg[ldg_idx];
                A_shared[A_tile_col + 1][i + A_tile_row] = A_ldg_reg[ldg_idx + 1];
                A_shared[A_tile_col + 2][i + A_tile_row] = A_ldg_reg[ldg_idx + 2];
                A_shared[A_tile_col + 3][i + A_tile_row] = A_ldg_reg[ldg_idx + 3];
            }
        }
    }
    if (by == 0) {
        #pragma unroll
        for (int i = 0; i < BK; i += B_tile_stride_row) {
            FETCH_FLOAT4(B_shared[i + B_tile_row][B_tile_col]) = FETCH_FLOAT4(B[OFFSET(
                k + i + B_tile_row,
                bx * BN + B_tile_col,
                N
            )]);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int tk = 0; tk < BK; ++tk) {
        if (tid < y_thread_per_block) {
            #pragma unroll
            for (int ti = 0; ti < TM; ti += 4) {
                
            }
        }
    }
}

// 1.1 sgemm (prefetch)

// 2. sgemv
template <const int ROW_PER_BLOCK>
__global__ void sgemv(float* __restrict__ A, float* __restrict__ x, float* __restrict__ y, const int M, const int N) {
    const int warp_size = 32;
    const int thread_lane = threadIdx.x & (warp_size - 1);
    const int curr_row = blockIdx.x * ROW_PER_BLOCK + threadIdx.x / warp_size;

    if (curr_row < M) {
        float tmp_sum = 0.0f;
        int k_iter = N / (warp_size * 4);
        if (k_iter == 0) k_iter = 1;
        A = &A[curr_row * N];

        for (int i = 0; i < k_iter; ++i) {
            int curr_elm_col = i * warp_size + thread_lane;
            if (curr_elm_col * 4 < N) {
                float4 A_val = reinterpret_cast<float4*>(A)[curr_elm_col];
                float4 x_val = reinterpret_cast<float4*>(x)[curr_elm_col];
                tmp_sum += A_val.x * x_val.x;
                tmp_sum += A_val.y * x_val.y;
                tmp_sum += A_val.z * x_val.z;
                tmp_sum += A_val.w * x_val.w;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int stride = 16; stride > 0; stride /= 2) {
            tmp_sum += __shfl_down_sync(0xffffffff, tmp_sum, stride);
        }
        if (thread_lane == 0) {
            y[curr_row] = tmp_sum;
        }
    }
}


// 2. sgemv
template <const int ROW_PER_BLOCK>
__global__ void sgemv(float* __restrict__ A, float* __restrict__ x, float* __restrict__ y, const int M, const int N) {
    const int warp_size = 32;
    const int thread_lane = threadIdx.x & (warp_size - 1);
    const int curr_row = blockIdx.x * ROW_PER_BLOCK + threadIdx.x / warp_size;

    if (curr_row < M) {
        int k_iter = N / warp_size;
        if (k_iter == 0) k_iter = 1;
        float tmp_sum = 0.0f;

        for (int i = 0; i < k_iter; ++i) {
            int curr_col_id = i * warp_size + thread_lane;
            if (curr_col_id * 4 < N) {
                float4 A_elm = reinterpret_cast<float4*>(&A[curr_row * N])[curr_col_id];
                float4 x_elm = reinterpret_cast<float4*>(&x[curr_col * 4])[0];
                tmp_sum += A_elm.x * x_elm.x;
                tmp_sum += A_elm.y * x_elm.y;
                tmp_sum += A_elm.z * x_elm.z;
                tmp_sum += A_elm.w * x_elm.w;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int stride = warp_size >> 1; stride > 0; stride >>= 1) {
            tmp_sum += __shfl_down_sync(0xffffffff, tmp_sum, stride);
        }

        if (thread_lane == 0) {
            y[curr_row] = tmp_sum;
        }
    }
}


// 3. spmv


// 5. reduction
template <const int TILE_SIZE>
__global__ void reduce(float* __restrict__ d_in, float* __restrict__ d_out, const int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * TILE_SIZE + tid;
    
    __shared__ float tile_shared[TILE_SIZE];
    if (idx < N) {
        tile_shared[tid] = d_in[idx];
    }
    __syncthreads();

    for (int stride = TILE_SIZE >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tile_shared[tid] += tile_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_out[blockIdx.x] = tile_shared[0];
    }
}



// 6. matrix transpose

// 7. softmax

// 8. histogram

// 9. mergesort

// 10. graph search