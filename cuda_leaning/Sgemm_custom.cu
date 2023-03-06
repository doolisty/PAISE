#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include "utils.h"

#define MATRIX_M 2048
#define MATRIX_N 2048
#define MATRIX_K 2048

#define FETCH_FLOAT4(arr) (reinterpret_cast<float4*>(&(arr))[0])
#define OFFSET(row, col, ld) (row * ld + col)

// In C: block_size = BM * BN, thread_size = TM * TN
template <
    const int BM,
    const int BN,
    const int BK,
    const int TM,
    const int TN,
    const bool ENABLE_PREFETCH
    >
__global__ void Sgemm(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int N,
    const int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int THREAD_X_PER_BLOCK = BN / TN; // = 16
    const int THREAD_Y_PER_BLOCK = BM / TM; // = 16
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    __shared__ float A_shared[2][BK][BM];  // 预先转置加速读取
    __shared__ float B_shared[2][BK][BN];

    float accum[TM][TN] = {0};

    float A_reg[2][TM];
    float B_reg[2][TN];

    const int A_ldg_num = (BM * BK) / (THREAD_NUM_PER_BLOCK * 4);
    const int B_ldg_num = (BN * BK) / (THREAD_NUM_PER_BLOCK * 4);

    // 数据预取需要用到
    float A_ldg_reg[4 * A_ldg_num];
    float B_ldg_reg[4 * B_ldg_num];

    const int A_TILE_THREAD_PER_ROW = BK / 4;
    const int B_TILE_THREAD_PER_ROW = BN / 4;

    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int A_TILE_COL_START = (tid % A_TILE_THREAD_PER_ROW) * 4;
    const int B_TILE_COL_START = (tid % B_TILE_THREAD_PER_ROW) * 4;

    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // 第一轮预取
    #pragma unroll
    for (int i = 0; i < BM; i += A_TILE_ROW_STRIDE) {
        int ldg_idx = (i / A_TILE_ROW_STRIDE) * 4;
        FETCH_FLOAT4(A_ldg_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
            by * BM + i + A_TILE_ROW_START,
            A_TILE_COL_START,
            K
        )]);
        A_shared[0][A_TILE_COL_START][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx];
        A_shared[0][A_TILE_COL_START + 1][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx + 1];
        A_shared[0][A_TILE_COL_START + 2][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx + 2];
        A_shared[0][A_TILE_COL_START + 3][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx + 3];
    }
    // if (bx == 0 && by == 0 && tx == 0)

    #pragma unroll
    for (int i = 0; i < BK; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(B_shared[0][B_TILE_ROW_START + i][B_TILE_COL_START])
            = FETCH_FLOAT4(B[OFFSET(
                i + B_TILE_ROW_START,
                bx * BN + B_TILE_COL_START,
                N
            )]);
    }
    __syncthreads();

    // if (bx == 0 && by == 0 && tx == 0 && ty == 0) {
    //     printf("%f\n", A_shared[0][0][0]);
    // }
    #pragma unroll
    for (int thread_y = 0; thread_y < TM; thread_y += 4) {
        FETCH_FLOAT4(A_reg[0][thread_y]) = FETCH_FLOAT4(A_shared[0][0][TM * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < TM; thread_x += 4) {
        FETCH_FLOAT4(B_reg[0][thread_x]) = FETCH_FLOAT4(B_shared[0][0][TN * tx + thread_x]);
    }

    int write_idx = 1;
    int tile_idx = 0;
    do {
        tile_idx += BK;
        if (tile_idx < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += A_TILE_ROW_STRIDE) {
                int ldg_idx = (i / A_TILE_ROW_STRIDE) * 4;
                FETCH_FLOAT4(A_ldg_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
                    by * BM + i + A_TILE_ROW_START,
                    tile_idx + A_TILE_COL_START,
                    K
                )]);
            }
            #pragma unroll
            for (int i = 0; i < BK; i += B_TILE_ROW_STRIDE) {
                int ldg_idx = (i / B_TILE_ROW_STRIDE) * 4;
                FETCH_FLOAT4(B_ldg_reg[ldg_idx]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + i + B_TILE_ROW_STRIDE,
                    bx * BK + B_TILE_COL_START,
                    N
                )]);
            }
        }

        int load_idx = write_idx ^ 1;

        #pragma unroll
        for (int j = 0; j < BK - 1; ++j) {
            #pragma unroll
            for (int thread_y = 0; thread_y < TM; thread_y += 4) {
                FETCH_FLOAT4(A_reg[(j+1)%2][thread_y]) = FETCH_FLOAT4(A_shared[load_idx][j+1][ty * TM + thread_y]);
            }
            #pragma unroll
            for (int thread_x = 0; thread_x < TN; thread_x += 4) {
                FETCH_FLOAT4(B_reg[(j+1)%2][thread_x]) = FETCH_FLOAT4(B_shared[load_idx][j+1][tx * TN + thread_x]);
            }
            #pragma unroll
            for (int thread_y = 0; thread_y < TM; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < TN; ++thread_x) {
                    accum[thread_y][thread_x] += A_reg[j%2][thread_y] * B_reg[j%2][thread_x];
                }
            }
        }

        if (tile_idx < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += A_TILE_ROW_STRIDE) {
                int ldg_idx = (i / A_TILE_ROW_STRIDE) * 4;
                A_shared[write_idx][A_TILE_COL_START][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx];
                A_shared[write_idx][A_TILE_COL_START + 1][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx + 1];
                A_shared[write_idx][A_TILE_COL_START + 2][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx + 2];
                A_shared[write_idx][A_TILE_COL_START + 3][i + A_TILE_ROW_START] = A_ldg_reg[ldg_idx + 3];
            }
            #pragma unroll
            for (int i = 0; i < BK; i += B_TILE_ROW_STRIDE) {
                int ldg_idx = (i / B_TILE_ROW_STRIDE) * 4;
                FETCH_FLOAT4(B_shared[write_idx][i + B_TILE_ROW_START][B_TILE_COL_START]) = FETCH_FLOAT4(B_ldg_reg[ldg_idx]);
            }
            __syncthreads();
            write_idx ^= 1;
        }

        #pragma unroll
        for (int thread_y = 0; thread_y < TM; thread_y += 4) {
            FETCH_FLOAT4(A_reg[0][thread_y]) = FETCH_FLOAT4(A_shared[load_idx ^ 1][0][ty * TM + thread_y]);
        }
        #pragma unroll
        for (int thread_x = 0; thread_x < TN; thread_x += 4) {
            FETCH_FLOAT4(B_reg[0][thread_x]) = FETCH_FLOAT4(B_shared[load_idx ^ 1][0][tx * TN + thread_x]);
        }

        #pragma unroll
        for (int thread_y = 0; thread_y < TM; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < TN; ++thread_x) {
                accum[thread_y][thread_x] += A_reg[1][thread_y] * B_reg[1][thread_x];
            }
        }
    } while (tile_idx < K);

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            FETCH_FLOAT4(C[OFFSET(
                by * BM + ty * TM + i,
                bx * BN + tx * TN + j,
                N
            )]) = FETCH_FLOAT4(accum[i][j]);
        }
    }
}


void cublas_GEMM(const float *A, const float *B, float *C) {
    cublasHandle_t handle;
    float alpha = 1.f, beta = 0.f;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublas handle init failed!\n";
        return;
    }

    double now = get_now();
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha, A, MATRIX_K, B, MATRIX_N, &beta, C, MATRIX_N);
    double elapsed = get_now() - now;
    std::cout << "cublas GEMM elapsed = " << elapsed << std::endl;

    cudaDeviceSynchronize();
    cublasDestroy(handle);
}


int main(int argc, char *argv[]) {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    dim3 grid_size(MATRIX_N / BN, MATRIX_M / BM);
    dim3 block_size(BN / TN, BM / TM);

    float *A, *B, *kernel_C, *cublas_C;

    cudaMallocManaged((void**)&A, MATRIX_M * MATRIX_K * sizeof(float));
    cudaMallocManaged((void**)&B, MATRIX_N * MATRIX_K * sizeof(float));
    cudaMallocManaged((void**)&kernel_C, MATRIX_M * MATRIX_N * sizeof(float));
    cudaMallocManaged((void**)&cublas_C, MATRIX_M * MATRIX_N * sizeof(float));

    for (int i = 0; i < MATRIX_M; ++i) {
        for (int j = 0; j < MATRIX_K; ++j) {
            A[i * MATRIX_K + j] = float(rand() & 0xff);
        }
    }
    for (int i = 0; i < MATRIX_K; ++i) {
        for (int j = 0; j < MATRIX_N; ++j) {
            B[i * MATRIX_N + j] = float(rand() & 0xff);
        }
    }

    cublasHandle_t handle;
    float alpha = 1.f, beta = 0.f;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublas handle init failed!\n";
        return;
    }

    double cublas_now = get_now();
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha, A, MATRIX_K, B, MATRIX_N, &beta, cublas_C, MATRIX_N);
    cudaDeviceSynchronize();
    double cublas_elapsed = get_now() - cublas_now;
    std::cout << "cublas GEMM elapsed = " << cublas_elapsed << std::endl;

    cublasDestroy(handle);

    double now = get_now();
    Sgemm<BM, BN, BK, TM, TN><<<grid_size, block_size>>>(A, B, kernel_C, M, N, K);
    cudaDeviceSynchronize();
    double elapsed = get_now() - now;
    std::cout << "custom GEMM elapsed = " << elapsed << std::endl;

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            // if (i < 16 && j < 16) {
            //     std::cout << cublas_C[OFFSET(i, j, MATRIX_N)] << " ";
            // }
            if (std::abs(kernel_C[OFFSET(i, j, MATRIX_N)] - cublas_C[OFFSET(i, j, MATRIX_N)]) > 1e-5) {
                std::cout << "not match on [" << i << "], [" << j << "]: kernel = " << kernel_C[OFFSET(i, j, MATRIX_N)]
                    << ", cublas = " << cublas_C[OFFSET(i, j, MATRIX_N)] << "\n";
            }
        }
        // if (i < 16) {
        //     std::cout << std::endl;
        // }
    }
}