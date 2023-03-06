#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include "util.h"

//   C  =  A  *  B
// (M*N) (M*K) (K*N)
#define MATRIX_M 2048;
#define MATRIX_N 2048;
#define MATRIX_K 2048;

// #define BM 128
// #define BN 128
// #define BK 8
// #define TM 8
// #define TN 8

// ld = line distance?
#define OFFSET(row, col, ld) ((row) * (ld) + col)
#define FETCH_FLOAT4(arr) (std::reinterpret_cast<float4 *>(&(arr))[0])

// BM/TM 是 y 方向（高），BN/TN 是 X 方向（宽）
template<const int BM, const int BN, const int BK, const int TM, const int TN, const bool ENABLE_PREFETCH>
__global__ void GEMM(const float *A, const float *B, float *C) {
    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Thread number in Block's x/y direction
    const int THREAD_X_PER_BLOCK = BN / TN; // = 16
    const int THREAD_Y_PER_BLOCK = BM / TM; // = 16
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    // Thread id in current Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    // shared memory
    __shared__ float A_shared[2][BK][BM];  // 为了加速读取，需要转置一下
    __shared__ float B_shared[2][BK][BN];  // B不用转置
    
    // registers for temp result of C
    float accum[TM][TN] = {0};

    // registers for A and B in inner loop
    float A_reg[2][TM];
    float B_reg[2][TN];

    // registers for loading global memory to shared memory
    // A_ldg_num 计算方法：
    //      A 中一个 block 有 BM*BK 个数据，总共有 THREAD_NUM_PER_BLOCK (=256) 个线程
    //      因此每个线程搬运 BM*BK / THREAD_NUM_PER_BLOCK (=128*8/256 =4) 个数据
    //      既然每个线程只需要搬运 4 个数据，就用 float4 来搬运。
    //   注：一个线程搬运一个 float4 是比较好的配置，如果搬不完，一个线程可以搬多次（用 A_TILE_ROW_STRIDE）
    //      搬多少次，A_ldg_num 就是多少
    const int A_ldg_num = (BM * BK) / (THREAD_NUM_PER_BLOCK * 4);
    const int B_ldg_num = (BK * BN) / (THREAD_NUM_PER_BLOCK * 4);
    float A_ldg_reg[4 * A_ldg_num];
    float B_ldg_reg[4 * B_ldg_num];


    // 大迭代中搬运的线程安排：
    //      256 个线程搬运 128*8 个数据，每行由两个线程搬

    // Thread number in a single row
    const int A_TILE_THREAD_PER_ROW = BK / 4;
    const int B_TILE_THREAD_PER_ROW = BN / 4;

    // from which row/col current Thread need to work
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int A_TILE_COL_START = tid % A_TILE_THREAD_PER_ROW * 4;
    const int B_TILE_COL_START = tid % B_TILE_THREAD_PER_ROW * 4;

    // row stride, if 1 Thread have to load multiple lines in 1 tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    // 第一轮预取
    // load A from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BM; i += A_TILE_ROW_STRIDE) {
        int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(A_ldg_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
            by * BM + i + A_TILE_ROW_START,  // row
            A_TILE_COL_START,                // col
            MATRIX_K                         // ld, offset = row * ld + col
        )]);
        A_shared[0][A_TILE_COL_START][A_TILE_ROW_START + i] = A_ldg_reg[ldg_idx];
        A_shared[0][A_TILE_COL_START + 1][A_TILE_ROW_START + i] = A_ldg_reg[ldg_idx + 1];
        A_shared[0][A_TILE_COL_START + 2][A_TILE_ROW_START + i] = A_ldg_reg[ldg_idx + 2];
        A_shared[0][A_TILE_COL_START + 3][A_TILE_ROW_START + i] = A_ldg_reg[ldg_idx + 3];
    }
    // B 的第一轮可以直接赋值进来
    #pragma unroll
    for (int i = 0; i < BK; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(B_shared[0][B_TILE_ROW_START + i][B_TILE_COL_START])
            = FETCH_FLOAT4(B[OFFSET(
                i + B_TILE_ROW_START,
                bx * BN + B_TILE_COL_START,
                MATRIX_N
            )]);
    }
    __syncthreads();

    int write_idx = 1;
    int tile_idx = 0;
    do {
        // 大迭代预取：如果还有下一轮，继续预取
        // 注：只是预取到 ldg_reg 里面，后面还要放进 shared memory
        // load next tile from global memory
        tile_idx += BK;
        if (tile_idx < MATRIX_K) {
            #pragma unroll
            for (int i = 0; i < BM; i += A_TILE_ROW_STRIDE) {
                int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(A_ldg_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
                    by * BM + i + A_TILE_ROW_START,
                    tile_idx + A_TILE_COL_START,
                    MATRIX_K
                )]);
            }
            #pragma unroll
            for (int i = 0; i < BN; i += B_TILE_ROW_STRIDE) {
                int ldg_idx = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(B_ldg_reg[ldg_idx]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + i + B_TILE_ROW_START,
                    bx * BN + B_TILE_COL_START,
                    MATRIX_N
                )]);
            }
            // 注意，这里不需要__syncthreads()，因为取的是下一轮的，这轮用不到
        } // 大迭代预取结束

        // 第一轮小迭代预取
        int load_idx = write_idx ^ 1;
        #pragma unroll
        for (int thread_y = 0; thread_y < TM; thread_y += 4) {
            // load 的 idx 为 load_idx^1，因为 load_idx 是下一轮小迭代预取的 load idx，这一轮预取的显然是其相反
            FETCH_FLOAT4(A_reg[0][thread_y]) = FETCH_FLOAT(A_shared[load_idx^1][0][ty * TM + thread_y]);
        }
        #pragma unroll
        for (int thread_x = 0; thread_x < TN; thread_x += 4) {
            FETCH_FLOAT4(B_reg[0][thread_x]) = FETCH_FLOAT4(B_shared[load_idx^1][0][tx * TN + thread_x]);
        }
        
        // 小迭代的预取与计算，因为是预取，所以 j < BK - 1
        #pragma unroll
        for (int j = 0; j < BK - 1; ++j) {
            // load next tile from shared memory to register
            // load A_shared
            #pragma unroll
            for (int thread_y = 0; thread_y < TM; thread_y += 4) {
                FETCH_FLOAT4(A_reg[(j+1)%2][thread_y]) = FETCH_FLOAT4(
                    A_shared[load_idx][j+1][ty * TM + thread_y]
                );
            }
            // load B_shared
            #pragma unroll
            for (int thread_x = 0; thread_x < TN; thread_x += 4) {
                FETCH_FLOAT4(B_reg[(j+1)%2][thread_x]) = FETCH_FLOAT4(
                    B_shared[load_idx][j+1][tx * TN + thread_x]
                );
            }

            // compute
            #pragma unroll
            for (int thread_y = 0; thread_y < TM; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < TN; ++thread_x) {
                    // %2 取上一轮预取的数据，与这一轮预取数据的位置相反
                    accum[thread_y][thread_x] += A_reg[j%2][thread_y] * B_reg[j%2][thread_x];
                }
            }
        }

        // rewrite result from registers to shared memory
        if (tile_idx < MATRIX_K) {
            #pragma unroll
            for (int i = 0; i < BM; i += A_TILE_ROW_STRIDE) {
                int ldg_idx = i / A_TILE_ROW_STRIDE * 4;
                A_shared[write_idx][A_TILE_COL_START][i + A_TILE_ROW_START] = A_ldg_reg[idx];
                A_shared[write_idx][A_TILE_COL_START + 1][i + A_TILE_ROW_START] = A_ldg_reg[idx + 1];
                A_shared[write_idx][A_TILE_COL_START + 2][i + A_TILE_ROW_START] = A_ldg_reg[idx + 2];
                A_shared[write_idx][A_TILE_COL_START + 3][i + A_TILE_ROW_START] = A_ldg_reg[idx + 3];
            }
            #pragma unroll
            for (int i = 0; i < BN; i += B_TILE_ROW_STRIDE) {
                int ldg_idx = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(B_shared[write_idx][i + B_TILE_ROW_START][B_TILE_COL_START])
                    = FETCH_FLOAT4(B_ldg_reg[ldg_idx]);
            }
            __syncthreads();
            write_idx ^= 1;
        }

        // 最后一轮小迭代计算，单独完成
        #pragma unroll
        for (int thread_y = 0; thread_y < TM; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < TN; ++thread_x) {
                // 奇数轮读0写1，最后一轮为偶数轮，读1写0
                accum[thread_y][thread_x] += A_reg[1][thread_y] + B_reg[1][thread_x];
            }
        }
        
    } while (tile_idx < MATRIX_K);

    // write final result back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < TM; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < TN; thread_x += 4) {
            FETCH_FLOAT4(C[OFFSET(by * BM + ty * TM + thread_y, bx * BN + tx * TN + thread_y, MATRIX_N)])
                = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}

int main(int argc, char *argv[]) {
    timeval time;

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // grid(16, 16), block(16, 16)
    dim3 grid_size(MATRIX_M / BM, MATRIX_N / BN);
    dim3 block_size(BM / TM, BN / TN);

    float hA[MATRIX_M][MATRIX_K], hB[MATRIX_K][MATRIX_N], hC[MATRIX_M][MATRIX_N] = {0};

    // hA = (float*)malloc(MATRIX_M * MATRIX_K * sizeof(float));
    // hB = (float*)malloc(MATRIX_K * MATRIX_N * sizeof(float));
    // hC = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    for (int i = 0; i < MATRIX_M; ++i) {
        for (int j = 0; j < MATRIX_K; ++j) {
            hA[i][j] = (float)(rand() & 0xFF);
        }
    }
    for (int i = 0; i < MATRIX_K; ++i) {
        for (int j = 0; j < MATRIX_N; ++j) {
            hB[i][j] = (float)(rand() & 0xFF);
        }
    }

    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, MATRIX_M * MATRIX_K * sizeof(float));
    cudaMalloc((void**)&dB, MATRIX_K * MATRIX_N * sizeof(float));
    cudaMalloc((void**)&dC, MATRIX_M * MATRIX_N * sizeof(float));

    cudaMemcpy(dA, hA, MATRIX_M * MATRIX_K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, MATRIX_K * MATRIX_N * sizeof(float), cudaMemcpyHostToDevice);

    double now = get_now();
    GEMM<BM, BN, BK, TM, TN, true><<<grid_size, block_size>>>(dA, dB, dC);
    double elapsed = get_now() - now;

    

    cudaMemcpy(hC, dC, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}