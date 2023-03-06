#include <cuda_runtime.h>

#define FETCH_FLOAT4(arr) (reinterpret_cast<float4*>(&(arr))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + col)

template <const int BM, const int BK, const int BN, const int TM, const int TN>
__global__ void sgemm_dictation(const float *A, const float *B, float *C, const int M, const int K, const int N) {
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int x_thread_num_per_block = BN / TN;
    const int y_thread_num_per_block = BM / TM;
    const int thread_num_per_block = x_thread_num_per_block * y_thread_num_per_block;
    const int tid = ty * x_thread_num_per_block + tx;

    __shared__ float A_shared[BK][BM];
    __shared__ float B_shared[BK][BN];
    float A_reg[TM];
    float B_reg[TN];
    float C_tmp_res[TM][TN];

    // 搬运：global memory -> shared memory
    const int A_thread_num_per_row = BK / 4;
    const int A_tile_row = tid / A_thread_num_per_row;
    const int A_tile_col = (tid % A_thread_num_per_row) * 4;
    const int A_tile_stride_row = thread_num_per_block / A_thread_num_per_row;

    const int B_thread_num_per_row = BN / 4;
    const int B_tile_row = tid / B_thread_num_per_row;
    const int B_tile_col = (tid % B_thread_num_per_row) * 4;
    const int B_tile_stride_row = thread_num_per_block / B_thread_num_per_row;

    const int A_ldg_round = BM * BK / (thread_num_per_block * 4);
    float A_ldg_reg[4 * A_ldg_round];

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += A_tile_stride_row) {
            int ldg_idx = (i / A_tile_stride_row) * 4;
            FETCH_FLOAT4(A_ldg_reg[ldg_idx]) = FETCH_FLOAT4(A[OFFSET(
                by * BM + i + A_tile_row,
                k * BK + A_tile_col,
                K
            )]);
            A_shared[A_tile_col][i + A_tile_row] = A_ldg_reg[ldg_idx];
            A_shared[A_tile_col + 1][i + A_tile_row] = A_ldg_reg[ldg_idx + 1];
            A_shared[A_tile_col + 2][i + A_tile_row] = A_ldg_reg[ldg_idx + 2];
            A_shared[A_tile_col + 3][i + A_tile_row] = A_ldg_reg[ldg_idx + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += B_tile_stride_row) {
            FETCH_FLOAT4(B_shared[B_tile_row][B_tile_col]) = FETCH_FLOAT4(B[OFFSET(
                k * BK + i + B_tile_row,
                bx * BN + B_tile_col,
                N
            )]);
        }
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < BK; ++tk) {
            #pragma unroll
            for (int ti = 0; ti < TM; ti += 4) {
                FETCH_FLOAT4(A_reg[ti]) = FETCH_FLOAT4(A_shared[tk][ty * TM + ti]);
            }
            #pragma unroll
            for (int ti = 0; ti < TN; ti += 4) {
                FETCH_FLOAT4(B_reg[ti]) = FETCH_FLOAT4(B_shared[tk][tx * TN + ti]);
            }
            __syncthreads();

            #pragma unroll
            for (int ti = 0; ti < TM; ++ti) {
                #pragma unroll
                for (int tj = 0; tj < TN; ++tj) {
                    C_tmp_res[ti][tj] += A_reg[ti] * B_reg[tj];
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int ti = 0; ti < TM; ++ti) {
        #pragma unroll
        for (int tj = 0; tj < TN; tj += 4) {
            FETCH_FLOAT4(C[OFFSET(
                by * BM + ty * TM + ti,
                bx * BN + tx * TN + tj,
                N
            )]) = FETCH_FLOAT4(C_tmp_res[ti][tj]);
        }
    }
}

int main(int argc, char *argv[]) {
    const int bm = 128, bn = 128, bk = 8, rm = 8, rn = 8;
    dim3 grid_size(MATRIX_M / bm, MATRIX_N / bn);
    dim3 block_size(bm/rm, bn/rn);

    const float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    size_t mat_size = MATRIX_M * MATRIX_N * sizeof(float);

    hA = malloc(mat_size);
    hB = malloc(mat_size);
    hC = malloc(mat_size);
    return 0;
}