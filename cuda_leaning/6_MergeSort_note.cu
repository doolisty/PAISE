#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void merge_sequential(const int *A, int m, const int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] < B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {
        while (j < n) {
            C[k++] = B[j++];
        }
    } else {
        while (i < m) {
            C[k++] = A[i++];
        }
    }
}

int co_rank(int k, int *A, int m, int *B, int n) {
    int i = k < m ? k : m; // i 初始化为最右边，即 right
    int j = k - i;
    int i_min = k-n > 0 ? k-n : 0; // B 都选进去够 k 个吗？不够就最少选 k-n 个，否则可能一个都不选
    int j_min = k-m > 0 ? k-m : 0;
    bool active = true;
    while (active) {
        // 退出条件：A[i-1] <= B[j] && B[j-1] < A[i]（哪个取等号都可以）
        if (i >= 0 && j < n && A[i-1] > B[j]) {
            int delta = (i - i_min + 1) >> 1;
            j_min = j;
            j += delta;
            i -= delta;
        } else if (i < m && j >= 0 && B[j-1] >= A[i]) {
            int delta = (j - j_min + 1) >> 1;
            i_min = i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__global__ void merge_basic_kernel(const int *A, int m, const int *B, int n, int *C) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_curr = tid * ceil((m + n) / (gridDim.x * blockDim.x));
    int k_next = min(m+n, (tid + 1) * ceil((m + n) / (gridDim.x * blockDim.x)));
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k - i_curr;
    int j_next = k - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

template <int TILE_SIZE>
__global__ void merge_tiled_kernel(const int *A, int m, const int *B, int n, int *C) {
    extern __shared__ int AB_shared[];
    int *A_shared = &AB_shared[0];
    int *B_shared = &AB_shared[TILE_SIZE];
    int C_curr = blockIdx.x * ceil((m + n) / gridDim.x); // 每个 block 处理一个 tile
    int C_next = min(m+n, (blockIdx.x + 1) * ceil((m + n) / gridDim.x);

    if (threadIdx.x == 0) {
        // 由于对第 0 个 block 而言 C_curr == 0，所以实际上第 0 个 block 的 A_curr 和 B_curr 都是 0
        A_shared[0] = co_rank(C_curr, A, m, B, n);
        A_shared[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();
    int A_curr = A_shared[0];
    int A_next = A_shared[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    
    int iter_cnt = 0;
    int C_len = C_next - C_curr;
    int A_len = A_next - A_curr;
    int B_len = B_next - B_curr;

    int total_iter = (C_len + TILE_SIZE - 1) / TILE_SIZE;
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    // 大循环一次完成一个 tile 的搬运和排序，书里的例子是 1024 个元素，一共搬完一个 block 负责的 4000 个元素
    while (iter_cnt < total_iter) {
        // [STEP 1] 搬运一个 tile 到 shared memory 中
        // 小循环一次搬 blockDim.x 个元素，例子里是 128 个
        for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
            if (i < A_length - A_consumed) {
                A_shared[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < TILE_SIZE; i += blockDim.x) {
            if (i < B_len - B_consumed) {
                B_shared[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
    }
    __syncthreads();
}

int main(int argc, char *argv[]) {
    return 0;
}