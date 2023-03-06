#include <cuda_runtime.h>

#define FETCH_FLOAT4(arr) (reinterpret_cast<float4*>(&(arr))[0])

__global__ void add(float * __restrict__ a, float * __restrict__ b, float* c) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}

__global__ void vec4_add(float *a, float *b, float *c) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(c[idx]) = reg_c;
}

// 列数 n = 32 的矩阵进行矩阵向量乘（gemv）
template <unsigned int WARP_SIZE>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WARP_SIZE >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16);
}
