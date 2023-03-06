#include <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2
#define O_TILE_WIDTH 16     //12
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
#define CLAMP(x) (min(max((x), 0.0), 1.0))
 
//@@ INSERT CODE HERE
__global__ void convolution_2d_kernel(float *I, const float* __restrict__ M, float *P,
                            int channels, int width, int height) {
   __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
   int i,j,k;
 
   int tx = threadIdx.x;
   int ty = threadIdx.y;
   int row_o = blockIdx.y*O_TILE_WIDTH + ty;
   int col_o = blockIdx.x*O_TILE_WIDTH + tx;
   int row_i = row_o - MASK_RADIUS;
   int col_i = col_o - MASK_RADIUS;
    
   for (k = 0; k < channels; k++) {
       if((row_i >=0 && row_i < height) && (col_i >=0 && col_i < width))
           Ns[ty][tx] = I[(row_i * width + col_i) * channels + k];
       else
           Ns[ty][tx] = 0.0f;
       
       __syncthreads();
       
       float output = 0.0f;
       if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){
         for(i = 0; i < MASK_WIDTH; i++) {
            for(j = 0; j < MASK_WIDTH; j++) {
               output += M[j * MASK_WIDTH + i] * Ns[i+ty][j+tx];
            
            }
         }
           
         if(row_o < height && col_o < width)
            P[(row_o * width + col_o) * channels + k] = CLAMP(output);
       
       }
       
       __syncthreads();
      // printf("kernel %f \n ",P[row_o * width + col_o]);
   }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    printf("imageChannels =%d\n", imageChannels);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 dimGrid(ceil((float)imageWidth/O_TILE_WIDTH), ceil((float)imageHeight/O_TILE_WIDTH));
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    convolution_2d_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData,
                                       imageChannels, imageWidth, imageHeight);
    cudaDeviceSynchronize(); // note this 
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}