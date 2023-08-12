#include <stdio.h>
#include <iostream>
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* const file,
    const int line)
    {
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            // We don't exit when we encounter CUDA errors in this example.
            // std::exit(EXIT_FAILURE);
        }
    }

#define MAX_WIDTH 7680 // 7680 3840 1920
__global__ void tile_check(unsigned char* device_src)
{
    // int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ uchar3 srcTile[2][MAX_WIDTH];  // cache 2rows for 1 dst pixel
    for( int w = threadIdx.x ; w < MAX_WIDTH ; w+=blockDim.x){
        for (int row = 0; row < 2; row++){
            srcTile[row][w].x = 2;
            srcTile[row][w].y = 3;
            srcTile[row][w].z = 4;
        }
    }
    __syncthreads();
    int x = 1;
    // printf("x: %d\n", srcTile[0][x].x);
    // printf("sizeof(srcTile): %ld, %ld , %ld , %ld, %ld\n", sizeof(srcTile) , sizeof(srcTile[0]) , sizeof(srcTile[0][0]), sizeof(uchar3), sizeof(unsigned char));
}

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
//   for (int i = 0; i < nDevices; i++) {
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, i);
//     printf("Device Number: %d\n", i);
//     printf("  Device name: %s\n", prop.name);
//     printf("  Memory Clock Rate (KHz): %d\n",
//            prop.memoryClockRate);
//     printf("  Memory Bus Width (bits): %d\n",
//            prop.memoryBusWidth);
//     printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
//            2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
//     printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
//     printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
//     printf("  Max Registers per Block: %d\n", prop.regsPerBlock);
//     printf("  Shared Memory per Block: %ld\n", prop.sharedMemPerBlock);
//     printf("  Total Constant Memory: %ld\n", prop.totalConstMem);
//     printf("  Memory Pitch: %ld\n", prop.memPitch);
//     }

dim3 dimBlock(1024, 1,1); // maximum threads: 1024
dim3 dimGrid(1920, 50,1);
int SRC_SIZE = 1920*1080*50;
int DST_SIZE = 20*20*50;

// printf("%d\n", SRC_SIZE);

unsigned char *host_src = (unsigned char *) malloc(sizeof(unsigned char) * SRC_SIZE);
unsigned char *host_dst = (unsigned char *) malloc(sizeof(unsigned char) * DST_SIZE);

// init src image
for(int i = 0; i < SRC_SIZE; i++){
    host_src[i] = 1;
}
unsigned char *device_src, *device_dst;
CHECK_CUDA_ERROR(cudaMalloc((unsigned char **)&device_src, SRC_SIZE* sizeof(unsigned char)));
CHECK_CUDA_ERROR(cudaMalloc((unsigned char **)&device_dst, DST_SIZE* sizeof(unsigned char)));

CHECK_CUDA_ERROR(cudaMemcpy(device_src , host_src , SRC_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice));

tile_check<<<dimGrid, dimBlock, 0>>>(device_src);

free(host_src);
free(host_dst);
cudaFree(device_src);
cudaFree(device_dst);
return 0;
}



// struct cudaDeviceProp {
//     char name[256];
//     size_t totalGlobalMem;
//     size_t sharedMemPerBlock;
//     int regsPerBlock;
//     int warpSize;
//     size_t memPitch;
//     int maxThreadsPerBlock;
//     int maxThreadsDim[3];
//     int maxGridSize[3];
//     size_t totalConstMem;
//     int major;
//     int minor;
//     int clockRate;
//     size_t textureAlignment;
//     int deviceOverlap;
//     int multiProcessorCount;
//     int kernelExecTimeoutEnabled;
//     int integrated;
//     int canMapHostMemory;
//     int computeMode;
//     int concurrentKernels;
//     int ECCEnabled;
//     int pciBusID;
//     int pciDeviceID;
//     int tccDriver;
// }

// nvcc stat.cu -o stat.o && ./stat.o