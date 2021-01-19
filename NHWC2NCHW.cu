#include <stdio.h>

__global__ void transpose(unsigned char *odata, const unsigned char *idata)
{
    int H = blockDim.x * gridDim.x; // # dst_height
    int W = blockDim.y * gridDim.y; // # dst_width 
    int h = blockDim.x * blockIdx.x + threadIdx.x;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # x / h-th row
    int w = blockDim.y * blockIdx.y + threadIdx.y;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # y / w-th col
    int C = 3; // # ChannelDim
    int c = blockIdx.z % 3 ; // [0,2] # ChannelIdx
    int n = blockIdx.z / 3 ; // [0 , Batch size-1], # BatchIdx

    long src_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    long dst_idx = n * (C * H * W) +
                    c * (H * W)+
                    h * W+
                    w;

    odata[dst_idx] = idata[src_idx];
}

int main(){
    // dim3 dimBlock(32,32,1);  << Max total is 1024 , so , x=32 ,y=32 ,  some one use 1024 to handle flatten tensor is fine.
    // dim3 dimGrid(19,19,3); << x = 608 / 32 = 19  , same on y , z = channel * batch_size, assume channel = 3. 
    dim3 dimBlock(32,32,1);
    dim3 dimGrid(19,19,3);

    // init host array
    unsigned char host_src[608*608*3]; // N H W C
    // unsigned char host_dst[1108992];
    unsigned char host_dst[608*608*3]; // N C H W

    // init src image
    for(int i = 0; i < 608*608*3; i++){
        // host_src[i] = i+1;
        host_src[i] = (i%3);
    }

    // init device array
    unsigned char *device_src, *device_dst;
	cudaMalloc((unsigned char **)&device_src, 608*608*3* sizeof(unsigned char));
    cudaMalloc((unsigned char **)&device_dst, 608*608*3* sizeof(unsigned char));
    
	cudaMemcpy(device_src , host_src , 608*608*3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // run kernel
    transpose<<<dimGrid, dimBlock>>>(device_dst, device_src);
    cudaDeviceSynchronize();
    
    // take out output
    cudaMemcpy(host_dst, device_dst, 608*608*3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // DEBUG : print first image in batch , first 30 pixel in 3 channels.

    for(int i = 0; i < 30*3; i+=3){ // N H W C
        printf("%d\n",host_src[i]);
    }
    printf("============================\n");
 
    for(int c = 0; c<3*608*608 ; c+=608*608){ // N C H W
        for(int i = 0 ; i < 30; i++){
            printf("%d %d %d\n", c+i, i, host_dst[c+i]);
        }
        printf("------------------------------\n");
    }


    // deinit GPU
	cudaFree(device_src);
	cudaFree(device_dst);

    return 0;
}
// clear && clear && nvcc NHWC2NCHW.cu -o trans.o && ./trans.o
