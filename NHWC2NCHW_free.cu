#include <stdio.h>

__global__ void transpose(unsigned char *odata, const unsigned char *idata,
                        int H, int W)
{
    int N = gridDim.y; // batch size
    int n = blockIdx.y; // batch number
    int C = gridDim.z; // channel 
    int c = blockIdx.z; // channel number
    long idx = n * blockDim.x * gridDim.x * C + 
               threadIdx.x * gridDim.x * C +
               blockIdx.x * C+
               c;

    int img_coor = idx % (H*W*C); //coordinate of one image, not idx of batch image
    int h = img_coor / (W*C); // dst idx 
    int w = img_coor % (W*C)/C; // dst idx
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
    
    int BATCH = 10;
    int HEIGHT = 50;
    int WIDTH = 50;
    int C = 3;
    int SIZE = HEIGHT * WIDTH * C;

    cudaStream_t stream1;
    cudaStreamCreate ( &stream1) ;

    dim3 dimBlock(1024, 1, 1);
    dim3 dimGrid(int(SIZE/C/1024)+1,BATCH,C);

    // init host array
    unsigned char host[SIZE*BATCH];

    // init src image
    for(int i = 0; i < SIZE*BATCH; i++){
        // host_src[i] = i+1;
        host[i] = (i%C);
    }

    for(int i = 0; i < 30*3; i+=3){ // N H W C
        printf("%d\n",host[i]);
    }
    printf("============================\n");

    // init device array
    unsigned char *device_src, *device_dst;
	cudaMalloc((unsigned char **)&device_src, SIZE* BATCH* sizeof(unsigned char));
    cudaMalloc((unsigned char **)&device_dst, SIZE* BATCH* sizeof(unsigned char));
    
	cudaMemcpy(device_src , host , SIZE * BATCH * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // run kernel
    transpose<<<dimGrid, dimBlock, 0, stream1>>>(device_dst, device_src, HEIGHT, WIDTH);
    cudaDeviceSynchronize();
    
    // take out output
    cudaMemcpy(host, device_dst, SIZE * BATCH * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // DEBUG : print first image in batch , first 30 pixel in 3 channels.


 
    for(int n = 0; n<SIZE*BATCH ; n+=SIZE){
        for(int c = 0; c<SIZE ; c+=HEIGHT*WIDTH){ // N C H W
            for(int i = 0 ; i < 10; i++){
                printf("batch: %d, idx: %d, count: %d, value: %d\n", n/SIZE, n+c+i, i, host[n+c+i]);
            }
        }
        printf("------------------------------\n");
    }


    // deinit GPU
	cudaFree(device_src);
	cudaFree(device_dst);

    return 0;
}
// clear && clear && nvcc NHWC2NCHW_free.cu -o trans.o && ./trans.o