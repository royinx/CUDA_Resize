#include <stdio.h>

__device__ float lerp1d(int a, int b, float w)
{
    if(b>a){
        return a + w*(b-a);
    }
    else{
        return b + w*(a-b);
    }
}

__device__ float lerp2d(int f00, int f01, int f10, int f11,
                        float centroid_h, float centroid_w )
{
    centroid_w = (1 + lroundf(centroid_w) - centroid_w)/2;
    centroid_h = (1 + lroundf(centroid_h) - centroid_h)/2;
    
    float r0, r1, r;
    r0 = lerp1d(f00,f01,centroid_w);
    r1 = lerp1d(f10,f11,centroid_w);

    r = lerp1d(r0, r1, centroid_h); //+ 0.00001
    // printf("re: %f, %f | %f, %f | %f, %f | %f | %d, %d, %d, %d \n", centroid_x , centroid_y, centroid_x_re, centroid_y_re, r0, r1, r, f00, f01, f10, f11);
    return r;
}


__global__ void GPU_validation(void)
{
    printf("GPU has been activated \n");
}

__global__ void cuRESIZE(unsigned char* src_img, unsigned char* dst_img, 
    int src_h, int src_w, 
    int dst_h, int dst_w,
    float stride_h, float stride_w)
{
    /* 
    Input: 
        src_img - NHWC
        channel C, default = 3 
    
    Output:
        dst_img - NHWC

    */

    int N = gridDim.y; // batch size
    int n = blockIdx.y; // batch number
    int C = gridDim.z; // channel 
    int c = blockIdx.z; // channel number
    long idx = n * blockDim.x * gridDim.x * C + 
              threadIdx.x * gridDim.x * C +
              blockIdx.x * C+
              c;
    if (idx >= dst_h* dst_w * C * N){return;} // some overhead threads in each image process, 
    /*
    Now implementation : 
    ( (1024 * int(DST_SIZE/3/1024)+1) - (src_h * src_w) )* N
    = overhead * N times
    
    to do: put the batch into gridDim.x
    dim3 dimGrid(int(DST_SIZE*batch/3/1024)+1,1,3);

    */
    int H = dst_h;
    int W = dst_w;

    int img_coor = idx % (dst_h*dst_w*C); //coordinate of one image, not idx of batch image
    int h = img_coor / (W*C); // dst idx 
    int w = img_coor % (W*C)/C; // dst idx

    float centroid_h, centroid_w;  
    centroid_h = stride_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = stride_w * (w + 0.5); // 

    // unsigned long = 4,294,967,295 , up to (1080p,RGB)*600 imgs
    unsigned long f00,f01,f10,f11;

    int src_h_idx = lroundf(centroid_h)-1;
    int src_w_idx = lroundf(centroid_w)-1;
    if (src_h_idx<0){src_h_idx=0;}
    if (src_w_idx<0){src_w_idx=0;}
    // printf("h:%d w:%d\n",src_h_idx,src_w_idx);
    // printf("src_h_idx:%d , h: %d | src_w_idx:%d , w: %d\n",src_h_idx,h,src_w_idx,w);

    // idx = NHWC = n*(HWC) + h*(WC) + w*C + c;
    f00 = n * src_h * src_w * C + 
          src_h_idx * src_w * C + 
          src_w_idx * C +
          c;
    f01 = n * src_h * src_w * C +
          src_h_idx * src_w * C +
          (src_w_idx+1) * C +
          c;
    f10 = n * src_h * src_w * C +
          (src_h_idx+1) * src_w * C +
          src_w_idx * C +
          c;
    f11 = n * src_h * src_w * C + 
          (src_h_idx+1) * src_w * C +
          (src_w_idx+1) * C +
          c;
          
    // bool bl_a = (f01 == (f00 + 3));
    // bool bl_b = (f10 == (f00 + src_w * 3));
    // bool bl_c = (f11 == (f00 + src_w * 3 + 3));
    // printf("%d, %d, %d | %d, %d, %d | %d\n", bl_a,bl_b,bl_c, f01-f00, f10-f00,f11-f00, src_w);
    // printf("%d, %d, %d, %d\n", src_img[f00],src_img[f01],src_img[f10],src_img[f11]);
    // printf("%ld, %ld, %ld, %ld\n", f00, f01,f10,f11);

    // printf("GPU has been activated, %d, %d, %d\n", idx, idx/W, idx%W);
    if (f00 >= src_h_idx * src_w * C )
    {
        if (src_h_idx >= 1080){
            printf("%d, %d, %d, %d | %ld\n", n * src_h * src_w * C , 
                                            src_h_idx * src_w * C, 
                                            src_w_idx * C ,
                                            c,
                                            f00);
            printf("%d, cent: %.6f, stride: %.6f, h:%d\n", src_h_idx, centroid_h, stride_h,h);
        }
    }
    // printf("%ld\n",idx);
    // printf("%d, %d, %d, %d \n", src_img[f00], src_img[f01], src_img[f10], src_img[f11]);
    printf("%d\n",h);
    if (h>600){
        printf("src_h_idx:%d , h: %d | src_w_idx:%d , w: %d\n",src_h_idx,h,src_w_idx,w);
        printf("%d, %d, %d, %d \n", src_img[f00], src_img[f01], src_img[f10], src_img[f11]);
        // printf("%d, %d, %d, %d\n", f00,,,);
    }
    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
                    centroid_h, centroid_w));
    // int rs = idx%256;
    // int rs = c;

    // printf("rs: %d | centroid: h:%f, w:%f | h: %d, w: %d | %d, %d, %d , %d | %d, %d | %d, %d, %d, %d \n", rs, centroid_h, centroid_w, src_h_idx, src_w_idx, f00,f01,f10,f11, C, c, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);
    // printf("rs: %d | stride h: %f , w: %f  | centroid: h:%f, w:%f| h: %d, w: %d | %d, %d, %d , %d | %d, %d | %d, %d, %d, %d \n", rs, stride_h, stride_w, centroid_h, centroid_w, src_h_idx, src_w_idx, f00,f01,f10,f11, C, c, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);

    dst_img[idx] = (unsigned char)rs;
}

__global__ void tester(unsigned char* src_img, unsigned char* dst_img, 
                       int src_h, int src_w, 
                       int dst_h, int dst_w,
                       float stride_h, float stride_w)
{
    int H = blockDim.x * gridDim.x; // # dst_DST_HEIGHT
    int W = blockDim.y * gridDim.y; // # dst_DST_WIDTH 
    int h = blockDim.x * blockIdx.x + threadIdx.x;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # x / h-th row
    int w = blockDim.y * blockIdx.y + threadIdx.y;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # y / w-th col
    int C = 3; // # ChannelDim
    int c = blockIdx.z % 3 ; // [0,2] # ChannelIdx
    int n = blockIdx.z / 3 ; // [0 , Batch size-1], # BatchIdx
    int N = gridDim.z / 3 ;
    
    // printf("%d(%d), %d(%d), %d(%d), %d(%d) \n",n,N,c,C,h,H,w,W);
    printf("%d, %d, %d | %d, %d, %d \n",blockDim.x,blockDim.y,blockDim.z,gridDim.x,gridDim.y,gridDim.z);
    // idx = NHWC = n*(HWC) + h*(WC) + w*C + c;
    int idx = n * (H * W * C) + 
              h * (W * C) +
              w * C +
              c;
    
    // idx = NCHW = n*(CHW) + c*(HW) + h*W + w
    // int idx = n * (C * H * W) +
    //           c * (H * W)+
    //           h * W+
    //           w;

    // int idx = x * blockDim.y * gridDim.y * gridDim.z + y * gridDim.z + z; // x * 608(DST_WIDTH) * 3(channel) + y * 3(channel) + [0,2]
    
    float centroid_h, centroid_w;  
    centroid_h = stride_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = stride_w * (w + 0.5); // 

    int f00,f01,f10,f11;

    int src_h_idx = lroundf(centroid_h)-1;
    int src_w_idx = lroundf(centroid_w)-1;
    if (src_h_idx<0){src_h_idx=0;}
    if (src_w_idx<0){src_w_idx=0;}
    // printf("h:%d w:%d\n",src_h_idx,src_w_idx);

    // idx = NHWC = n*(HWC) + h*(WC) + w*C + c;
    f00 = n * src_h * src_w * C + 
          src_h_idx * src_w * C + 
          src_w_idx * C +
          c;
    f01 = n * src_h * src_w * C +
          src_h_idx * src_w * C +
          (src_w_idx+1) * C +
          c;
    f10 = n * src_h * src_w * C +
          (src_h_idx+1) * src_w * C +
          src_w_idx * C +
          c;
    f11 = n * src_h * src_w * C + 
          (src_h_idx+1) * src_w * C +
          (src_w_idx+1) * C +
          c;
          
    // bool bl_a = (f01 == (f00 + 3));
    // bool bl_b = (f10 == (f00 + src_w * 3));
    // bool bl_c = (f11 == (f00 + src_w * 3 + 3));
    // printf("%d, %d, %d | %d, %d, %d | %d\n", bl_a,bl_b,bl_c, f01-f00, f10-f00,f11-f00, src_w);



    // printf("h: %d, w: %d | %d, %d, %d , %d | %d, %d | %d, %d, %d, %d \n", src_h_idx, src_w_idx, f00,f01,f10,f11, C, c, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);

    
    // lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11],);
    // printf("%d, %d | %d, %d, %d, %d \n", src_h_idx, src_w_idx, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);

    // float temp = lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
    //                     centroid_y, centroid_x);
    // printf("z: %d | %f, %f | %f | %d, %d, %d, %d \n", z, centroid_x, centroid_y, temp, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);
    // printf("%f",temp);


    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
                            centroid_h, centroid_w));
    // printf("rs: %d | centroid: h:%f, w:%f | h: %d, w: %d | %d, %d, %d , %d | %d, %d | %d, %d, %d, %d \n", rs, centroid_h, centroid_w, src_h_idx, src_w_idx, f00,f01,f10,f11, C, c, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);
    // printf("rs: %d | stride h: %f , w: %f  | centroid: h:%f, w:%f| h: %d, w: %d | %d, %d, %d , %d | %d, %d | %d, %d, %d, %d \n", rs, stride_h, stride_w, centroid_h, centroid_w, src_h_idx, src_w_idx, f00,f01,f10,f11, C, c, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);
    // printf("z: %d | %f, %f | %d | %d, %d, %d, %d \n", z, centroid_x, centroid_y, rs, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);

    dst_img[idx] = (unsigned char)rs;
}

int main(){
    int SRC_HEIGHT = 20;
    int SRC_WIDTH = 20;
    int SRC_SIZE = SRC_HEIGHT * SRC_WIDTH * 3;

    int DST_HEIGHT = 40;
    int DST_WIDTH = 40;
    int DST_SIZE = DST_HEIGHT * DST_WIDTH * 3;

    int batch = 1;
    
    dim3 dimBlock(1024,1,1); // maximum threads: 1024
    dim3 dimGrid(int(DST_SIZE/3/1024)+1,batch,3);

    // dim3 dimBlock(32,32,1);  << Max total is 1024 , so , x=32 ,y=32 ,  some one use 1024 to handle flatten tensor is fine.
    // dim3 dimGrid(19,19,3); << x = 608 / 32 = 19  , same on y , z = channel * batch_size, assume channel = 3. 
        
    unsigned char host_src[SRC_SIZE];
    // unsigned char host_dst[1108992];
    unsigned char host_dst[DST_SIZE];

    // init src image
    for(int i = 0; i < SRC_SIZE; i++){
        host_src[i] = i+1;
        // host_src[i] = (i%3);
    }

    float stride_h = (float)SRC_HEIGHT / DST_HEIGHT;
    float stride_w = (float)SRC_WIDTH / DST_WIDTH;

    unsigned char *device_src, *device_dst;
	cudaMalloc((unsigned char **)&device_src, SRC_SIZE* sizeof(unsigned char));
    cudaMalloc((unsigned char **)&device_dst, DST_SIZE* sizeof(unsigned char));
    
	cudaMemcpy(device_src , host_src , SRC_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);

    GPU_validation<<<1,1>>>();
    cudaDeviceSynchronize();

    dim3 dimBlock2(1024, 1,1); // maximum threads: 1024
    dim3 dimGrid2(int(DST_SIZE/3/1024)+1,batch,3);
    cuRESIZE<<<dimGrid2,dimBlock2>>>(device_src, device_dst, 
        SRC_HEIGHT, SRC_WIDTH,
        DST_HEIGHT, DST_WIDTH,
        stride_h, stride_w);

    cudaDeviceSynchronize();


    // for(int i = 0; i<10; i++){
    // tester<<<dimGrid, dimBlock>>>(device_src, device_dst, 
    //                               SRC_HEIGHT, SRC_WIDTH,
    //                               stride_h, stride_w);
    // cudaDeviceSynchronize();
    // }
    
    cudaMemcpy(host_dst, device_dst, DST_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // DEBUG : print first image in batch , first 30 pixel in 3 channels.

    // for(int i = 0; i < 30*3; i+=3){ // NHWC
    //     printf("%d\n",host_src[i]);
    // }
    printf("============================\n");
 
    // for(int c = 0; c<3*DST_HEIGHT*DST_WIDTH ; c+=DST_HEIGHT*DST_WIDTH){ // if NCHW
    //     for(int i = 0 ; i < 30; i++){
    //         printf("%d %d %d\n", c+i, i, host_dst[c+i]);
    //     }
    //     printf("------------------------------\n");
    // }

    // print first 30 elements from each chanel
    // for(int c = 0; c<3; c++){ // NHWC
    //     for(int i = 0 ; i < 30; i++){
    //         int idx = i*3 +c;
    //         printf("%d %d %d\n", c+i*3, i, host_dst[idx]);
    //     }
    //     printf("------------------------------\n");
    // }

    // int count_0=0;
    // int count_1=0;
    // int count_2=0;
    // for(int idx = 0; idx<sizeof(host_dst)/sizeof(unsigned char); idx++){ // NHWC
    //     printf("%d %d\n", idx, host_dst[idx]);
    //     if (host_dst[idx]==0){count_0++;}
    //     if (host_dst[idx]==1){count_1++;}
    //     if (host_dst[idx]==2){count_2++;}
    // }
    // printf("%d, %d, %d\n",count_0,count_1,count_2);
    // printf("%ld \n",sizeof(host_dst)/sizeof(unsigned char));

	cudaFree(device_src);
	cudaFree(device_dst);

    return 0;
}
// clear && nvcc resize_free.cu -o resize_free.o && ./resize_free.o