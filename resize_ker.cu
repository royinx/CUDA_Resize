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

__global__ void tester(unsigned char* src_img, unsigned char* dst_img, 
                       int src_h, int src_w, 
                       float stride_h, float stride_w)
{
    int H = blockDim.x * gridDim.x; // # dst_height
    int W = blockDim.y * gridDim.y; // # dst_width 
    int h = blockDim.x * blockIdx.x + threadIdx.x;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # x / h-th row
    int w = blockDim.y * blockIdx.y + threadIdx.y;  // 32 * bkIdx[0:18] + tdIdx; [0,607]   # y / w-th col
    int C = 3; // # ChannelDim
    int c = blockIdx.z % 3 ; // [0,2] # ChannelIdx
    int n = blockIdx.z / 3 ; // [0 , Batch size-1], # BatchIdx
    int N = gridDim.z / 3 ;
    
    // printf("%d(%d), %d(%d), %d(%d), %d(%d) \n",n,N,c,C,h,H,w,W);
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

    // int idx = x * blockDim.y * gridDim.y * gridDim.z + y * gridDim.z + z; // x * 608(width) * 3(channel) + y * 3(channel) + [0,2]
    
    float centroid_h, centroid_w;  
    centroid_h = stride_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = stride_w * (w + 0.5); // 

    int f00,f01,f10,f11;

    int src_h_idx = lroundf(centroid_h)-1;
    int src_w_idx = lroundf(centroid_w)-1;
    if (src_h_idx<0){src_h_idx=0;}
    if (src_w_idx<0){src_w_idx=0;}
    // printf("h:%d w:%d\n",src_h_idx,src_w_idx);

    // // idx = NHWC = n*(HWC) + h*(WC) + w*C + c;
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
    // dim3 dimBlock(32,32,1);  << Max total is 1024 , so , x=32 ,y=32 ,  some one use 1024 to handle flatten tensor is fine.
    // dim3 dimGrid(19,19,3); << x = 608 / 32 = 19  , same on y , z = channel * batch_size, assume channel = 3. 
    dim3 dimBlock(32,32,1);
    dim3 dimGrid(19,19,3);

    unsigned char host_src[1920*1080*3];
    // unsigned char host_dst[1108992];
    unsigned char host_dst[608*608*3];

    // init src image
    for(int i = 0; i < 1920*1080*3; i++){
        host_src[i] = i+1;
        // host_src[i] = (i%3);
    }

    float stride_h = 1080.0 / 608;
    float stride_w = 1920.0 / 608;

    unsigned char *device_src, *device_dst;
	cudaMalloc((unsigned char **)&device_src, 1920*1080*3* sizeof(unsigned char));
    cudaMalloc((unsigned char **)&device_dst, 608*608*3* sizeof(unsigned char));
    
	cudaMemcpy(device_src , host_src , 1920*1080*3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    tester<<<dimGrid, dimBlock>>>(device_src, device_dst, 
                                  1080, 1920,
                                  stride_h, stride_w);
    cudaDeviceSynchronize();
    
    cudaMemcpy(host_dst, device_dst, 608*608*3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // DEBUG : print first image in batch , first 30 pixel in 3 channels.

    for(int i = 0; i < 30*3; i+=3){ // NHWC
        printf("%d\n",host_src[i]);
    }
    printf("============================\n");
 
    // for(int c = 0; c<3*608*608 ; c+=608*608){ // if NCHW
    //     for(int i = 0 ; i < 30; i++){
    //         printf("%d %d %d\n", c+i, i, host_dst[c+i]);
    //     }
    //     printf("------------------------------\n");
    // }
    for(int c = 0; c<3; c++){ // NHWC
        for(int i = 0 ; i < 30; i++){
            int idx = i*3 +c;
            printf("%d %d %d\n", c+i, i, host_dst[idx]);
        }
        printf("------------------------------\n");
    }


    
	cudaFree(device_src);
	cudaFree(device_dst);

    return 0;
}