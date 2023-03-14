extern "C"{
#define TILE_DIM 32
#define BLOCK_ROWS 8

__device__ double lerp1d(int a, int b, float w)
{
    return fma(w, (float)b, fma(-w,(float)a,(float)a));
}

__device__ float lerp2d(int f00, int f01, int f10, int f11,
                        float centroid_h, float centroid_w )
{
    centroid_w = (1 + lroundf(centroid_w) - centroid_w)/2;
    centroid_h = (1 + lroundf(centroid_h) - centroid_h)/2;
    
    double r0, r1, r;
    r0 = lerp1d(f00,f01,centroid_w);
    r1 = lerp1d(f10,f11,centroid_w);

    r = lerp1d(r0, r1, centroid_h); //+ 0.00001
    return r;
}

__global__ void cuResize(unsigned char* src_img, unsigned char* dst_img, 
    const int src_h, const int src_w, 
    const int dst_h, const int dst_w,
    const float scale_h, const float scale_w)
{
    /* 
    Input: 
        src_img - NHWC
        channel C, default = 3 
    
    Output:
        dst_img - NHWC
    */

    // __shared__ float srcTile[TILE_DIM][TILE_DIM];

    int n = blockIdx.y; // batch number
    int C = gridDim.z; // channel 
    int c = blockIdx.z; // channel number
    long long idx = n * blockDim.x * gridDim.x * C + 
                    threadIdx.x * gridDim.x * C +
                    blockIdx.x * C+
                    c;
    
    // some overhead threads in each image process
    // when thread idx in one image exceed one image size return;
    if (idx%(blockDim.x * gridDim.x * C) >= dst_h* dst_w * C){return;} 

    int H = dst_h;
    int W = dst_w;
    int img_coor = idx % (dst_h*dst_w*C); //coordinate of one image, not idx of batch image
    int h = img_coor / (W*C);
    int w = img_coor % (W*C)/C; 

    float centroid_h, centroid_w;  
    centroid_h = scale_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = scale_w * (w + 0.5); // 

    long long f00,f01,f10,f11;

    int src_h_idx = lroundf(centroid_h)-1;
    int src_w_idx = lroundf(centroid_w)-1;
    if (src_h_idx<0){src_h_idx=0;}
    if (src_w_idx<0){src_w_idx=0;}

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
    if (src_w_idx+1>=src_w){f01 = f00; f11 = f10;}
    if (src_h_idx+1>=src_h){f10 = f00; f11 = f01;}
    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
        centroid_h, centroid_w));

    long long dst_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    dst_img[dst_idx] = (unsigned char)rs;
}
__global__ void cuResize_fp(unsigned char* src_img, float* dst_img, 
    const int src_h, const int src_w, 
    const int dst_h, const int dst_w,
    const float scale_h, const float scale_w,
    const bool normalise)
{
    /* 
    Input: 
        src_img - NHWC
        channel C, default = 3 
    
    Output:
        dst_img - NHWC
    */

    // __shared__ float srcTile[TILE_DIM][TILE_DIM];

    int n = blockIdx.y; // batch number
    int C = gridDim.z; // channel 
    int c = blockIdx.z; // channel number
    long long idx = n * blockDim.x * gridDim.x * C + 
              threadIdx.x * gridDim.x * C +
              blockIdx.x * C+
              c;
    
    // some overhead threads in each image process
    // when thread idx in one image exceed one image size return;
    if (idx%(blockDim.x * gridDim.x * C) >= dst_h* dst_w * C){return;} 

    int H = dst_h;
    int W = dst_w;
    int img_coor = idx % (dst_h*dst_w*C); //coordinate of one image, not idx of batch image
    int h = img_coor / (W*C);
    int w = img_coor % (W*C)/C; 

    float centroid_h, centroid_w;  
    centroid_h = scale_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = scale_w * (w + 0.5); // 

    long long f00,f01,f10,f11;

    int src_h_idx = lroundf(centroid_h)-1;
    int src_w_idx = lroundf(centroid_w)-1;
    if (src_h_idx<0){src_h_idx=0;}
    if (src_w_idx<0){src_w_idx=0;}

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
    
    // printf("%ld, %ld, %ld, %ld \n", f00, f01, f10, f11);
    if (src_w_idx+1>=src_w){f01 = f00; f11 = f10;}
    if (src_h_idx+1>=src_h){f10 = f00; f11 = f01;}

    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
        centroid_h, centroid_w));

    long long dst_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    float rs_f;

    if (normalise){
        float pixel_mean[3] = {0.485, 0.456, 0.406};
        float pixel_std[3]  = {0.229, 0.224, 0.225};
        rs_f = (unsigned char)rs/ 255.0;
        rs_f = (rs_f - pixel_mean[c])/pixel_std[c];
    }else{
        rs_f = (unsigned char)rs/ 255.0;
    }
    dst_img[dst_idx] = rs_f;
}
}