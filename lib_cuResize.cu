extern "C"{
#define MAX_WIDTH 3840 // 7680 3840 1920


__device__ float lerp1d(int a, int b, float w)
{
    return fma(w, (float)b, fma(-w,(float)a,(float)a));
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
    return r;
}

__global__ void cuResize_xyz(unsigned char* src_img, unsigned char* dst_img,
    const int SRC_H, const int SRC_W,
    const int DST_H, const int DST_W,
    const float scale_h, const float scale_w)
{
    /*
    Input:
        src_img - NHWC
        channel C, default = 3

    Output:
        dst_img - NHWC
    */
    if (DST_W < SRC_W & threadIdx.x>=SRC_W){return;}
    const uchar3* src = (uchar3*)(src_img);
    uchar3* dst = (uchar3*)(dst_img);

    // coordinate dst pixel in src image
    int dst_row_idx = blockIdx.x;
    float centroid_h;
    centroid_h = scale_h * (dst_row_idx + 0.5);
    // printf("dst_row_idx %f %f, %d\n", centroid_h, scale_h , dst_row_idx );
    // return;
    // if (threadIdx.x>=DST_W){return;}
    // printf("centroid_h %f\n", centroid_h);
    int src_h_idx = lroundf(centroid_h)-1;
    // printf("src_h_idx %d\n", src_h_idx);
    // return;
    if (src_h_idx<0){src_h_idx=0;}

    int n = blockIdx.y; // batch number
    __shared__ uchar3 srcTile[2][MAX_WIDTH];  // cache `2 src rows` for `1 dst row` pixel
    int row_start;
    int pix_idx;

    for( int w = threadIdx.x ; w < SRC_W ; w+=blockDim.x){
        pix_idx = n * SRC_H * SRC_W +   // move to the start of image in batch
                  src_h_idx * SRC_W ;   // move to the start of row index of src image
        // loop over 2 row image
        for (int row = 0; row < 2; row++){
            row_start = pix_idx + SRC_W * row;            // jump to next row
            // printf("row %d , w %d , row_start %d,\t %d\n", row, w, src_h_idx, pix_idx);
            // printf("(%d,) (%d,%d) | row %d , w %d | %d  %d | %d %d %d %d\n",
            //         threadIdx.x, blockIdx.x , blockIdx.y ,
            //         row, w,
            //         pix_idx , src_h_idx * SRC_W,
            //         row_start+w, src[row_start+w].x, src[row_start+w].y, src[row_start+w].z);
            srcTile[row][w].x = src[row_start+w].x;
            srcTile[row][w].y = src[row_start+w].y;
            srcTile[row][w].z = src[row_start+w].z;
            }
    }
    // for( int row = 0; row < 2; row++){
    //     for( int w = threadIdx.x ; w < SRC_W ; w+=blockDim.x){
    //         printf("x, %d %d %d\n", row, w , srcTile[row][w].x);
    //         printf("y, %d %d %d\n", row, w , srcTile[row][w].y);
    //         printf("z, %d %d %d\n", row, w , srcTile[row][w].z);
    //     }
    // }
    __syncthreads();

    // return;

    // some overhead threads in each image process
    // when thread idx in one image exceed one image size return;

    long long pixel_idx = n * DST_H * DST_W +  // offset batch
                          blockIdx.x * DST_W + // offset row(height)
                          threadIdx.x;         // offset col(width)

    // if(pixel_idx > DST_H * DST_W){
    //     printf("%d, %ld %d %d %d %d \n",
    //     DST_H * DST_W, pixel_idx, blockIdx.x * DST_W , blockIdx.x , DST_W , threadIdx.x);
    // }
    // return;
    uchar3 *f00, *f01, *f10, *f11;
    float centroid_w;
    for( int w = threadIdx.x ; w < DST_W ; w+=blockDim.x){

        centroid_w = scale_w * (w + 0.5);
        int src_w_idx = lroundf(centroid_w)-1;
        if (src_w_idx<0){src_w_idx=0;}

        // loop over 2 row image

        f00 = &srcTile[0][src_w_idx];
        f01 = &srcTile[0][src_w_idx+1];
        f10 = &srcTile[1][src_w_idx];
        f11 = &srcTile[1][src_w_idx+1];

        // printf("%d %d %d %d \n", srcTile[row][src_w_idx].x, srcTile[row][src_w_idx+1].x, srcTile[row+1][w].x, srcTile[row+1][src_w_idx+1].x);

        if (src_w_idx+1>=SRC_W){f01 = f00; f11 = f10;}
        if (src_h_idx+1>=SRC_H){f10 = f00; f11 = f01;}

        // printf("x, %ld tidx.x %d\tw %d\tf: %d\t%d\t%d\t%d\n", pixel_idx, threadIdx.x, w, (*f00).x, (*f01).x, (*f10).x, (*f11).x);
        // printf("y, %ld tidx.x %d\tw %d\tf: %d\t%d\t%d\t%d\n", pixel_idx, threadIdx.x, w, (*f00).y, (*f01).y, (*f10).y, (*f11).y);
        // printf("z, %ld tidx.x %d\tw %d\tf: %d\t%d\t%d\t%d\n", pixel_idx, threadIdx.x, w, (*f00).z, (*f01).z, (*f10).z, (*f11).z);

        dst[pixel_idx].x = (unsigned char) lroundf(lerp2d((*f00).x, (*f01).x, (*f10).x, (*f11).x, centroid_h, centroid_w));
        dst[pixel_idx].y = (unsigned char) lroundf(lerp2d((*f00).y, (*f01).y, (*f10).y, (*f11).y, centroid_h, centroid_w));
        dst[pixel_idx].z = (unsigned char) lroundf(lerp2d((*f00).z, (*f01).z, (*f10).z, (*f11).z, centroid_h, centroid_w));

        // idx += w;

        // if(pixel_idx > DST_H * DST_W){
        // printf("%ld , %d , %d , %d\n", pixel_idx, threadIdx.x, w,blockDim.x );
        // }
        // gridDim.x * blockIdx.y + // DST_H * n
        // blockIdx.x + // row number
        // w;


        // printf("%d \n",pixel_idx);
        // centroid_w += scale_w*blockDim.x;
        pixel_idx += blockDim.x;

    }
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
    // printf("c %d, %d %d %d %d\n", c, src_img[f00], src_img[f01], src_img[f10], src_img[f11]);
    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11],
        centroid_h, centroid_w));
    // printf("%d\n", rs);
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