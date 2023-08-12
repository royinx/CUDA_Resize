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

__global__ void cuResize(unsigned char* src_img, unsigned char* dst_img,
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
    int src_h_idx = lroundf(centroid_h)-1;
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
            srcTile[row][w].x = src[row_start+w].x;
            srcTile[row][w].y = src[row_start+w].y;
            srcTile[row][w].z = src[row_start+w].z;
            }
    }
    __syncthreads();

    long long pixel_idx = n * DST_H * DST_W +  // offset batch
                          blockIdx.x * DST_W + // offset row(height)
                          threadIdx.x;         // offset col(width)

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

        if (src_w_idx+1>=SRC_W){f01 = f00; f11 = f10;}
        if (src_h_idx+1>=SRC_H){f10 = f00; f11 = f01;}

        dst[pixel_idx].x = (unsigned char) lroundf(lerp2d((*f00).x, (*f01).x, (*f10).x, (*f11).x, centroid_h, centroid_w));
        dst[pixel_idx].y = (unsigned char) lroundf(lerp2d((*f00).y, (*f01).y, (*f10).y, (*f11).y, centroid_h, centroid_w));
        dst[pixel_idx].z = (unsigned char) lroundf(lerp2d((*f00).z, (*f01).z, (*f10).z, (*f11).z, centroid_h, centroid_w));

        pixel_idx += blockDim.x;

    }
}
}