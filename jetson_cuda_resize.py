import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.autoinit import context
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import cv2

module = SourceModule("""

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
    return r;
}

__global__ void Transpose(unsigned char *odata, const unsigned char *idata,
                            int H, int W)
{
    int N = gridDim.y; // batch size
    int n = blockIdx.y; // batch number
    int C = gridDim.z; // channel 
    int c = blockIdx.z; // channel number
    long long idx = n * blockDim.x * gridDim.x * C + 
               threadIdx.x * gridDim.x * C +
               blockIdx.x * C+
               c;
    int img_coor = idx % (H*W*C); //coordinate of one image, not idx of batch image
    int h = img_coor / (W*C); // dst idx 
    int w = img_coor % (W*C)/C; // dst idx

    long long src_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    long long dst_idx = n * (C * H * W) +
                    c * (H * W)+
                    h * W+
                    w;

    odata[dst_idx] = idata[src_idx];
}

__global__ void cuResize(unsigned char* dst_img, unsigned char* src_img,
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
    centroid_h = stride_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = stride_w * (w + 0.5); // 

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

          
    // int rs;   
    // if (int(f10/ (src_h * src_w * C)) > n ){
    //     centroid_w = (1 + lroundf(centroid_w) - centroid_w)/2;
    //     rs = lroundf(lerp1d(f00,f01,centroid_w));
    // }else{
    //     rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
    //         centroid_h, centroid_w));
    // }
    
    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
        centroid_h, centroid_w));

    long long dst_idx = n * (H * W * C) + 
                    h * (W * C) +
                    w * C +
                    c;

    dst_img[dst_idx] = (unsigned char)rs;
}
    """)

cuResizeKer = module.get_function("cuResize")
TransposeKer = module.get_function("Transpose")

def gpu_resize(input_img: np.ndarray, stream):
    """
    Resize the batch image to (608,608) 
    and Convert NHWC to NCHW
    pass the gpu array to normalize the pixel ( divide by 255)

    Application oriented

    input_img : batch input, format: NHWC , recommend RGB. *same as the NN input format 
                input must be 3 channel, kernel set ChannelDim as 3.
    out : batch resized array, format: NCHW , same as intput channel
    """
    # ========= Init Params =========


    # convert to array
    batch, src_h, src_w, channel = input_img.shape
    dst_h, dst_w = 480, 640
    DST_SIZE = dst_h* dst_w* 3
    # Mem Allocation
    # input memory
    inp = cuda.managed_zeros(shape=(batch,src_h,src_w,channel),
                             dtype=np.uint8,
                             mem_flags=cuda.mem_attach_flags.GLOBAL)

    inp[:,:src_h,:src_w,:] = input_img

    # output data
    out = cuda.managed_zeros(shape=(batch,dst_h,dst_w,channel),
                             dtype=np.uint8,
                             mem_flags=cuda.mem_attach_flags.GLOBAL)

    #Transpose
    trans = cuda.managed_zeros(shape=(batch,channel,dst_h,dst_w),
                             dtype=np.uint8,
                             mem_flags=cuda.mem_attach_flags.GLOBAL)

    cuResizeKer(out, inp, 
                np.int32(src_h), np.int32(src_w),
                np.int32(dst_h), np.int32(dst_w),
                np.float32(src_h/dst_h), np.float32(src_w/dst_w),
                block=(1024, 1, 1),
                grid=(int(DST_SIZE/3//1024)+1,batch,3),
                stream=stream)

    TransposeKer(trans,out,
                np.int32(dst_h), np.int32(dst_w),
                block=(1024, 1, 1),
                grid=(int(DST_SIZE/3//1024)+1,batch,3),
                stream=stream)

    # Wait for kernel completion before host access
#     stream.synchronize()
    context.synchronize()

    return trans


if __name__ == "__main__":
    import cv2

    batch = 32
    img_batch = np.tile(cv2.resize(cv2.imread("debug_image/helmet.jpg"),(1920,1080)),[batch,1,1,1])
        
    pix = gpu_resize(img_batch,stream)
    pix = np.transpose(pix,[0,2,3,1]) 