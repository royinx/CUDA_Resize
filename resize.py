import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import cv2
from line_profiler import LineProfiler

profile = LineProfiler()

bl_Normalize = 0
bl_Trans = 1
pagelock = 1

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

__global__ void Transpose(unsigned char *odata, const unsigned char *idata)
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

__global__ void Transpose_and_normalise(float *odata, const unsigned char *idata)
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

    odata[dst_idx] = idata[src_idx]/255.0;
}

__global__ void YoloResize(unsigned char* src_img, unsigned char* dst_img, 
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
    
    int idx = n * (H * W * C) + 
              h * (W * C) +
              w * C +
              c;

    float centroid_h, centroid_w;  
    centroid_h = stride_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
    centroid_w = stride_w * (w + 0.5); // 

    int f00,f01,f10,f11;

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
          
    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
                            centroid_h, centroid_w));

    dst_img[idx] = (unsigned char)rs;
}
    """)

# block = (32, 32, 1)   blockDim | threadIdx 
# grid = (19,19,3))     gridDim  | blockIdx

YoloResizeKer = module.get_function("YoloResize")
TransposeKer = module.get_function("Transpose")
TransNorKer = module.get_function("Transpose_and_normalise")

@profile
def gpu_resize(input_img: np.ndarray):
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
    stream = cuda.Stream()

    # convert to array
    batch, src_h, src_w, channel = input_img.shape
    dst_h, dst_w = 608, 608

    # Mem Allocation
    # input data 
    input_img = input_img.astype(np.uint8)
    if pagelock: #  = = = = = = Pagelock emory = = = = = = 
        inp = {"host":cuda.pagelocked_empty_like(input_img,
                                                 mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        inp["host"] = input_img
    else: #  = = = = = = Global memory = = = = = = 
        inp = {"host":input_img}
        inp["device"] = cuda.mem_alloc(input_img.nbytes)

    
    inp["device"] = cuda.mem_alloc(inp["host"].nbytes)
    cuda.memcpy_htod_async(inp["device"], inp["host"],stream)


    # output data
    if pagelock: #  = = = = = = Pagelock emory = = = = = = 
        out = {"host":cuda.pagelocked_zeros(shape=(batch,dst_h,dst_w,channel), 
                                        dtype=np.uint8,
                                        mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
    else: #  = = = = = = Global memory = = = = = = 
        out = {"host":np.zeros(shape=(batch,dst_h,dst_w,channel), dtype=np.uint8)}  # N H W C
    
    out["device"] = cuda.mem_alloc(out["host"].nbytes)
    cuda.memcpy_htod_async(out["device"], out["host"],stream)


    #Transpose (and Normalize)
    if bl_Normalize or bl_Trans:
        if bl_Normalize:
            if pagelock:
                trans = {"host":cuda.pagelocked_zeros(shape=(batch,channel,dst_h,dst_w), 
                                                      dtype=np.float32,
                                                      mem_flags=cuda.host_alloc_flags.DEVICEMAP)}  # N C H W
            else:
                trans = {"host":np.zeros(shape=(batch,channel,dst_h,dst_w), dtype=np.float32)}  # N C H W
        else:
            if pagelock:
                trans = {"host":cuda.pagelocked_zeros(shape=(batch,channel,dst_h,dst_w), 
                                                      dtype=np.uint8,
                                                      mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
            else:
                trans = {"host":np.zeros(shape=(batch,channel,dst_h,dst_w), dtype=np.uint8)}  # N C H W

        trans["device"] = cuda.mem_alloc(trans["host"].nbytes)
        cuda.memcpy_htod_async(trans["device"], trans["host"],stream)

    # init resize , store kernel in cache
    YoloResizeKer(inp["device"], out["device"], 
               np.int32(src_h), np.int32(src_w),
               np.float32(src_h/dst_h), np.float32(src_w/dst_w),
               block=(32, 32, 1),
               grid=(19,19,3*batch))

    # ========= Testing =========

    for _ in range(10):
        YoloResizeKer(inp["device"], out["device"], 
                        np.int32(src_h), np.int32(src_w),
                        np.float32(src_h/dst_h), np.float32(src_w/dst_w),
                        block=(32, 32, 1),
                        grid=(19,19,3*batch))

    # ========= Copy out result =========

    if bl_Normalize:
        TransNorKer(trans["device"],out["device"],
                    block=(32, 32, 1),
                    grid=(19,19,3*batch))
        cuda.memcpy_dtoh_async(trans["host"], trans["device"],stream)
        stream.synchronize()
        return trans["host"]
    elif bl_Trans:
        TransposeKer(trans["device"],out["device"],
                    block=(32, 32, 1),
                    grid=(19,19,3*batch))
        cuda.memcpy_dtoh_async(trans["host"], trans["device"],stream)
        stream.synchronize()
        return trans["host"]
    else:
        cuda.memcpy_dtoh_async(out["host"], out["device"],stream)
        stream.synchronize()
        return out["host"]

if __name__ == "__main__":
    grid = 19
    block = 32
    batch = 2

    img = cv2.resize(cv2.imread("trump.jpg"),(1080,1920))
    # img = cv2.imread("trump.jpg")
    img = np.tile(img,[batch,1,1,1])


    pix = gpu_resize(img)
    print(pix.shape)
    if bl_Normalize or bl_Trans:
        pix = np.transpose(pix,[0,2,3,1])
        print(pix.shape)
    print(pix)
    cv2.imwrite("trans.jpg", pix[0])

    # profile.print_stats()
    # print(pix.shape)
    # cv2.imwrite("pycuda_outpuut.jpg", pix[0])