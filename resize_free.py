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

__global__ void Transpose(unsigned char *odata, const unsigned char *idata,
                            int H, int W)
{
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

__global__ void Transpose_and_normalise(float *odata, const unsigned char *idata,
                            int H, int W)
{
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

    odata[dst_idx] = idata[src_idx]/255.0;
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

# block = (32, 32, 1)   blockDim | threadIdx 
# grid = (19,19,3))     gridDim  | blockIdx

cuResizeKer = module.get_function("cuResize")
TransposeKer = module.get_function("Transpose")
TransNorKer = module.get_function("Transpose_and_normalise")

@profile
def gpu_resize(input_img: np.ndarray, shape=(608,608)):
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
    dst_h, dst_w = shape[0], shape[1]
    DST_SIZE = dst_h* dst_w* 3
    # Mem Allocation
    # input memory
    
    if pagelock: #  = = = = = = Pagelock emory = = = = = = 
        inp = {"host":cuda.pagelocked_zeros(shape=(batch,src_h,src_w,channel),
                                            dtype=np.uint8,
                                            mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        # inp = {"host":cuda.pagelocked_empty_like(input_img,
                                                #  mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        # print(inp["host"].shape,input_img.shape)
        inp["host"][:,:src_h,:src_w,:] = input_img
    else: #  = = = = = = Global memory = = = = = = 
        inp = {"host":input_img}

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

    import time
    time.sleep(5)
    
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
    cuResizeKer(inp["device"], out["device"], 
               np.int32(src_h), np.int32(src_w),
               np.int32(dst_h), np.int32(dst_w),
               np.float32(src_h/dst_h), np.float32(src_w/dst_w),
               block=(1024, 1, 1),
               grid=(int(DST_SIZE/3//1024)+1,batch,3),
               stream=stream)
    
    # ========= Testing =========

    for _ in range(1):
        cuResizeKer(inp["device"], out["device"], 
            np.int32(src_h), np.int32(src_w),
            np.int32(dst_h), np.int32(dst_w),
            np.float32(src_h/dst_h), np.float32(src_w/dst_w),
            block=(1024, 1, 1),
            grid=(int(DST_SIZE/3//1024)+1,batch,3))

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
                    np.int32(dst_h), np.int32(dst_w),
                    block=(1024, 1, 1),
                    grid=(int(DST_SIZE/3//1024)+1,batch,3))
        cuda.memcpy_dtoh_async(trans["host"], trans["device"],stream)
        stream.synchronize()
        return trans["host"]
    else:
        cuda.memcpy_dtoh_async(out["host"], out["device"],stream)
        stream.synchronize()
        return out["host"]

if __name__ == "__main__":
    # img = cv2.resize(cv2.imread("trump.jpg"),(1920,1080))
    # img = cv2.imread("trump.jpg")
    # img = np.tile(img,[batch,1,1,1])

    # img = np.zeros(shape=(3,1080,1920,3),dtype = np.uint8)
    # img[0,:48,:64,:] = cv2.resize(cv2.imread("trump.jpg"),(64,48))
    # img[1,:480,:640,:] = cv2.resize(cv2.imread("trump.jpg"),(640,480))
    # img[2,:1080,:1920,:] = cv2.resize(cv2.imread("trump.jpg"),(1920,1080))

    batch = 50
    # img_batch_0 = np.tile(cv2.resize(cv2.imread("trump.jpg"),(20,20)),[batch,1,1,1])
    # img_batch_1 = np.tile(cv2.resize(cv2.imread("trump.jpg"),(320,240)),[batch,1,1,1])
    img_batch_2 = np.tile(cv2.resize(cv2.imread("trump.jpg"),(1920,1080)),[batch,1,1,1])
    
    # rgba_img = cv2.resize(cv2.imread("rgba.png"),(20,20))
    # img_batch_0[10] = rgba_img
    # img_batch_0[20] = rgba_img
    # img_batch_0[53] = rgba_img

    # pix_0 = gpu_resize(img_batch_0)
    # pix_1 = gpu_resize(img_batch_1)
    pix_2 = gpu_resize(img_batch_2,shape = (480,640))
    if bl_Normalize or bl_Trans:
        # print(1)
        # pix_0 = np.transpose(pix_0,[0,2,3,1])
        # pix_1 = np.transpose(pix_1,[0,2,3,1])
        pix_2 = np.transpose(pix_2,[0,2,3,1])
    # cv2.imwrite("trans0.jpg", pix_0[0])
    # cv2.imwrite("trans1.jpg", pix_1[0])
    cv2.imwrite("trans2.jpg", pix_2[0])
    print("Done")

    # print(pix_0[0])
    # print(pix_0[-1])
    # print(pix_0.shape)

    # imgs = pix_1
    # for idx,img in enumerate(list(imgs)):
    #     print(idx)
    #     assert np.array_equal(imgs[0],img)

    # profile.print_stats()
    # print(pix.shape)
    # cv2.imwrite("pycuda_outpuut.jpg", pix[0])