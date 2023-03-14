import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import cv2
from line_profiler import LineProfiler

profile = LineProfiler()

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
                       int frame_h, int frame_w, 
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

    f00 = n * frame_h * frame_w * C + 
          src_h_idx * frame_w * C + 
          src_w_idx * C +
          c;
    f01 = n * frame_h * frame_w * C +
          src_h_idx * frame_w * C +
          (src_w_idx+1) * C +
          c;
    f10 = n * frame_h * frame_w * C +
          (src_h_idx+1) * frame_w * C +
          src_w_idx * C +
          c;
    f11 = n * frame_h * frame_w * C + 
          (src_h_idx+1) * frame_w * C +
          (src_w_idx+1) * C +
          c;
          
    int rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
                            centroid_h, centroid_w));

    dst_img[idx] = (unsigned char)rs;
}
    """)

# block = (32, 32, 1)   blockDim | threadIdx 
# grid = (19,19,3))     gridDim  | blockIdx


class GPU_RESIZE_PROCESSOR():
    """docstring for ClassName"""
    def __init__(self, frame_h,frame_w, batch):
        # ========= Init Params ========= 
        # size of frame
        self.batch = batch
        self.channel = 3
        self.frame_h = frame_h # 1080 / 1080*n
        self.frame_w = frame_w #1920 / 1920*n
        self.dst_h = 608
        self.dst_w = 608
        
        # memory 
        self.inp = None
        self.out = None
        self.trans = None
        # async stream
        self.stream = cuda.Stream()

        # CUDA kernel
        self.YoloResizeKer = module.get_function("YoloResize")
        self.TransposeKer = module.get_function("Transpose")
        self.TransNorKer = module.get_function("Transpose_and_normalise")

        self.allocate_memory()
        self.warm_up() # warm up

    def allocate_memory(self):
        self.inp = {"host":cuda.pagelocked_zeros(shape=(self.batch,self.frame_h,self.frame_w,self.channel),
                                                 dtype=np.uint8,
                                                 mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        self.inp["device"] = cuda.mem_alloc(self.inp["host"].nbytes)


        self.out = {"host":cuda.pagelocked_zeros(shape=(self.batch,self.dst_h,self.dst_w,self.channel), 
                                                 dtype=np.uint8,
                                                 mem_flags=cuda.host_alloc_flags.DEVICEMAP)}
        self.out["device"] = cuda.mem_alloc(self.out["host"].nbytes)


        self.trans = {"host":cuda.pagelocked_zeros(shape=(self.batch,self.channel,self.dst_h,self.dst_w), 
                                                #    dtype=np.float32,
                                                   dtype=np.uint8,
                                                   mem_flags=cuda.host_alloc_flags.DEVICEMAP)}  # N C H W
        self.trans["device"] = cuda.mem_alloc(self.trans["host"].nbytes)

    def warm_up(self):
        self.YoloResizeKer(self.inp["device"], self.out["device"],
                            np.int32(self.frame_h), np.int32(self.frame_w),
                            np.int32(self.frame_h), np.int32(self.frame_w),
                            np.float32(1), np.float32(1),
                            block=(32, 32, 1),
                            grid=(19,19,3*self.batch))
        # self.TransNorKer(self.trans["device"],self.out["device"],
        #                  block=(32, 32, 1),
        #                  grid=(19,19,3*self.batch))
        self.TransposeKer(self.trans["device"],self.out["device"],
                          block=(32, 32, 1),
                          grid=(19,19,3*self.batch)) 

    @profile
    def resize(self, input_img: np.ndarray):
        """
        Resize the batch image to (608,608) 
        and Convert NHWC to NCHW
        pass the gpu array to normalize the pixel ( divide by 255)

        Application oriented

        input_img : batch input, format: NHWC , recommend RGB. *same as the NN input format 
                    input must be 3 channel, kernel set ChannelDim as 3.
        out : batch resized array, format: NCHW , same as intput channel
        """
        batch, src_h, src_w, channel = input_img.shape
        assert (src_h <= self.frame_h) & (src_w <= self.frame_w)
        self.inp["host"][:,:src_h,:src_w,:] = input_img
        cuda.memcpy_htod_async(self.inp["device"], self.inp["host"],self.stream)

        self.YoloResizeKer(self.inp["device"], self.out["device"], 
                           np.int32(src_h), np.int32(src_w),
                           np.int32(self.frame_h), np.int32(self.frame_w),
                           np.float32(src_h/self.dst_h), np.float32(src_w/self.dst_w),
                           block=(32, 32, 1),
                           grid=(19,19,3*self.batch))
        # self.TransNorKer(self.trans["device"],self.out["device"],
        #                  block=(32, 32, 1),
        #                  grid=(19,19,3*self.batch))  

        self.TransposeKer(self.trans["device"],self.out["device"],
                          block=(32, 32, 1),
                          grid=(19,19,3*self.batch))   
        cuda.memcpy_dtoh_async(self.trans["host"], self.trans["device"],self.stream)

        self.stream.synchronize()
        # self.cleanup()
        return self.trans["host"]

    def cleanup(self):
        self.inp["host"][:,:,:,:] = 0 

    # def deallocate(self):
    #     free(gpu_mem)
@profile
def main():

    batch_size = 58
    CUDA_processor = GPU_RESIZE_PROCESSOR(frame_h=1080,frame_w=1920,batch=batch_size)
    
    shape = [(64,48),(320,240),(1920,1080)]
    for idx,batch in enumerate(shape):
        img_batch = np.tile(cv2.resize(cv2.imread("trump.jpg"),batch),[batch_size,1,1,1])
        pix = CUDA_processor.resize(img_batch)
        pix = np.transpose(pix,[0,2,3,1])
        cv2.imwrite(f"trans{idx}.jpg", pix[0])

    profile.print_stats()
    # print(pix.shape)
    # cv2.imwrite("pycuda_outpuut.jpg", pix[0])

if __name__ == "__main__":
    main()
    