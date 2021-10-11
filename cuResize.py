import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np
import cv2
from line_profiler import LineProfiler

profile = LineProfiler()

bl_Normalize = 0
bl_Trans = 0
pagelock = 0

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

__global__ void Transpose_and_normalise(float *odata, const unsigned char *idata,
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

    odata[dst_idx] = idata[src_idx]/255.0;
}

__global__ void cuResize(unsigned char* src_img, unsigned char* dst_img, 
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

# block = (32, 32, 1)   blockDim | threadIdx 
# grid = (19,19,3))     gridDim  | blockIdx

cuResizeKer = module.get_function("cuResize")
TransposeKer = module.get_function("Transpose")
TransNorKer = module.get_function("Transpose_and_normalise")



class cuResize():
    """docstring for ClassName"""
    def __init__(self, shape=(1920,1080), batch=50, frame_w=1920, frame_h=1080):
        # ========= Init Params ========= 
        # size of frame
        self.batch = batch # limited by bytes, maximum around 200* 1080p ~= 50 * 4k
        self.channel = 3
        self.frame_w = frame_w # 1920 / 1920*n  , fixed input image size
        self.frame_h = frame_h # 1080 / 1080*n  , fixed input image size
        self.dst_w = shape[0] # 1920
        self.dst_h = shape[1] # 1080
        self.DST_SIZE = self.dst_h * self.dst_w * 3
        
        # memory 
        self.inp = None
        self.out = None
        # async stream
        self.stream = cuda.Stream()

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



    def warm_up(self):
        cuResizeKer(self.inp["device"], self.out["device"], 
                    np.int32(self.dst_h), np.int32(self.dst_w),
                    np.int32(self.dst_h), np.int32(self.dst_w),
                    np.float32(1), np.float32(1),
                    block=(1024, 1, 1),
                    grid=(int(self.DST_SIZE/3//1024)+1,self.batch,3),
                    stream=self.stream)                

    @profile
    def __call__(self, input_img: np.ndarray):
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

        cuResizeKer(self.inp["device"], self.out["device"], 
                    np.int32(src_h), np.int32(src_w),
                    np.int32(self.dst_h), np.int32(self.dst_w),
                    np.float32(src_h/self.dst_h), np.float32(src_w/self.dst_w),
                    block=(1024, 1, 1),
                    grid=(int(self.DST_SIZE/3//1024)+1,self.batch,3),
                    stream=self.stream)

        cuda.memcpy_dtoh_async(self.out["host"], self.out["device"],self.stream)

        self.stream.synchronize()
        # self.cleanup()
        return self.out["host"]

    def cleanup(self):
        self.inp["host"][:,:,:,:] = 0 

    def print_stats(self):
        profile.print_stats()

    # def deallocate(self):
    #     free(gpu_mem)


if __name__ == "__main__":
    from time import perf_counter
    batch = 200
    img_batch = np.tile(cv2.resize(cv2.imread("trump.jpg"),(1920,1080)),[batch,1,1,1])
    resizer = cuResize(shape=(1920,1080), batch=200, frame_h=1080, frame_w=1920)  # C backend hv to pre allocate input frame maximum dimension

    for _ in range(10):
        start = perf_counter()
        batch_result = resizer(img_batch)
        print("cuResize: ",perf_counter()- start,"s")
    print(batch_result.shape)
    resizer.print_stats()

    # batch_result = np.transpose(batch_result,[0,2,3,1])

    cv2.imwrite("output_1.jpg", batch_result[0])
    cv2.imwrite("output_50.jpg", batch_result[49])
    cv2.imwrite("output_102.jpg", batch_result[101])
    print(batch_result.shape)