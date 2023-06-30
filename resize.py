# pylint: disable=line-too-long, invalid-name, too-many-locals, raising-bad-type, c-extension-no-member, redefined-outer-name
import cv2
import cupy as cp
import numpy as np
from line_profiler import LineProfiler

with open('lib_cuResize.cu', 'r', encoding="utf-8") as reader:
    module = cp.RawModule(code=reader.read())

cuResizeKer = module.get_function("cuResize")
profile = LineProfiler()

@profile
def cuda_resize(inputs: cp.ndarray, # src: (N,H,W,C)
                shape: tuple, # (dst_h, dst_w)
                out: cp.ndarray=None, # dst: (N,H,W,C)
                pad: bool=True):

    out_dtype = cp.uint8

    N, src_h, src_w, C = inputs.shape
    assert C == 3 # resize kernel only accept 3 channel tensors.
    dst_h, dst_w = shape
    DST_SIZE = dst_h * dst_w * C

    # define kernel configs
    block = (1024, )
    grid = (int(DST_SIZE/3//1024)+1,N,3)

    if len(shape)!=2:
        raise "cuda resize target shape must be (h,w)"
    if out:
        assert out.dtype == out_dtype
        assert out.shape[1] == dst_h
        assert out.shape[2] == dst_w

    resize_scale = 1
    left_pad = 0
    top_pad = 0
    if pad:
        padded_batch = cp.zeros((N, dst_h, dst_w, C), dtype=out_dtype)
        if src_h / src_w > dst_h / dst_w:
            resize_scale = dst_h / src_h
            ker_h = dst_h
            ker_w = int(src_w * resize_scale)
            left_pad = int((dst_w - ker_w) / 2)
        else:
            resize_scale = dst_w / src_w
            ker_h = int(src_h * resize_scale)
            ker_w = dst_w
            top_pad = int((dst_h - ker_h) / 2)
    else:
        ker_h = dst_h
        ker_w = dst_w

    shape = (N, ker_h, ker_w, C)
    if not out:
        out = cp.empty(tuple(shape),dtype = out_dtype)

    with cp.cuda.stream.Stream() as stream:
        cuResizeKer(grid, block,
                (inputs, out,
                cp.int32(src_h), cp.int32(src_w),
                cp.int32(ker_h), cp.int32(ker_w),
                cp.float32(src_h/ker_h), cp.float32(src_w/ker_w)
                )
            )
        if pad:
            if src_h / src_w > dst_h / dst_w:
                padded_batch[:, :, left_pad:left_pad + out.shape[2], :] = out
            else:
                padded_batch[:, top_pad:top_pad + out.shape[1], :, :] = out
            padded_batch = cp.ascontiguousarray(padded_batch)
        stream.synchronize()

    if pad:
        return resize_scale, top_pad, left_pad, padded_batch
    return resize_scale, top_pad, left_pad, out



def main(input_array: cp.ndarray, resize_shape:tuple):
    input_array_gpu = cp.empty(shape=input_array.shape,dtype=input_array.dtype)

    if isinstance(input_array, cp.ndarray): # DtoD
        cp.cuda.runtime.memcpy(dst = int(input_array_gpu.data), # dst_ptr
                                src = int(input_array.data), # src_ptr
                                size=input_array.nbytes,
                                kind=3) # 0: HtoH, 1: HtoD, 2: DtoH, 3: DtoD, 4: unified virtual addressing
    elif isinstance(input_array, np.ndarray):
        cp.cuda.runtime.memcpy(dst = int(input_array_gpu.data), # dst_ptr
                                src = input_array.ctypes.data, # src_ptr
                                size=input_array.nbytes,
                                kind=1)

    resize_scale, top_pad, left_pad, output_array = cuda_resize(input_array_gpu,
                                                                    resize_shape,
                                                                    pad=False) # N,W,H,C

    return output_array, [resize_scale, top_pad, left_pad]

if __name__ == "__main__":
    # prepare data
    batch = 50
    img_batch = np.tile(cv2.resize(cv2.imread("trump.jpg"),
                                   (128,64)),
                        [batch,1,1,1])
    output_array, _ = main(img_batch, (256,128))

    for idx, img in enumerate(cp.asnumpy(output_array)):
        cv2.imwrite(f"output_{idx}.jpg", img)
