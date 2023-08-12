# pylint: disable=line-too-long, invalid-name, too-many-locals, c-extension-no-member, redefined-outer-name

# built-in library
import sys
import os
import time

# third party library
import cv2
import cupy as cp
import numpy as np
import pandas as pd
from resize import cuda_resize

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

def warm_up():
    input_array_gpu = cp.ones(shape=(200,1080,1920,3),dtype=np.uint8)
    _, _, _, output_array = cuda_resize(input_array_gpu,
                                                                    (128,256),
                                                                    pad=False) # N,W,H,C
    print("Warm up:", output_array.shape)


if __name__ == "__main__":
    # prepare data
    batch = 100
    warm_up()
    size = [(3840,2160),(1920,1080), (960,540), (480,270), (240,135), (120,67), (60,33), (30,16)]
    benchmark = pd.DataFrame(columns=[str(size_) for size_ in size],
                             index=[str(size_) for size_ in size])

    # benchmark = defaultdict(dict)
    for src_shape in size:
        if os.path.exists(f"{src_shape}.npy"):
            imgs = np.load(f"{src_shape}.npy")
        else:
            imgs = [cv2.resize(cv2.imread(f"val2017/{img_name}"),src_shape) for img_name in os.listdir("val2017")[:1000]]
            imgs = np.asarray(imgs)
            np.save(f"{src_shape}.npy",imgs)

        for dst_shape in size:
            # CPU benchmark
            cpu_metrics = []
            start = time.perf_counter()
            for index in range(0, len(imgs), batch):
                start = time.perf_counter()
                cpu_output = [cv2.resize(img,(dst_shape))for img in imgs[index:index+batch]]
                cpu_metrics.append(time.perf_counter() - start)
                # cv2.imwrite(f"{index}_output_cpu.jpg", cpu_output[0])

            # CUDA benchmark
            cuda_metrics = []
            for index in range(0, len(imgs), batch):
                input_array = imgs[index:index+batch]
                input_array_gpu = cp.empty(shape=input_array.shape,dtype=input_array.dtype)
                cp.cuda.runtime.memcpy(dst = int(input_array_gpu.data), # dst_ptr
                                        src = input_array.ctypes.data, # src_ptr
                                        size=input_array.nbytes,
                                        kind=1)
                # input_array_gpu = cp.load(f"{src_shape}.npy")


                # execution
                start = time.perf_counter()
                _, _, _, output_array = cuda_resize(input_array_gpu,
                                                    dst_shape[::-1],
                                                    pad=False) # N,W,H,C

                cuda_metrics.append(time.perf_counter() - start)
                # cv2.imwrite(f"{index}_output_cuda.jpg", cp.asnumpy(output_array[0]))
                del input_array_gpu
                cp.get_default_memory_pool().free_all_blocks()
            cpu_ = sum(cpu_metrics)
            gpu_ = sum(cuda_metrics)
            speedup = cpu_/gpu_
            benchmark[f"{src_shape}"][f"{dst_shape}"] = speedup
            # print(f"{src_shape} -> {dst_shape}: \t CPU: {cpu_} \t | CUDA: {gpu_} \t | Speedup: {speedup}")
            # print(benchmark)
        del imgs
    print(benchmark)