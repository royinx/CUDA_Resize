
# PyCUDA, CUDA Bilinear interpolation

Ultra fast Bilinear interpolation in image resize with CUDA.


Concept and base are in [`lerp.py`](lerp.py) (single thread, may take a while to run). <br/>
Boosting with C CUDA in [`resize_ker.cu`](resize_ker.cu). <br/>
PyCUDA example in [`resize.py`](resize.py)<br/>


Requirements:
>- GPU (compute capability: 3.0 or above, testing platform: 7.5)
>- CUDA driver
>- Docker and nvidia docker
---
Pros:
- support Batch image.
- no shared object .so and .dll binary file
- Install PyCUDA and use
- Compatible to `Numpy` library (only for data copy, not calculation)
- pass the GPU array to TensorRT directly. 

Cons:
- still need the concept of CUDA programming
- SourceModule have to write in C CUDA, including all CUDA kernel and device code

---
### Build 

```bash
git clone https://github.com/royinx/CUDA_Resize.git 
cd CUDA_Resize
docker build -t lerp_cuda .
docker run -it --runtime=nvidia -v ${PWD}:/py -w /py lerp_cuda bash 

# For PyCUDA implementation
python3 resize_free.py

# For concept
python3 lerp.py

# For CUDA kernel testing
nvcc resize_free.cu -o resize_free.o && ./resize_free.o

```
Remark: Development platform is in dockerfile.opencv with OpenCV in C for debugging

Function Working well in pycuda container, you dont need to build OpenCV.
