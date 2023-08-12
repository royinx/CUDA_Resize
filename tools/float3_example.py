import cupy as cp

float3_code = r"""
extern "C"{

__global__ void test_sum(const float* x, const float* y, float* out, \
                         unsigned int N)
{
    unsigned int h = threadIdx.x;
    unsigned int w = threadIdx.y;
    unsigned int tid = blockDim.x * threadIdx.x + threadIdx.y ;
    // printf("idx: %d, idy: %d , dimx: %d, dimy: %d \n", blockIdx.x, blockIdx.y, blockDim.x, blockDim.y);
    // printf("tid: %d, N: %d \n", tid, N);
    float3* tensor_x   = (float3* )(x);
    float3* tensor_y   = (float3* )(y);
    float3* tensor_out = (float3* )(out);
    // printf("x: %f, y: %f, out: %f \n", tensor_x[tid].x, tensor_y[tid].x, tensor_out[tid].x);
    printf("x: %f, y: %f, out: %f \n", tensor_x[tid].x, tensor_y[tid].x, tensor_out[tid].x);

    if (tid < N)
    {
        printf("x: %f, y: %f, out: %f \n", tensor_x[tid].x, tensor_y[tid].x, tensor_out[tid].x);
        tensor_out[tid].x = tensor_x[tid].x + tensor_y[tid].x;
        tensor_out[tid].y = tensor_x[tid].y + tensor_y[tid].y;
        tensor_out[tid].z = tensor_x[tid].z + tensor_y[tid].z;
    }

    // printf("x: %f, y: %f, out: %f \n", tensor_x[tid].x, tensor_y[tid].x, tensor_out[tid].x);
}
}
"""
mod = cp.RawModule(code=float3_code)
# mod = cp.RawModule(code=float3_code)
ker = mod.get_function('test_sum')
# ker = mod.get_function('test_sum<float>')
a_array = cp.arange(12, dtype=cp.float32).reshape((2,2,3))
b_array = cp.arange(12, 24, dtype=cp.float32).reshape((2,2,3))
result = cp.zeros((2,2,3), dtype=cp.float32)

ker((1,), (2,2,), (a_array, b_array, result, result.size//3))
print("A\n", a_array)
print("\nB\n", b_array)
print("\nresult\n",result)

print("cpu", a_array + b_array)
# assert cp.allclose(result, 5*(2*x)+3*n)  # note that we've multiplied by 2 earlier