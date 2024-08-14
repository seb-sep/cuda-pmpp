#include <cuda.h>
#include <torch/types.h>

const size_t TILE_SIZE = 32;
__global__
void kogge_stone(const float* input, float* output, int size) {
    __shared__ float input_s[TILE_SIZE];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
        input_s[threadIdx.x] = input[idx];


    for (int stride=1; stride<blockDim.x; stride*=2) {
        __syncthreads();
        // why is there no overlap? 
        if (threadIdx.x >= stride)
            input_s[threadIdx.x] += input_s[threadIdx.x - stride];

    }

    if (idx < size)
        output[idx] = input_s[threadIdx.x];

}

__global__
void brent_kung(const float* input, float* output, int size) {
    __shared__ float input_s[TILE_SIZE];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
        input_s[threadIdx.x] = input[idx];

    for (uint stride=1; stride<=TILE_SIZE/2; stride*=2) {
        __syncthreads();

        // you're doubling how far a thread reaches across each time,
        // and in doing so you push half your threads over the threshold 
        // of usability in each iteration
        // threads 0-31, 0-15, 0-7, and so on
        // Keeps threads contiguous
        int i = (threadIdx.x+1) * 2 * stride - 1;
        if (i < TILE_SIZE)
            input_s[i] += input_s[i - stride];
    }

    // how about the reverse tree? 
    // start with largest necessary stride
    // only one value should be a multiple of half the stride, the halfway point
    // Push that value to the right by a stride
    // Then divide the stride by 2 and repeat (2 more values push right by
    // the shorter stride)
    for (uint stride=TILE_SIZE/4; stride>=1; stride/=2) {
        __syncthreads();

        int i = (threadIdx.x+1) * stride * 2 - 1;
        if (i + stride < TILE_SIZE)
            input_s[i + stride] += input_s[i];
    }


    if (idx < size)
        output[idx] = input_s[threadIdx.x];


}

torch::Tensor add_scan(torch::Tensor input) {
    torch::Tensor output = torch::empty_like(input);
    auto size = input.numel();

    auto n_tiles = size / TILE_SIZE + (size%TILE_SIZE);
    brent_kung<<<n_tiles, TILE_SIZE>>>(
        (float*)input.const_data_ptr(), (float*)output.mutable_data_ptr(), size);

    return output;
}