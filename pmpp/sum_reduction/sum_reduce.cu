#include <cuda.h>
#include <torch/types.h>

const size_t TILE_SIZE = 128;

template <typename T>
__global__
void sum_reduce_kernel(const T* matrix, T* results, int64_t size) {

    // first, copy into shared
    // each thread loads two into shared: the one in your tile, and the next tile without threads
    // theoretically, this should coalesce acceses as long as each thread accesses in the same position
    __shared__ T addends[TILE_SIZE];
    auto idx = blockIdx.x*(2*blockDim.x) + threadIdx.x;
    if (idx < size) {
        addends[threadIdx.x] = matrix[idx];
    } else {
        addends[threadIdx.x] = 0; // pad with zeros
    }

    if (idx + blockDim.x < size) {
        addends[threadIdx.x + blockDim.x] = matrix[idx + blockDim.x];
    } else {
        addends[threadIdx.x + blockDim.x] = 0;
    }

    // loop for how many reductions we need to do
    // now, instead of strides increasing, they decrease
    // start halfway across, then decrease
    // NOTE: half the threads in the kernel stop doing work here. You could instead
    // have each thread load in two values and you can now launch twice the # of threads 
    for (int stride=TILE_SIZE/2; stride>=1; stride/=2) {
        __syncthreads();
        if (threadIdx.x < stride)
            addends[threadIdx.x] += addends[threadIdx.x + stride];
    }
    __syncthreads();

    results[blockIdx.x] = addends[0];
}

torch::Tensor sum_reduce(torch::Tensor matrix) {
    const auto size = matrix.numel();

    const auto dtype = matrix.dtype();
    auto options = torch::TensorOptions().device(matrix.device()).dtype(dtype);
    auto n_tiles = size / TILE_SIZE + (size%TILE_SIZE);

    torch::Tensor sums = torch::empty(n_tiles, options);


    if (dtype==torch::kFloat32) {
        sum_reduce_kernel<float><<<n_tiles, TILE_SIZE/2>>>(
            (float*)matrix.const_data_ptr(), (float*)sums.mutable_data_ptr(), size);
    } else if (dtype==torch::kInt32) {
        sum_reduce_kernel<int><<<n_tiles, TILE_SIZE/2>>>(
            (int*)matrix.const_data_ptr(), (int*)sums.mutable_data_ptr(), size);
    }

    return sums.sum();
}

