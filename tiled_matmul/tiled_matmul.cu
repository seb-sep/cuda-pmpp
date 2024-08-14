#include <cuda.h>
#include <torch/types.h>


const size_t TILE_WIDTH = 2;

// Remember that PyTorch tensors are row-major
template <typename T>
__global__ 
void tiled_matmul_kernel(const T* A, const T* B, T* result, 
                        size_t m, size_t k, size_t n) {
                        
    // blockDim is useless here since we know it's tile width
    // threadIdx is the index within the tile
    // blockIdx is the index of the whole tile
    // gridDim is the dimension of tiles across the matrix
    
    size_t k_tiles = k / TILE_WIDTH + (k % TILE_WIDTH);

    // shared memory tiles
    __shared__ T sma[TILE_WIDTH][TILE_WIDTH];
    __shared__ T smb[TILE_WIDTH][TILE_WIDTH];

    // loop across phases of tiling
    // tile across the inner dim
    for (auto p=0; p<k_tiles; ++p) {
        // remember, tiling moves across row of a and col of b
        // each thread loads in the corresponding index of the tile
        
        // is thread index within matrix a? if so, load in value
        auto ax = TILE_WIDTH*p + threadIdx.x;
        auto ay = TILE_WIDTH*blockIdx.y + threadIdx.y;
        if ((ay < m) && (ax < k)) {
            sma[threadIdx.y][threadIdx.x] = A[k*ay + ax];
        } else {
            sma[threadIdx.y][threadIdx.x] = 0;
        }
        // load into shared matrix b
        // column of b
        auto bx = TILE_WIDTH*blockIdx.x + threadIdx.x;
        auto by = TILE_WIDTH*p + threadIdx.y;
        if ((by < k) && (bx < n)) {
            smb[threadIdx.y][threadIdx.x] = B[n*by + bx];
        } else {
            smb[threadIdx.y][threadIdx.x] = 0;
        }

        // need to wait for all threads to load before you actually multiply
        __syncthreads();

        // find your spot in the result matrix
        // this is the y of a and the x of b (mxn)
        // this should not be dependent on the phase
        if ((ay < m) && (bx < n)) {
            // dot product a row from shared memory
            T dot = 0;
            for (auto i=0; i<TILE_WIDTH; ++i)
                dot += sma[threadIdx.y][i]*smb[i][threadIdx.x];
            // remember, result matrix is mxn
            result[n*ay + bx] += dot;
        }
        
        __syncthreads();
    }
}



torch::Tensor tiled_matmul(torch::Tensor A, torch::Tensor B) {
    const auto m = A.size(0);
    const auto k = A.size(1);
    const auto n = B.size(1);
    assert(k == B.size(0) && "matrices must share inner dim");
    const auto dtype = A.dtype(); 

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(dtype);
    auto result = torch::zeros({m, n}, options);

    dim3 tile_dim(TILE_WIDTH, TILE_WIDTH);

    // tile across resultant matrix
    // only might add one more tile, not a bunch more
    size_t m_tiles = m / TILE_WIDTH + (m % TILE_WIDTH != 0);
    size_t n_tiles = n / TILE_WIDTH + (n % TILE_WIDTH != 0);
    dim3 grid_dim(n_tiles, m_tiles);

    if (dtype==torch::kFloat32) {
        tiled_matmul_kernel<float><<<grid_dim, tile_dim>>>(
            (float*)A.const_data_ptr(), (float*)B.const_data_ptr(), (float*)result.mutable_data_ptr(), m, k, n);
    } else if (dtype==torch::kFloat64) {
        tiled_matmul_kernel<double><<<grid_dim, tile_dim>>>(
            (double*)A.const_data_ptr(), (double*)B.const_data_ptr(), (double*)result.mutable_data_ptr(), m, k, n);
    } else if (dtype==torch::kInt32) {
        std::cout << "about to launch int32 kernel\n";
        tiled_matmul_kernel<int><<<grid_dim, tile_dim>>>(
            (int*)A.const_data_ptr(), (int*)B.const_data_ptr(), (int*)result.mutable_data_ptr(), m, k, n);
    } else if (dtype==torch::kInt64) {
        tiled_matmul_kernel<int64_t><<<grid_dim, tile_dim>>>(
            (int64_t*)A.const_data_ptr(), (int64_t*)B.const_data_ptr(), (int64_t*)result.mutable_data_ptr(), m, k, n);
    }
    return result;
}