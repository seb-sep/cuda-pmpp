#include <cuda.h>
#include <torch/types.h>
#include <stdio.h>

__global__
void unroll_kernel(const float* X, float* X_unroll, int C, int H, int W, int K) {
    // each thread gets kxk input elems
    // total threads C*Hout*Wout
    // 1d thread blocks
    // Each thread builds a single kxk section of a single column, 

    // flat thread grid?
    // t is which conv tile sized chunk of X you are unloading 
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    // H and W are the height and width of the original matrix
    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // the fact that we have a w_unroll suggests a parallelization over channels 
    // w_unroll is the width of the unrolled input matrix surprise surprise
    // this makes sense, because your unrolled matrix needs a column for each 
    // element in the output matrix
    int W_unroll = H_out * W_out;

    // do a 'row-major' access into a column chunk of the matrix
    // contiguous threads process contiguous columns of a single input channel
    if (t < C * W_unroll) {
        // first, figure out where we are in the output matrix
        // get the index of the channel the thread is computing in
        int c = t / W_unroll;

        // get the index of the column within a channel
        // does not change over the iteration
        // w_unroll is essentially the index of the output value
        int w_unroll = t % W_unroll;

        // in the input matrix, you're at [H_out, W_out]


        // need these because we want to get a square from the input matrix.
        // W_out is as far as we will go across in the original matrix before we wrap to
        // the next row
        int h_out = w_unroll / W_out;
        int w_out = w_unroll % W_out;

        int w_base = c*K*K;

        // loop over all the values within a mask
        for (int p=0; p<K; ++p) {
            for (int q=0; q<K; ++q) {

                // need w_base because you iterate over the size of the conv mask,
                // not the size of the output matrix
                int h_unroll = w_base + (p*K+q);

                X_unroll[h_unroll*W_unroll + w_unroll] = X[c*(H*W) + W*(h_out+p) + (w_out+q)];
            }
        }
    }
    __syncthreads();
}

const size_t TILE_WIDTH = 8;

// Remember that PyTorch tensors are row-major
__global__ 
void tiled_matmul_kernel(const float* A, const float* B, float* result, 
                        size_t m, size_t k, size_t n) {
                        
    // blockDim is useless here since we know it's tile width
    // threadIdx is the index within the tile
    // blockIdx is the index of the whole tile
    // gridDim is the dimension of tiles across the matrix
    
    size_t k_tiles = k / TILE_WIDTH + (k % TILE_WIDTH);

    // shared memory tiles
    __shared__ float sma[TILE_WIDTH][TILE_WIDTH];
    __shared__ float smb[TILE_WIDTH][TILE_WIDTH];

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
            float dot = 0;
            for (auto i=0; i<TILE_WIDTH; ++i)
                dot += sma[threadIdx.y][i]*smb[i][threadIdx.x];
            // remember, result matrix is mxn
            // multiply by the width of the row to properly linearlize access
            result[n*ay + bx] += dot;
        }
        
        __syncthreads();
    }
}


torch::Tensor unroll(torch::Tensor X, int K) {
    // launch kernel with flattened grid of C*h_out*w_out threads
    auto C = X.size(0);
    auto H = X.size(1);
    auto W = X.size(2);

    auto H_out = H - K + 1;
    auto W_out = W - K + 1;

    const auto dtype = X.dtype();
    auto options = torch::TensorOptions().device(X.device()).dtype(dtype);
    torch::Tensor X_unroll = torch::empty({C*K*K, H_out*W_out}, options);

    int n_threads = C * H_out * W_out;
    int n_blocks = n_threads / CUDA_MAX_THREADS_PER_BLOCK + 1;

    unroll_kernel<<<n_blocks, n_threads>>>(
        (float*)X.const_data_ptr(), (float*)X_unroll.mutable_data_ptr(), C, H, W, K);

    return X_unroll;
}

// Note that K is not actually implicit here; the number of convolutions present 
// in your weight matrix depends on the size of K
torch::Tensor conv2d(torch::Tensor X, torch::Tensor W_unroll, int K) {

    // unroll input matrix
    torch::Tensor X_unroll = unroll(X, K);

    std::cout <<  W_unroll << "\n" << X_unroll;

    // # of output channels
    auto m = W_unroll.size(0);
    // # of elements to multiply and add in each convolution
    auto k = W_unroll.size(1); 
    // # values in each channel (# of convolutions)
    auto n = X_unroll.size(1);

    const auto dtype = X.dtype();
    auto options = torch::TensorOptions().device(X.device()).dtype(dtype);

    // because the kernel accumulates in the result, must initialize with zeros
    torch::Tensor res = torch::zeros({m, n}, options);

    // from this point, a totally generic matmul
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH);
    size_t m_tiles = m / TILE_WIDTH + (m % TILE_WIDTH != 0);
    size_t n_tiles = n / TILE_WIDTH + (n % TILE_WIDTH != 0);
    dim3 grid_dim(n_tiles, m_tiles);

    tiled_matmul_kernel<<<grid_dim, block_dim>>>(
        (float*)W_unroll.const_data_ptr(), (float*)X_unroll.const_data_ptr(), 
        (float*)res.mutable_data_ptr(), m, k, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    return res;
}

