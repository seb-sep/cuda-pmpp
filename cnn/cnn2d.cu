#include <cuda.h>
#include <torch/types.h>
#include <stdio.h>

__global__
void cnn_conv2d(const float* X, float* Y, const float* W, int M, int C, int H, int K) {

}

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


torch::Tensor unroll(torch::Tensor X, int K) {
    // launch kernel with flattened grid of C*h_out*w_out threads
    auto C = X.size(0);
    auto H = X.size(1);
    auto W = X.size(2);

    auto H_out = H - K + 1;
    auto W_out = W - K + 1;

    const auto dtype = X.dtype();
    auto options = torch::TensorOptions().device(X.device()).dtype(dtype);
    torch::Tensor X_unroll = torch::zeros({C*K*K, H_out*W_out}, options);

    int n_threads = C * H_out * W_out;
    int n_blocks = n_threads / CUDA_MAX_THREADS_PER_BLOCK + 1;

    std::cout << n_threads << " threads, " << n_blocks << " blocks, " << "\n";
    


    unroll_kernel<<<n_blocks, n_threads>>>(
        (float*)X.const_data_ptr(), (float*)X_unroll.mutable_data_ptr(), C, H, W, K);

    return X_unroll;
}