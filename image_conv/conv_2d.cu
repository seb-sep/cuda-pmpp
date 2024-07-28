#include <cuda.h>

#include <torch/types.h>

#include <stdio.h>

const float PAD_VALUE = 0;
const size_t TILE_WIDTH = 32;

// You DON'T need to know the size of the mask beforehand, just allocate more than you will need
// and only memcpy over what you actually use
// Make the mask linearized, so that you can easily copy into it whatever you need
#define MAX_MASK_SIZE 64
__constant__ float MASK[MAX_MASK_SIZE];

__global__ void conv_2d_tiled(const float* N, float* P, 
                            int height, int width, int mask_width) {

    int halo_width = mask_width/2;
    
    // indices in the whole matrix
    int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x*TILE_WIDTH + threadIdx.x;
    
    // load into shared, ONLY for inner cells not halo
    __shared__ float N_s[TILE_WIDTH][TILE_WIDTH];
    if ((row < height) && (col < width)) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = PAD_VALUE;
    }
    
    __syncthreads();

    if ((row < height) && (col < width)) {
        float res = 0;
        // loop over the mask indices
        for (int i=0; i<mask_width; ++i) {
            for (int j=0; j<mask_width; ++j) {
                // check whether (i,j) is an inner or halo cell
                // N_row and col are the indices that a given element of the mask is overlapping
                float mask_factor = MASK[i*mask_width+j];
                int N_row = row + i - halo_width;
                int N_col = col + j - halo_width;

                int mask_row = threadIdx.y + i - halo_width;
                int mask_col = threadIdx.x + j - halo_width;

                if ((mask_row >= 0) && (mask_row < TILE_WIDTH)
                    && (mask_col >= 0) && (mask_col < TILE_WIDTH)) {
                        res += N_s[mask_row][mask_col] * mask_factor;
                } else {
                    // only get a halo cell if it's in bounds, otherwise pad
                    if ((N_row >= 0) && (N_row < height) && (N_col >= 0) && (N_col < width)) {
                        res += N[N_row*width + N_col] * mask_factor;
                    } else {
                        res += PAD_VALUE * mask_factor;
                    }
                }
            }
        }

        P[row * width + col] = res;
    }
}


torch::Tensor conv_2d(torch::Tensor matrix, torch::Tensor mask) {
    torch::Tensor res = torch::empty_like(matrix);

    int height = matrix.size(0);
    int width = matrix.size(1);
    int mask_size = mask.numel();

    assert(mask_size <= MAX_MASK_SIZE);
    assert(mask.size(0) == mask.size(1));

    // load into the mask
    // since the underlying buffer is always 1d and is row-major, it's okay to naively copy buffer
    cudaMemcpyToSymbol(MASK, (float*)mask.const_data_ptr(), sizeof(float)*mask_size);

    auto x_tiles = width / TILE_WIDTH + (width%TILE_WIDTH);
    auto y_tiles = height / TILE_WIDTH + (height%TILE_WIDTH);

    dim3 grid_dim(y_tiles, x_tiles);
    dim3 tile_dim(TILE_WIDTH, TILE_WIDTH);
    conv_2d_tiled<<<grid_dim, tile_dim>>>(
        (float*)matrix.const_data_ptr(), 
        (float*)res.mutable_data_ptr(), 
        height, width, mask.size(0));

    cudaDeviceSynchronize();
    
    return res;
}