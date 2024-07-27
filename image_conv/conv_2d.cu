#include <cuda.h>

#include <torch/types.h>

const float PAD_VALUE = 0;
const size_t TILE_WIDTH = 32;

// You DON'T need to know the size of the mask beforehand, just allocate more than you will need
// and only memcpy over what you actually use
#define MAX_MASK_WIDTH 10
__constant__ float MASK[MAX_MASK_WIDTH][MAX_MASK_WIDTH];

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
            for (int j=0; j<mask_width; ++i) {
                // check whether (i,j) is an inner or halo cell
                int N_row = row + i - halo_width;
                int N_col = col + j - halo_width;
                int tile_row = row - threadIdx.y;
                int tile_col = col - threadIdx.x;

                if (N_row >= tile_row && N_row < tile_row + blockDim.y
                    && N_col >= tile_col && N_col < tile_col + blockDim.x) {
                        res += N_s[threadIdx.y - halo_width + i][threadIdx.x - halo_width + j] * MASK[i][j];
                } else {
                    // only get a halo cell if it's in bounds, otherwise pad
                    if (N_row >= 0 && N_row < height && N_col >= 0 && N_col < width) {
                        res += N[N_row*width + N_col] * MASK[i][j];
                    } else {
                        res += PAD_VALUE * MASK[i][j];
                    }
                }
            }
        }

        P[row * width + col] = res;
    }
}

torch::Tensor conv_2d(torch::Tensor vector, torch::Tensor stencil) {
    torch::Tensor res = torch::empty_like(vector);
    int width = vector.numel();
    int mask_width = stencil.numel();
    assert(mask_width <= MAX_MASK_WIDTH);

    // load into the mask
    cudaMemcpyToSymbol(MASK, (float*)stencil.const_data_ptr(), sizeof(float)*mask_width);

    auto n_tiles = width / TILE_WIDTH + (width%TILE_WIDTH);
    conv_1d_tiled<<<n_tiles, TILE_SIZE>>>(
        (float*)vector.const_data_ptr(), 
        (float*)res.mutable_data_ptr(), 
        mask_width, width);
    
    return res;
}