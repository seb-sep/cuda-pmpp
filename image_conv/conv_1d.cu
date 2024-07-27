#include <cuda.h>

#include <torch/types.h>

const float PAD_VALUE = 0;
const size_t TILE_SIZE = 32;

// You DON'T need to know the size of the mask beforehand, just allocate more than you will need
// and only memcpy over what you actually use
#define MAX_MASK_WIDTH 10
__constant__ float MASK[MAX_MASK_WIDTH];

__global__ void conv_1d_tiled(const float* N, float* P, int mask_width, int width) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    // don't worry about shared for halo cells; they will be cached in L2
    __shared__ float N_s[TILE_SIZE];
    if (idx < width) {
        N_s[threadIdx.x] = N[idx];
    } else {
        N_s[threadIdx.x] = PAD_VALUE;
    }

    __syncthreads();

    int cur_tile_idx = blockIdx.x * blockDim.x;
    int next_tile_idx = (blockIdx.x+1) * blockDim.x;
    
    // where the computation for a dot product actually starts because of mask width
    int mask_offset = idx - (mask_width/2); 

    float res = 0;
    for (int j=0; j<mask_width; ++j) {
        int N_idx = mask_offset + j;
        // bounds check
        if (N_idx >= 0 && N_idx < width) {
            // halo or inner check
            if (N_idx >= cur_tile_idx && N_idx < next_tile_idx) {
                res += N_s[threadIdx.x + j-(mask_width/2)] * MASK[j];
            } else {
                res += N[N_idx] * MASK[j];
            }
        } else {
            res += PAD_VALUE * MASK[j];
        }
    }

    P[idx] = res;
    
}

__global__ void conv_1d_halo_tiled(const float* N, float* P, int mask_width, int width) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    // load tile of vector into shared
    // size of shared memory must be statically known
    __shared__ float N_s[TILE_SIZE + MAX_MASK_WIDTH-1];
    // load left halo cells
    int halo_width = mask_width/2;
    // halo_idx_left is some index into the previous tile's inner cells
    int halo_idx_left = (blockIdx.x-1)*blockDim.x + threadIdx.x;
    if (threadIdx.x >= blockDim.x - halo_width) { // if you're on the far right side of the tile
        // load into the far left side of shared either the right value from the matrix or 0
        N_s[threadIdx.x - (blockDim.x - halo_width)] = (halo_idx_left < 0) ? PAD_VALUE : N[halo_idx_left];
    }

    // now, load center cells
    if (idx < width) {
        N_s[threadIdx.x+halo_width] = N[idx];
    } else {
        N_s[threadIdx.x+halo_width] = PAD_VALUE;
    }
    
    // now, load right halo
    int halo_idx_right = (blockIdx.x+1)*blockDim.x + threadIdx.x;
    if (threadIdx.x < halo_width) { // if you're on the far left side of the tile
        // load into far right sided of shared
        // add block dim, because the shared tile is WIDER than block dim
        // you need to add a halo width because theres a halo_width on BOTH sides of the tile
        N_s[blockDim.x + halo_width + threadIdx.x] = (halo_idx_right >= width) ? PAD_VALUE : N[halo_idx_right];
    }

    __syncthreads();
    
    float res = 0;
    if (idx < width) {
        for (int j=0; j<mask_width; ++j)
            res += N_s[threadIdx.x+j] * MASK[j];
        
        P[idx] = res;
    }
}


torch::Tensor conv_1d(torch::Tensor vector, torch::Tensor stencil) {
    torch::Tensor res = torch::empty_like(vector);
    int width = vector.numel();
    int mask_width = stencil.numel();
    assert(mask_width <= MAX_MASK_WIDTH);

    // load into the mask
    cudaMemcpyToSymbol(MASK, (float*)stencil.const_data_ptr(), sizeof(float)*mask_width);

    auto n_tiles = width / TILE_SIZE + (width%TILE_SIZE);
    // convolution_1D_tiled_kernel<<<n_tiles, TILE_SIZE>>>(
    conv_1d_tiled<<<n_tiles, TILE_SIZE>>>(
        (float*)vector.const_data_ptr(), 
        (float*)res.mutable_data_ptr(), 
        mask_width, width);
    
    return res;
}