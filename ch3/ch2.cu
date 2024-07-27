#include <stdio.h>

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {


    // alloc on device
    int size = n*sizeof(float);
    float *A_d, *B_d, *C_d;
    // cudaMalloc takes in the ADDRESS of a pointer, not a pointer itself
    // this pointer will be set to point to the new alloc'd object
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Remember, these are addresses on DEVICE, derefing them in host code won't do anything


    // copy vectors from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    /*
    call the kernel
    first specify the # of blocks in the grid, then # of threads per block
    since we launch with 256 threads per block, we divide n by 256 to figure out
    how many blocks we need to cover our n
    Can't make any assumptions about the order in which threads and blocks execute
    */
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // copy result vector from device to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // since we don't need to change the value of the pointer, we just need the pointer,
    // we don't need to deref the pointer to free the value
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void printVec(float* A_h, int n) {
    printf("[ ");
    for (int i=0; i<n; ++i) 
        printf("%.1f ", A_h[i]);
    printf("]\n");
}

/*
CUDA thread hierarchy:
- Thread grid, executing the same kernel across all threads
- Each grid is an array of thread blocks
- Each block contains the same number of threads, up to 1024
    - # of threads per dim of block should be a multiple of 32


The money with CUDA kernels is to figure out how to mathematically map your data to the structure of threads
- blockDim (.x, .y, .z): # of threads in a dimension of a block
- threadIdx (.x .y, .z): coordinate of thread within a block
- blockIdx (.x, .y, .z): coordinate of the block a thread is in
Thus, a thread is uniquely identified by its block and thread index
We often calculate in such a way, multiplying the block index by its dimension size (almost like a stride):
`blockIdx.x*blockDim.x + threadIdx.x`
This flattening lets us map across 1D arrays
*/


/*
we don't demarcate kernel vars with _h or _d
Global dunder means that kernel is executed on device,
but can be called from both host and device (Cuda Dynamic Parallelism)
Each kernel call launches a new grid of threads on the device

__device__ dunder can only be called on device, does NOT launch new threads
__host__ dunder is just a C fn executing on host
    A function is a host function if it contains no CUDA keywords
Using both device and host dunders generates a version of the function for both host and device
*/
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    /* 
    Why don't we just make sure to not launch more threads than we need?
    we never want to launch a block with less than 32 threads
    */
    if (i<n)
        C[i] = A[i] + B[i];
}

int main() {
    float A[] = {0.5, 0.2, 0.3};
    float B[] = {0.5, 0.2, 0.3};

    int n = 3;
    float C[n];
    vecAdd(A, B, C, n);

    printVec(A, n);
    printVec(B, n);
    printVec(C, n);

}

/*
NVCC compiles the .cu file into a pure ANSI C host executable and a PTX binary for the device

*/