#include <cuda.h>
#include <torch/types.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>

// Print out a matrix that's on host.
void printMatrix(const float* M_h, uint width, uint height) {
    for (int j=0; j<height; ++j) {
        std::cout << "[ ";
        for (int i=0; i<width; ++i) {
            std::cout << M_h[width*j+i] << " ";  
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

curandStatus_t genRandMatrix(curandGenerator_t gen, float** M_d, uint width, uint height) {

    size_t size = sizeof(float) * width * height;
    cudaMalloc((void**)M_d, size);

    return curandGenerateUniform(gen, *M_d, width * height);
}


__global__
void matMul(const float* A, const float* B, float* C, int width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // check if the coordinate is within the square matrix
    if (row < width && col < width) {
        float prod = 0;
        // in a loop like this, using i++ will involve creating a temporary copy, since you return the value then increment
        for (int i=0; i<width; ++i) {
            // the ith value of the col of A, and the ith value of the row of B
            prod += A[width*row + i] * B[width*i + col];
        }
        C[row*width + col] = prod;
    }
}


torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    // for now, just get it working on square matrices
    assert((A.size(0) == A.size(1)) && (B.size(0) == B.size(1)) && (A.size(0) == B.size(0)) && "all tensors must be the same square shape");

    auto result = torch::empty_like(A);
    auto width = A.size(0);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (width + threads_per_block.y - 1) / threads_per_block.y);

    matMul<<<number_of_blocks, threads_per_block>>>(
        (float*)A.const_data_ptr(), (float*)B.const_data_ptr(), (float*)result.mutable_data_ptr(), width);

    return result;
}

// int main() {
//     // instantiate curand generator
//     curandGenerator_t gen;
//     curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//     curandSetPseudoRandomGeneratorSeed(gen, 420ULL);

//     float *A_d, *B_d, *C_d, *Ct_d;

//     uint width = 5;
//     uint height = 5;
    
//     // generate two random matrices
//     genRandMatrix(gen, &A_d, width, height);
//     genRandMatrix(gen, &B_d, width, height);

//     // allocate for the result matrix
//     uint size = sizeof(float) * width * height;
//     cudaMalloc((void**)&C_d, size);
//     cudaMalloc((void**)&Ct_d, size);

//     // perform the homemade matmul 
//     dim3 blockDim(8, 8); // square thread block that's a multiple of 32
//     dim3 gridDim(ceil(width/static_cast<float>(blockDim.x)), ceil(height/static_cast<float>(blockDim.y)));
//     matMul<<<gridDim, blockDim>>>(A_d, B_d, C_d, width);


//     // allocate tensor on host to move the result to
//     float* C_h = (float*)malloc(size);
//     cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

//     std::cout << "homemeade gemm\n";
//     printMatrix(C_h, width, height);

//     // perform cublas matmul
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     const float alpha = 1.0f;
//     const float beta = 0.0f;
//     // leading dimension: # of elements between successive rows/cols in MEMORY
//     // cublas is COLUMN major, so we transpose our row-major matrices with CUBLAS_OP_T
//     cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T, height, width, width, &alpha, A_d, width, B_d, width, &beta, Ct_d, width);
//     // The resultant is column major, so we transpose it
//     cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, width, height, &alpha, Ct_d, width, &beta, Ct_d, width, C_d, width);
//     cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
//     std::cout << "cublas gemm\n";
//     printMatrix(C_h, width, height);

//     cudaFree(A_d);
//     cudaFree(B_d);
//     cudaFree(C_d);
//     free(C_h);

//     // matmul simple 2x2
//     // float x_h[4] = {1.0, 1.0, 
//     //             1.0, 1.0};
//     // float y_h[4] = {2.0, 2.0, 
//     //             0.0, 0.0};
//     // float *x_d, *y_d, *z_d;
//     // cudaMalloc((void**)&x_d, sizeof(x_h));
//     // cudaMalloc((void**)&y_d, sizeof(y_h));
//     // cudaMalloc((void**)&z_d, sizeof(y_h));
//     // cudaMemcpy(x_d, x_h, sizeof(x_h), cudaMemcpyHostToDevice);
//     // cudaMemcpy(y_d, y_h, sizeof(y_h), cudaMemcpyHostToDevice);
//     // cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T, 2, 2, 2, &alpha, x_d, 2, y_d, 2, &beta, z_d, 2);
//     // float z_h[4] = {0.0, 0.0, 0.0, 0.0};
//     // cudaMemcpy(z_h, z_d, sizeof(x_h), cudaMemcpyDeviceToHost);
//     // printMatrix(z_h, 2, 2);

//     // cudaFree(x_d);
//     // cudaFree(y_d);
//     // cudaFree(z_d);

//     cublasDestroy(handle);
//     curandDestroyGenerator(gen);

// } 