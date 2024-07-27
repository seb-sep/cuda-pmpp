#include <torch/extension.h>

torch::Tensor matmul(torch::Tensor A, torch::Tensor B);
torch::Tensor square_matrix(torch::Tensor matrix);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("gemm", torch::wrap_pybind_function(matmul), "matmul");
m.def("square", torch::wrap_pybind_function(square_matrix), "square");
}