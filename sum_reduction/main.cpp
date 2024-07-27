#include <torch/extension.h>

torch::Tensor sum_reduce(torch::Tensor matrix);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("sum", torch::wrap_pybind_function(sum_reduce), "sum");
}