#include <torch/extension.h>

torch::Tensor unroll(torch::Tensor X, int K);
torch::Tensor conv2d(torch::Tensor X, torch::Tensor W_unroll, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("unroll", torch::wrap_pybind_function(unroll), "unroll");
    m.def("conv2d", torch::wrap_pybind_function(conv2d), "conv2d");
}