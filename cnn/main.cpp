#include <torch/extension.h>

torch::Tensor unroll(torch::Tensor X, int K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("unroll", torch::wrap_pybind_function(unroll), "unroll");
}