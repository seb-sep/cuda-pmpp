
#include <torch/extension.h>

torch::Tensor conv_1d(torch::Tensor vector, torch::Tensor stencil);
torch::Tensor conv_2d(torch::Tensor vector, torch::Tensor stencil);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("conv1d", torch::wrap_pybind_function(conv_1d), "conv1d");
m.def("conv2d", torch::wrap_pybind_function(conv_2d), "conv2d");
}

