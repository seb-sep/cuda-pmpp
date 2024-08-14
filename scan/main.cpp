#include <torch/extension.h>

torch::Tensor add_scan(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("scan", torch::wrap_pybind_function(add_scan), "scan");
}