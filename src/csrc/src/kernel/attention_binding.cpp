#include "attention_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &attention_forward, "attention forward");
}
