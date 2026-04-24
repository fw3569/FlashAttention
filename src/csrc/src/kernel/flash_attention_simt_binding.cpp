#include "flash_attention_simt_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attention_simt_forward,
        "flash attention simt forward");
}
