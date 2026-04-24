#include "flash_attention_tensor_op_kernel.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attention_tensor_op_forward,
        "flash attention tensor op forward");
}
