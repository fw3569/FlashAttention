#pragma once
#include <torch/extension.h>

torch::Tensor flash_attention_tensor_op_forward(const torch::Tensor& Q,
                                                const torch::Tensor& K,
                                                const torch::Tensor& V);
