#pragma once
#include <torch/extension.h>

torch::Tensor flash_attention_tensor_op_forward(torch::Tensor Q,
                                                torch::Tensor K,
                                                torch::Tensor V);
