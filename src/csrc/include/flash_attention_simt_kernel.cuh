#pragma once
#include <torch/extension.h>

torch::Tensor flash_attention_simt_forward(torch::Tensor Q, torch::Tensor K,
                                           torch::Tensor V);
