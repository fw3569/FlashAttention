#pragma once
#include <torch/extension.h>

torch::Tensor attention_forward(const torch::Tensor& Q, const torch::Tensor& K,
                                const torch::Tensor& V);
