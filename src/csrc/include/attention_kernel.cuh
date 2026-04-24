#pragma once
#include <torch/extension.h>

torch::Tensor attention_forward(torch::Tensor Q, torch::Tensor K,
                                torch::Tensor V);
