#pragma once
#include <cutlass/half.h>
extern "C" {
void flash_attention_tensor_op_forward(cutlass::half_t* Q, cutlass::half_t* K,
                                       cutlass::half_t* V, cutlass::half_t* O,
                                       int batch, int heads, int seq_len,
                                       int head_dim);
}
