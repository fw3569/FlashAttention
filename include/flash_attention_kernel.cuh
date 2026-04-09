#pragma once
extern "C" {
void flash_attention_forward(float* Q, float* K, float* V, float* O, int batch,
                             int heads, int seq_len, int head_dim);
}
