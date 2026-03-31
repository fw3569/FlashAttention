#pragma once
extern "C" {
void attention_forward(float* Q, float* K, float* V, float* O, int batch,
                       int heads, int seq_len, int head_dim);
}
