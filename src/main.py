import os
import sys
import time
import math
import torch
import torch.nn.functional as F
sys.path.insert(0, "./build")
import attention_extension
import flash_attention_simt_extension
import flash_attention_tensor_op_extension
# force fp32 accum
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

def naive_attention(Q, K, V, scale=None):
    if scale is None:
        scale = Q.shape[-1] ** -0.5

    score = torch.matmul(Q, K.transpose(-2, -1)) * scale

    attention = F.softmax(score, dim=-1)

    out = torch.matmul(attention, V)
    return out

def custom_native_attention(Q, K, V):
    return attention_extension.forward(Q, K, V)

def custom_simt_attention(Q, K, V):
    return flash_attention_simt_extension.forward(Q, K, V)

def custom_tensor_op_attention(Q, K, V):
    return flash_attention_tensor_op_extension.forward(Q, K, V)

def custom_attention(Q, K, V):
    return flash_attention_tensor_op_extension.forward(Q, K, V)

#unsupport in sm75
def pytorch_sdpa_flash(Q, K, V, scale=None):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)

def pytorch_sdpa_math(Q, K, V, scale=None):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)

#default and fastest backend in mx450
def pytorch_sdpa_mem_efficient(Q, K, V, scale=None):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)

#unsupport in sm75
def pytorch_sdpa_cudnn(Q, K, V, scale=None):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)

def pytorch_sdpa(Q, K, V, scale=None):
    return pytorch_sdpa_mem_efficient(Q, K, V, scale)


def run_correctness_check(batch=2, heads=4, seq_len=512, head_dim=128, device="cuda", dtype = torch.half):
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    ref = pytorch_sdpa(Q, K, V)
    if Q.dtype == torch.half:
        out = custom_tensor_op_attention(Q, K, V)
    elif Q.dtype == torch.float32:
        out = custom_simt_attention(Q, K, V)

    max_diff = (ref - out).abs().max().item() * (ref.abs().max().item() * out.abs().max().item() + 1e-6)**-0.5
    print(f"[correctness] custom vs sdpa: max_diff = {max_diff:.2e}")
    assert max_diff < 1e-3, f"correctness check failed: {max_diff}"
    print("[correctness] PASSED")

def run_benchmark(batch=4, heads=4, seq_len = 512, head_dim=128, warmup=100, iters=100, device="cuda", dtype = torch.half):
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    if seq_len <= 2048:
        for _ in range(warmup):
            _ = custom_native_attention(Q, K, V)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = custom_native_attention(Q, K, V)
        torch.cuda.synchronize()
        custom_native_ms = (time.perf_counter() - start) / iters * 1000
    else :
        custom_native_ms = 0.

    if Q.dtype == torch.float32:
        for _ in range(warmup):
            _ = custom_simt_attention(Q, K, V)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = custom_simt_attention(Q, K, V)
        torch.cuda.synchronize()
        custom_simt_ms = (time.perf_counter() - start) / iters * 1000
    else:
        custom_simt_ms = 0.


    if Q.dtype == torch.half:
        for _ in range(warmup):
            _ = custom_tensor_op_attention(Q, K, V)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = custom_tensor_op_attention(Q, K, V)
        torch.cuda.synchronize()
        custom_tensor_op_ms = (time.perf_counter() - start) / iters * 1000
    else:
        custom_tensor_op_ms = 0.

    if seq_len <= 2048:
        for _ in range(warmup):
            _ = pytorch_sdpa_math(Q, K, V)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _ = pytorch_sdpa_math(Q, K, V)
        torch.cuda.synchronize()
        sdpa_math_ms = (time.perf_counter() - start) / iters * 1000
    else:
        sdpa_math_ms = 0.

    for _ in range(warmup):
        _ = pytorch_sdpa_mem_efficient(Q, K, V)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = pytorch_sdpa_mem_efficient(Q, K, V)

    torch.cuda.synchronize()
    sdpa_mem_efficient_ms = (time.perf_counter() - start) / iters * 1000

    return custom_native_ms, custom_simt_ms, custom_tensor_op_ms, sdpa_math_ms, sdpa_mem_efficient_ms

def benchmark(warmup=100, iters=100, device="cuda", dtype = torch.half):

    print(f"\n[benchmark]")
    print(f"| seq_len | head_dim | custom native | custom simt | custom tensor | sdpa math | sdpa mem efficient |")
    token_size = 8192
    dim = 256
    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        if seq_len > token_size:
            break
        for head_dim in [64, 128]:
            custom_native_ms, custom_simt_ms, custom_tensor_op_ms, sdpa_math_ms, sdpa_mem_efficient_ms = run_benchmark(math.ceil(token_size / seq_len), math.ceil(dim / head_dim), seq_len, head_dim, warmup = warmup, iters = iters, device=device, dtype=dtype)
            print(f"| {str(seq_len):>7.7} | {str(head_dim):>8.8} | {str(custom_native_ms):>13.13} | {str(custom_simt_ms):>11.11} | {str(custom_tensor_op_ms):>13.13} | {str(sdpa_math_ms):>9.9} | {str(sdpa_mem_efficient_ms):>18.18} |")

def run_memory_check(batch=1, heads=1, seq_len = 512, head_dim=128, device="cuda", dtype = torch.half):
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = custom_native_attention(Q, K, V)
    torch.cuda.synchronize()

    if Q.dtype == torch.float32:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = custom_simt_attention(Q, K, V)
        torch.cuda.synchronize()

    if Q.dtype == torch.half:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _ = custom_tensor_op_attention(Q, K, V)
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = pytorch_sdpa_math(Q, K, V)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _ = pytorch_sdpa_mem_efficient(Q, K, V)
    torch.cuda.synchronize()

# use ncu to run this function
def memory_check(batch=1, heads=1, device="cuda", dtype = torch.half):
    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        for head_dim in [64, 128]:
            run_memory_check(batch, heads, seq_len, head_dim, device = device, dtype = dtype)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    dtype = torch.float16
    print(f"dtype: {dtype}\n")
    run_correctness_check(device=device, dtype = dtype)
    benchmark(device=device, dtype = dtype)
    # memory_check(device=device, dtype = dtype)
