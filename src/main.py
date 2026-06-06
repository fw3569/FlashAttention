import sys
import math
import torch
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention as xformers_mea
from xformers.ops import LowerTriangularMask
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

    score = F.softmax(score, dim=-1)

    out = torch.matmul(score, V)
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

# [B, S, H, D]
def xformers_attention(Q, K, V, scale=None):
    out = xformers_mea(
        Q, K, V,
        attn_bias=LowerTriangularMask(),
        scale=scale,
    )
    return out

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

    torch.set_printoptions(profile="full", linewidth=120)
    max_diff = (ref - out).abs().max().item() * (ref.abs().max().item() * out.abs().max().item() + 1e-6)**-0.5
    print(f"[correctness] custom vs sdpa: max_diff = {max_diff:.2e}")
    assert max_diff < 1e-3, f"correctness check failed: {max_diff}"
    print("[correctness] PASSED")

def run_benchmark(batch=4, heads=4, seq_len = 512, head_dim=128, warmup=100, iters=100, device="cuda", dtype = torch.half):
    b, h, s, d = batch, heads, seq_len, head_dim
    Q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    K = torch.randn(b, h, s, d, device=device, dtype=dtype)
    V = torch.randn(b, h, s, d, device=device, dtype=dtype)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    def measure(fn, Q, K, V, warmup, iters):
        for _ in range(warmup):
            fn(Q, K, V)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(iters):
            fn(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / iters

    custom_native_ms = measure(custom_native_attention, Q, K, V, warmup, iters) if (seq_len * seq_len * batch * heads * head_dim <= 2**32) else 0.
    custom_simt_ms = measure(custom_simt_attention, Q, K, V, warmup, iters) if dtype == torch.float32 else 0.
    custom_tensor_op_ms = measure(custom_tensor_op_attention, Q, K, V, warmup, iters) if dtype == torch.float16 else 0.
    sdpa_math_ms = measure(pytorch_sdpa_math, Q, K, V, warmup, iters) if (seq_len * seq_len * batch * heads * head_dim <= 2**32) else 0.
    sdpa_mem_efficient_ms = measure(pytorch_sdpa_mem_efficient, Q, K, V, warmup, iters)

    # wrong anwser to redefine data, only for benchmark
    Q_p = Q.view(b, s, h, d)
    K_p = K.view(b, s, h, d)
    V_p = V.view(b, s, h, d)
    xformers_ms = measure(xformers_attention, Q_p, K_p, V_p, warmup, iters)

    return custom_native_ms, custom_simt_ms, custom_tensor_op_ms, sdpa_math_ms, sdpa_mem_efficient_ms, xformers_ms

def benchmark(warmup=100, iters=100, device="cuda", dtype = torch.half):

    print(f"\n[benchmark]")
    print(f"| seq_len | head_dim | custom native | custom simt | custom tensor | sdpa math | sdpa mem efficient | xformers |")
    token_size = 32768
    dim = 256
    for seq_len in [2048, 4096, 8192, 16384, 32768]:
        if seq_len > token_size:
            break
        for head_dim in [64, 128]:
            custom_native_ms, custom_simt_ms, custom_tensor_op_ms, sdpa_math_ms, sdpa_mem_efficient_ms, xformers_ms = run_benchmark(math.ceil(token_size / seq_len), math.ceil(dim / head_dim), seq_len, head_dim, warmup = warmup, iters = iters, device=device, dtype=dtype)
            print(f"| {str(seq_len):>7.7} | {str(head_dim):>8.8} | {str(custom_native_ms):>13.13} | {str(custom_simt_ms):>11.11} | {str(custom_tensor_op_ms):>13.13} | {str(sdpa_math_ms):>9.9} | {str(sdpa_mem_efficient_ms):>18.18} | {str(xformers_ms):>8.8} |")

def run_memory_check(batch=1, heads=1, seq_len = 512, head_dim=128, device="cuda", dtype = torch.half):
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    custom_native_attention(Q, K, V)
    if Q.dtype == torch.float32:
        custom_simt_attention(Q, K, V)
    if Q.dtype == torch.half:
        custom_tensor_op_attention(Q, K, V)
    pytorch_sdpa_math(Q, K, V)
    pytorch_sdpa_mem_efficient(Q, K, V)
    b, h, s, d = Q.shape
    # wrong anwser to redefine data, only for benchmark
    Q_p = Q.view(b, s, h, d)
    K_p = K.view(b, s, h, d)
    V_p = V.view(b, s, h, d)
    xformers_attention(Q_p, K_p, V_p)

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
