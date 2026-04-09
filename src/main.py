import torch
import torch.nn.functional as F
import ctypes

lib = ctypes.CDLL("../build/Debug/attention_kernel.dll")
lib.attention_forward.argtypes = [
    ctypes.c_void_p,  # Q
    ctypes.c_void_p,  # K
    ctypes.c_void_p,  # V
    ctypes.c_void_p,  # O
    ctypes.c_int,     # batch
    ctypes.c_int,     # heads
    ctypes.c_int,     # seq_len
    ctypes.c_int,     # head_dim
]
lib.attention_forward.restype = None
lib.flash_attention_forward.argtypes = [
    ctypes.c_void_p,  # Q
    ctypes.c_void_p,  # K
    ctypes.c_void_p,  # V
    ctypes.c_void_p,  # O
    ctypes.c_int,     # batch
    ctypes.c_int,     # heads
    ctypes.c_int,     # seq_len
    ctypes.c_int,     # head_dim
]
lib.flash_attention_forward.restype = None

def naive_attention(Q, K, V, scale=None):
    if scale is None:
        scale = Q.shape[-1] ** -0.5

    score = torch.matmul(Q, K.transpose(-2, -1)) * scale

    attention = F.softmax(score, dim=-1)

    out = torch.matmul(attention, V)
    return out

def custom_attention(Q, K, V):
    assert Q.is_cuda and Q.is_contiguous()
    O = torch.zeros_like(Q)
    lib.flash_attention_forward(
        Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(),
        Q.shape[0], Q.shape[1], Q.shape[2], Q.shape[3]
    )
    return O

def pytorch_sdpa(Q, K, V, scale=None):
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)


def run_correctness_check(batch=2, heads=4, seq_len=512, head_dim=64, device="cuda"):
    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float32)

    ref = pytorch_sdpa(Q, K, V)
    out = custom_attention(Q, K, V)

    max_diff = (ref - out).abs().max().item()
    print(f"[correctness] custom vs sdpa: max_diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"correctness check failed: {max_diff}"
    print("[correctness] PASSED")
    return Q, K, V, ref


def run_benchmark(Q, K, V, warmup=10, iters=100):
    import time

    device = Q.device

    for _ in range(warmup):
        _ = custom_attention(Q, K, V)
        _ = pytorch_sdpa(Q, K, V)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = custom_attention(Q, K, V)
    torch.cuda.synchronize()
    custom_ms = (time.perf_counter() - start) / iters * 1000

    start = time.perf_counter()
    for _ in range(iters):
        _ = pytorch_sdpa(Q, K, V)
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - start) / iters * 1000

    print(f"\n[benchmark] seq_len={Q.shape[2]}, head_dim={Q.shape[3]}")
    print(f"  custom attention : {custom_ms:.3f} ms")
    print(f"  pytorch sdpa    : {sdpa_ms:.3f} ms")
    print(f"  speedup (sdpa/custom): {custom_ms/sdpa_ms:.2f}x")

    return {"custom_ms": custom_ms, "sdpa_ms": sdpa_ms}


def run_memory_check(batch=2, heads=4, head_dim=64, device="cuda"):
    print("\n[memory] peak allocated vs seq_len:")
    print(f"  {'seq_len':>8}  {'custom (MB)':>12}  {'sdpa (MB)':>12}")

    for seq_len in [256, 512, 1024]:
        torch.manual_seed(0)
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device)

        torch.cuda.reset_peak_memory_stats()
        _ = custom_attention(Q, K, V)
        torch.cuda.synchronize()
        custom_mb = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.reset_peak_memory_stats()
        _ = pytorch_sdpa(Q, K, V)
        torch.cuda.synchronize()
        sdpa_mb = torch.cuda.max_memory_allocated() / 1e6

        print(f"  {seq_len:>8}  {custom_mb:>12.1f}  {sdpa_mb:>12.1f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    Q, K, V, ref = run_correctness_check(device=device)
    run_benchmark(Q, K, V)
    run_memory_check(device=device)
