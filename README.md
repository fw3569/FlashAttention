## Flash Attention CUDA Implementation
基于 CUTLASS 的 Flash Attention 自定义 CUDA 实现，使用 PyTorch 通过 ctypes 调用。

#### 环境
- **CUDA**: 12.4
- **CUTLASS**: 4.4.2
- **torch**: 2.6.0+cu124
- **CUDA Architecture**: sm_75（RTX mx450/Turing）

#### 构建前配置
硬编码了一些路径，使用前需根据本机环境修改

**./src/main.py**

- `cutlass_path` — cutlass include路径

#### 构建和运行
Windows + MSVC 环境下

``` bash
python main.py
```

#### 实现说明
提供三个 attention 变体：

`attention_forward` — Native attention（矩阵乘 + softmax + 矩阵乘），基线实现。

`flash_attention_simt_forward` — Flash Attention FP32，使用 CUTLASS SIMT warp MMA。

`flash_attention_tensor_op_forward` — Flash Attention FP16，使用 CUTLASS Tensor Core MMA。

#### 性能测试
**测试参数**：固定为`batch=2, heads=4, seq_len=512, head_dim=64`  
**测试方法**：耗时是在python打点`time.perf_counter`测试，10次warmup，100次运行取平均值。HBM读写量是使用ncu检查`dram__bytes_read.sum dram__bytes_write.sum`。

**Native Attention（FP32）**  
HBM 读 27.6M / 写 21.8M bytes
|实现|耗时|
|--|--|
|custom|0.887ms|
|pytorch sdpa|0.328 ms|
|加速比|0.37x|

**Flash Attention SIMT（FP32）**  
HBM 读 3.3M / 写 1.1M bytes，相比 sdpa 的 6.7M / 1.3M 有显著改善
|实现|耗时|
|--|--|
|custom|0.377 ms|
|pytorch sdpa|0.334 ms|
|加速比|0.89x|

**Flash Attention Tensor Op（FP16）**  
HBM 读 1.6M / 写 0.6M bytes，相比 sdpa 的 2.9M / 0.7M 进一步减少
|实现|耗时|
|--|--|
|custom|0.751 ms|
|pytorch sdpa|0.887 ms|
|加速比|1.18x|

FP16 版本相对 FP32 版本偏慢，原因推测是 V 矩阵需要额外转置（CUTLASS Tensor Core 要求 ColumnMajor B）。  
FP16 版本相对 FP32 版本加速比更高，推测是因为开发顺序靠后，有更多更仔细的修改和调参。

#### 其他
- head_dim 最大支持 64（SIMT/FP32）,如果要128版本需要重新调参数
- 没有做双缓冲是因为mx450实际不支持异步拷贝
- 考虑后续加上双缓冲版本并找个云端8.0+架构测试
