# WIP
flash attention，开发中

下面内容不保证最新

#### 构建

cmake构建dll后运行`python main.py`

#### 测试数据
native( 乘 + softmax + 乘 )。hbm读27,610,272，写17,635,328。
```
[benchmark] seq_len=512, head_dim=64
  custom attention : 0.836 ms
  pytorch sdpa    : 0.492 ms
  speedup (sdpa/custom): 1.70x

[memory] peak allocated vs seq_len:
   seq_len   custom (MB)     sdpa (MB)
       256           6.3           6.8
       512           8.9           9.4
      1024          13.6          14.7
      2048          23.1          25.2
```

masked native( 乘 + mask + softmax + 乘 )，以后默认都是masked。scaled_dot_product_attention在序列长度超过1024会挂掉，删了2048的测试。hbm读27,612,928，写21,795,296。
```
[benchmark] seq_len=512, head_dim=64
  custom attention : 0.887 ms
  pytorch sdpa    : 0.328 ms
  speedup (sdpa/custom): 2.70x

[memory] peak allocated vs seq_len:
   seq_len   custom (MB)     sdpa (MB)
       256           6.3           6.8
       512           8.9           9.4
      1024          13.6          14.7
```

flash attention 特征维度d设置最大64，再多算的是错的，mx450的smem太小。hbm读3,281,984，写1,098,208。参考spda的fmha读6,749,152，写1,274,624。
```
[benchmark] seq_len=512, head_dim=64
  custom attention : 0.377 ms
  pytorch sdpa    : 0.334 ms
  speedup (sdpa/custom): 1.13x

[memory] peak allocated vs seq_len:
   seq_len   custom (MB)     sdpa (MB)
       256           6.3           6.8
       512           8.9           9.4
      1024          13.6          14.7
```
