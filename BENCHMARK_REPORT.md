# PDSDP Julia 与 C++/CUDA 实现：Mittelmann-75 完整实验报告

日期：2026-09-15  
实验数据：2026-09-14 启动的两轮 Mittelmann-75 全量测试

## 1. 摘要

本报告比较原始 Julia/CUDA.jl PDSDP 与新实现的原生 C++17/CUDA
PDSDP，并同时评估 C++/CUDA 版本相对于 Cardal v294 的表现。

主要结论如下：

1. **当前 C++/CUDA 实现明显优于原始 Julia 实现。**按独立验证器的
   `p/d/g <= 1e-4` 容差口径，C++/CUDA 得到 35 个可行端点，Julia
   得到 16 个；Julia 的 16 个可行端点全部也被 C++/CUDA 解出，且
   C++/CUDA 的 solver time 和端点生成 wall time 均为 16/16 更快。
2. 在上述 16 个共同可行问题上，C++/CUDA 的 solver time 中位加速为
   2.48x、几何平均加速为 3.21x、累计时间加速为 1.50x；端点生成
   wall time 的中位加速为 8.41x、累计加速为 1.99x。
3. C++/CUDA 的稳定性也更好：35 个 native `CONVERGED`，没有
   `NUMERICAL_ERROR`；Julia 为 15 个 native `CONVERGED`、11 个
   `NUMERICAL_ERROR`，另有 47 个 `OUT_OF_TIME` 和 2 个无 native
   状态的记录。
4. C++/CUDA 仍不能替代 Cardal。按 `p/d/g <= 1e-4`，C++/CUDA 仅解出
   35/75，Cardal v294 为 70/75；C++/CUDA 没有新增解决 Cardal 未解决
   的问题，且 40 个问题超时。
5. C++/CUDA 在其能解出的子集上很有竞争力。双方都可行的 35 个
   C++/CUDA–Cardal 问题中，C++/CUDA 有 25 个 solver time 更快，累计
   solver time 约快 2.03x。
6. “大量极小 cone”假设得到强但非普遍的支持：`chs_5000` 和
   `ros_2000` 相对 Cardal 分别达到 2431.7x 和 258.2x solver 加速，
   但结构类似的 `mater-6` 在两倍预算内仍未收敛。

因此，当前 C++/CUDA 版本适合作为结构特化 solver 或 portfolio
solver；它已经全面取代 Julia 版作为后续 PDSDP 工程开发基线，但尚不适合
单独取代 Cardal。

## 2. 归档内容与原始证据

本目录包含：

- [`julia-original/`](julia-original/)：Julia/CUDA.jl 源码快照；
- [`cpp-cuda/`](cpp-cuda/)：C++17/CUDA 源码、运行器、验证包装器及
  README；
- 两个实现各自的 `benchmark-summary/`：manifest、comparison.csv、
  progress、最终状态和调度日志。

大型 checkpoint、每题 solver log 和 validation JSON 没有重复复制。
权威原始结果保留在：

```text
/DATA/disk2/wyc/sdp_project/PDSDP-main/benchmark_runs/full75_2x_v294_cuda128_20260914
/DATA/disk2/wyc/sdp_project/PDSDP-cuda/benchmark_runs/full75_2x_v294_gpu0_1_20260914
```

关键结果文件 SHA-256：

| artifact | SHA-256 |
|---|---|
| Julia `comparison.csv` | `65d095ae473c25b2f6b2f39f7b11ed900b3cb5fb7e7f55b777807f9811837a4f` |
| Julia `campaign_status.json` | `3a819b40a9d08cc0873714d1e7808c0385ff9f9482fab90517c8f768c23cbb20` |
| C++ `comparison.csv` | `b6b9bc2684ba8bc8e693f52d3cf65f5898a7e77ae2b4bb50b9db1e0a57610c9f` |
| C++ `campaign_status.json` | `a6f85337fbc3fbfb99d3633ded1f8277653cbf35a9dbf3010eaa0cef5ece4913` |

关键实现 SHA-256：

| implementation artifact | SHA-256 |
|---|---|
| Julia `mittelmann_runner.jl` | `6461b69cae19066de622f75c2b7250a50360b19d5f0efc76197b64bebe2da164` |
| Julia `src/halpern_pdhg.jl` | `215dbc06c2f4628236ad10b9cc8ef1e15a83788e08d0995de56b46fdb15ab614` |
| Julia `src/pdhg_core.jl` | `2ea68c79cfb3bda5433d5c57515a2114adfcaacac1b1f959b10f3ee315d6300e` |
| C++ `src/pdsdp_cuda.cu` | `54a03efffa68035dc574775201d5282bd578884685fb142b7176542d65652818` |
| C++ tested binary | `5d6bd9d92044b8025f635f6c76e2ba8c5678405841e6993c915d4ca2ee1049bc` |
| C++ campaign `dimacs.py` | `81b87fb9a4905ba8eebc007b042e33a7f99aabefcb9c98cf814e8ad19bd19613` |

归档中的上述源文件 hash 已与实验 manifest 核对一致。

## 3. 实验设计

### 3.1 输入、预算与硬件

两轮实验满足以下共同条件：

- 数据集：本地 Mittelmann SDPA-sparse 75 题；
- 逐题输入文件 SHA-256 完全一致；
- 每题 PDSDP 时间预算：`2 * Cardal v294 solver_time_sec`；
- 两份 manifest 的 75 个预算逐项完全一致；
- 迭代上限：1,000,000,000；
- 精度阈值：`1e-4`；
- CUDA runtime：12.8；
- GPU：NVIDIA H100 80GB HBM3，每个求解实例独占一张卡。

Julia 实验主要使用物理 GPU 6，最后 15 题按审计记录分片到其他空闲
H100。C++/CUDA 实验先使用 GPU 0/1，随后扩展到 0/1/2/7。卡号不同，
但所有卡型号一致。并行调度改变整轮实验的完成时间，不改变逐题记录的
solver time。

### 3.2 时间字段

本报告区分：

- **solver time**：求解器内部记录的 termination time；
- **端点生成 wall time**：从求解进程启动到 checkpoint 生成完毕的
  wall time，包括读取、初始化/JIT、求解和导出，但不包括离线独立验证；
- **独立验证时间**：仅用于确认结果，不计入上述两项。

Cardal 对比中的 wall 加速使用 `Cardal solver time / PDSDP 端点生成 wall
time`，因为基线 manifest 提供的是 Cardal solver time。报告中不会把它
描述为两个程序完全对称的进程 wall-time 对比。

### 3.3 停止与精度定义

两个 PDSDP 实现都以原问题坐标下的三项指标作为停止依据：

```text
p = ||A(X)-b||_2 / (1+||b||_inf)
d = dist(C+A'*lambda, K) / (1+||C||_inf)
g = |<C,X>+<b,lambda>| / (1+|<C,X>|+|<b,lambda>|)
```

C++/CUDA 每 40 次迭代检查一次，当 `max(p,d,g) <= 1e-4` 时报告
`CONVERGED`。独立验证器重新读取原始 SDPA 文件和导出解，报告 DIMACS
`err1`、`err4`、`err5` 对应的 `p`、`d`、`g`。

本报告同时保留两种验收口径：

1. **DIMACS-tolerance feasible**：独立验证得到有限的
   `p,d,g <= 1e-4`；这是报告的主要工程性能口径。
2. **transport-strict**：除上述条件外，还要求 dual transport 有效，
   且把完整 primal matrix 转换为 PSD factor 时不需要超过 `1e-10` 的
   primal 投影修正。这是此前采用的更保守口径。

`formal accepted` 还要求 native 状态为 `CONVERGED`、不超过逐题预算，
且 solver time 不超过既有的 3600 秒正式时间门槛。

## 4. 两个实现的工程差异

### 4.1 Julia/CUDA.jl 原版

Julia 版本采用模块化实现，依赖 CUDA.jl、CUSOLVER、CUSPARSE、MAT、
Arpack 等包。Mittelmann adapter 将 SDPA block 转换为 PDSDP 的 packed
symmetric 表示，使用 Halpern/rAPDHG、Ruiz/BC/Pock–Chambolle scaling，
并导出 MAT checkpoint。

优点是算法结构清晰、便于实验；主要性能成本来自 Julia 进程和 JIT 启动、
GPU 调用层开销，以及迭代路径中的对象和调度开销。

### 4.2 C++17/CUDA 新实现

C++/CUDA 版本把完整求解路径收敛到一个原生 CUDA 可执行文件，主要工程
设计包括：

- 持久化 cuSPARSE descriptor、workspace 和 device vector；
- 每次 rAPDHG trial 只做一次 `A'` 和一次 `A` sparse product，并复用
  线性组合；
- 小 cone 使用分块 `syevjBatched`，大 cone 使用 `syevd`；
- 1x1/LP cone 使用标量 CUDA kernel；
- 迭代循环中不分配内存，并融合向量 kernel；
- 在原坐标中计算停止残差；
- 检查每次 cuSOLVER projection 状态；
- 原子发布 checkpoint，保留原 cone 顺序和原尺度 dual。

完整实验后，调度器进一步拆成动态 GPU 求解队列和有界 CPU 独立验证
队列。GPU 发布 checkpoint 后立即领取下一题，验证不再占住 GPU worker。

## 5. Julia 与 C++/CUDA 的总体结果

### 5.1 求解和精度结果

| 指标 | Julia | C++/CUDA |
|---|---:|---:|
| 总问题数 | 75 | 75 |
| native `CONVERGED` | 15 | 35 |
| `OUT_OF_TIME` | 47 | 40 |
| `NUMERICAL_ERROR` | 11 | 0 |
| 无 native 状态记录 | 2 | 0 |
| 独立 `p/d/g <= 1e-4` 端点 | 16 | 35 |
| `p/d/g` 可行且 native 收敛 | 14 | 35 |
| transport-strict | 11 | 24 |
| transport-strict 且 formal accepted | 9 | 24 |

Julia 的 16 个 `p/d/g` 可行端点全部包含在 C++/CUDA 的 35 个可行端点
内；C++/CUDA 新增 19 个，Julia 独有 0 个。Julia 的 `neu2` 和
`1dc.1024` 在时间限制处仍生成了可行端点，但状态为 `OUT_OF_TIME`；
C++/CUDA 在这两题上均正常报告 `CONVERGED`。

### 5.2 共同可行 16 题的逐题性能

“加速”均为 `Julia time / C++ time`，大于 1 表示 C++/CUDA 更快。

| instance | Julia solver (s) | C++ solver (s) | solver 加速 | Julia wall (s) | C++ wall (s) | wall 加速 | 状态变化 |
|---|---:|---:|---:|---:|---:|---:|---|
| `neu3` | 5.813 | 0.368 | 15.81x | 50.071 | 1.403 | 35.68x | converged → converged |
| `chs_5000` | 6.968 | 0.444 | 15.70x | 51.428 | 1.577 | 32.62x | converged → converged |
| `Bex2_1_5` | 12.416 | 1.793 | 6.93x | 57.283 | 3.030 | 18.90x | converged → converged |
| `Bst_jcbpaf2` | 14.243 | 2.201 | 6.47x | 58.132 | 3.598 | 16.16x | converged → converged |
| `ros_2000` | 14.093 | 2.596 | 5.43x | 57.133 | 3.499 | 16.33x | converged → converged |
| `theta102` | 7.376 | 1.660 | 4.44x | 51.328 | 2.565 | 20.01x | converged → converged |
| `neu3g` | 10.330 | 3.590 | 2.88x | 56.534 | 4.780 | 11.83x | converged → converged |
| `1zc.1024` | 11.568 | 4.582 | 2.52x | 56.279 | 5.706 | 9.86x | converged → converged |
| `neu1` | 143.652 | 59.114 | 2.43x | 187.181 | 60.110 | 3.11x | converged → converged |
| `fap09` | 18.298 | 7.983 | 2.29x | 62.036 | 8.928 | 6.95x | converged → converged |
| `NH4+.1A1.STO6GN.r18` | 76.727 | 44.551 | 1.72x | 123.502 | 46.109 | 2.68x | converged → converged |
| `rose15` | 193.593 | 115.298 | 1.68x | 236.895 | 118.423 | 2.00x | converged → converged |
| `neu2` | 504.259 | 303.220 | 1.66x | 547.101 | 304.197 | 1.80x | timeout → converged |
| `rabmo` | 95.508 | 61.144 | 1.56x | 138.438 | 61.992 | 2.23x | converged → converged |
| `taha1a` | 94.052 | 72.512 | 1.30x | 139.426 | 74.231 | 1.88x | converged → converged |
| `1dc.1024` | 815.130 | 669.935 | 1.22x | 860.000 | 671.185 | 1.28x | timeout → converged |

汇总：

| 统计量 | solver time | 端点生成 wall time |
|---|---:|---:|
| C++ 更快题数 | 16/16 | 16/16 |
| 中位加速 | 2.48x | 8.41x |
| 几何平均加速 | 3.21x | 6.67x |
| `sum(Julia)/sum(C++)` | 1.50x | 1.99x |

累计 solver 加速低于几何平均，是因为 `neu2` 和 `1dc.1024` 两个长题
占总时间较大，而它们的加速分别为 1.66x 和 1.22x。

### 5.3 相同迭代次数的证据

若只比较最终耗时，差异可能同时来自每次迭代速度和收敛轨迹。以下三题
两版迭代次数完全相同，因而能更直接反映实现开销：

| instance | Julia iterations | C++ iterations | Julia solver | C++ solver | 每迭代加速 |
|---|---:|---:|---:|---:|---:|
| `neu3` | 80 | 80 | 5.813 s | 0.368 s | 15.81x |
| `chs_5000` | 240 | 240 | 6.968 s | 0.444 s | 15.70x |
| `theta102` | 320 | 320 | 7.376 s | 1.660 s | 4.44x |

另外，`rose15` 的 C++ 版本实际执行了更多迭代（91,120 对 84,000），
solver time 仍从 193.6 秒降到 115.3 秒。这些结果说明性能提升不只是停止
点变化，原生实现的每迭代效率确实更高。

### 5.4 C++/CUDA 新增的 19 个 `p/d/g` 可行问题

| instance | Julia 状态 | Julia solver (s) | C++ 状态 | C++ solver (s) | C++ `(p,d,g)` |
|---|---|---:|---|---:|---|
| `hamming_8_3_4` | timeout | 4.047 | converged | 0.299 | `(3.84e-6, 0, 8.29e-6)` |
| `cphil12` | timeout | 4.065 | converged | 0.352 | `(1.34e-6, 1.72e-7, 4.70e-7)` |
| `prob_2_4_1` | timeout | 4.151 | converged | 0.559 | `(2.95e-5, 3.13e-6, 2.32e-5)` |
| `prob_2_4_0` | timeout | 4.775 | converged | 0.565 | `(8.41e-6, 2.84e-6, 1.12e-6)` |
| `hamming_9_5_6` | timeout | 3.834 | converged | 1.239 | `(2.03e-7, 0, 6.20e-7)` |
| `theta123` | timeout | 988.902 | converged | 1.859 | `(2.06e-8, 3.52e-5, 3.13e-8)` |
| `cnhil10` | timeout | 4.805 | converged | 2.274 | `(7.22e-5, 3.40e-5, 9.27e-5)` |
| `theta12` | timeout | 26.351 | converged | 2.396 | `(2.12e-7, 2.23e-5, 3.96e-8)` |
| `yalsdp` | timeout | 17.634 | converged | 9.145 | `(7.87e-5, 7.64e-5, 2.04e-5)` |
| `NH2-.1A1.STO6G.r14` | timeout | 175.132 | converged | 15.166 | `(5.20e-5, 5.50e-5, 2.72e-5)` |
| `H3O+_1-A1_STO-6GN10r16g1T2_5` | timeout | 113.968 | converged | 52.996 | `(9.34e-5, 6.02e-5, 3.49e-5)` |
| `neu1g` | converged | 177.670 | converged | 68.816 | `(1.22e-5, 7.97e-5, 4.42e-6)` |
| `e_moment_stable_17_0.5_2_2` | timeout | 125.944 | converged | 69.597 | `(9.82e-5, 8.81e-7, 4.73e-5)` |
| `AlH_1-Sigma+_STO-6GN14r20g1T2_5` | timeout | 86.683 | converged | 78.545 | `(8.43e-6, 6.43e-5, 1.54e-5)` |
| `NH3_1-A1_STO-6GN10r16g1T2_5` | timeout | 104.936 | converged | 96.130 | `(6.99e-5, 7.74e-5, 5.94e-5)` |
| `taha1c` | timeout | 538.901 | converged | 107.297 | `(7.38e-5, 1.66e-5, 9.36e-5)` |
| `reimer5` | numerical error | 1067.591 | converged | 217.936 | `(7.55e-5, 7.58e-8, 6.87e-6)` |
| `taha1b` | timeout | 574.239 | converged | 315.686 | `(4.73e-6, 7.82e-6, 9.99e-5)` |
| `neosfbr30e8` | no status | — | converged | 1967.577 | `(1.00e-4, 2.60e-6, 1.65e-5)` |

其中最显著的是 `theta123`：Julia 跑满约 989 秒仍未得到容差可行端点，
C++/CUDA 在约 1.86 秒收敛。这类“只在 C++ 成功”的实例不能用简单
时间比表示严格 speedup，但它们对 solver 的实际可用性更重要。

## 6. 结构分层结果

按独立 `p/d/g <= 1e-4` 统计：

| 结构分组 | 题数 | Julia 可行 | C++ 可行 | 共同可行中 C++ 更快 |
|---|---:|---:|---:|---:|
| 单 PSD cone | 49 | 10 | 21 | 10/10 |
| 2–10 cones | 7 | 0 | 1 | — |
| 11–100 cones | 16 | 4 | 11 | 4/4 |
| 超过 100 cones | 3 | 2 | 2 | 2/2 |
| 最大 cone 维数 ≤16 | 3 | 2 | 2 | 2/2 |
| 最大 cone 维数 17–256 | 19 | 8 | 13 | 8/8 |
| 最大 cone 维数 >256 | 53 | 6 | 20 | 6/6 |

“大量极小 cone”三题的完整情况：

| instance | cone 数 | 最大维数 | Julia | C++ | Cardal | C++/Julia | C++/Cardal |
|---|---:|---:|---:|---:|---:|---:|---:|
| `chs_5000` | 4998 | 10 | 6.968 s | 0.444 s | 1079.504 s | 15.70x | 2431.72x |
| `ros_2000` | 1999 | 6 | 14.093 s | 2.596 s | 670.370 s | 5.43x | 258.19x |
| `mater-6` | 4968 | 11 | timeout, 7231 s | timeout, 7231 s | 3615.675 s | — | — |

由此可见，cone 数量和维数是重要信号，但不是充分条件。C++ 的优势更像
“某些具有大量同维极小 block、投影可高效 batch 的问题”，而不是所有
多 cone 问题。2–10 cone 组仍然很弱；单 cone 的成功率从 Julia 的
10/49 提高到 C++ 的 21/49，但也远未覆盖全部问题。

## 7. C++/CUDA 与 Cardal v294

采用本报告主要的 `p/d/g <= 1e-4` 口径：

| outcome | 题数 |
|---|---:|
| C++ 与 Cardal 均可行 | 35 |
| 仅 Cardal 可行 | 35 |
| 仅 C++ 可行 | 0 |
| 两者均不可行 | 5 |

共同可行的 35 题中：

- C++ solver time 更快：25/35（71.4%）；
- C++ 端点生成 wall time 更快：22/35（62.9%）；
- 累计 solver time：Cardal 8870.63 秒，C++ 4359.43 秒，C++ 快 2.03x；
- 累计端点生成 wall：C++ 4443.13 秒，对 Cardal solver time 的比值为
  2.00x；
- 逐题 solver 加速中位数：2.45x。

最显著的 solver 加速如下：

| instance | C++ solver | Cardal solver | 加速 |
|---|---:|---:|---:|
| `chs_5000` | 0.444 s | 1079.504 s | 2431.72x |
| `theta123` | 1.859 s | 494.448 s | 266.01x |
| `ros_2000` | 2.596 s | 670.370 s | 258.19x |
| `neu3` | 0.368 s | 37.471 s | 101.94x |
| `theta102` | 1.660 s | 150.965 s | 90.92x |
| `1zc.1024` | 4.582 s | 279.706 s | 61.04x |
| `Bst_jcbpaf2` | 2.201 s | 114.303 s | 51.93x |
| `Bex2_1_5` | 1.793 s | 78.663 s | 43.88x |
| `fap09` | 7.983 s | 207.929 s | 26.05x |
| `rose15` | 115.298 s | 1602.442 s | 13.90x |

这一子集优势不能掩盖整体鲁棒性问题。完整 75 题上，C++ 有 40 个问题
跑满两倍预算；累计 C++ solver work 为 57,261.65 秒，Cardal 为
35,320.31 秒，即 C++ 整体使用约 1.62 倍 solver 时间。Cardal 可行率
为 70/75，C++ 为 35/75，而且 C++ 没有解决 Cardal 未解决的 5 题。

因此，合理用法是先根据结构或短时 probe 选择 PDSDP，否则回退到 Cardal，
而不是让 PDSDP 独占完整预算。

## 8. PSD transport 与容差可行性的解释

C++ 有 35 个 native-converged 端点通过独立 `p/d/g <= 1e-4`。其中 24
个同时通过 `1e-10` 的 transport-strict guard，另有 11 个因为原始完整
primal matrix 含很小的负特征值、验证器在构造 PSD factor 时发生裁剪而
被 transport-strict 拒绝。

这 11 个端点的裁剪前归一化 primal PSD DIMACS error 位于：

```text
2.61e-10 ... 4.09e-7
```

最大值仍比 `1e-4` 的实验容差小约 244 倍。因此，把它们计入
DIMACS-tolerance feasible 的工程统计是合理的；`1e-10` transport guard
是远严格于求解精度的“无实质后处理”审计要求，不应与 `1e-4` 可行性混为
一谈。

但当前 checkpoint 存的是裁剪前完整矩阵，而 `p/d/g` 是验证器完成 PSD
factor transport 后计算的。若用于正式发布或第三方复验，最佳做法不是仅
修改标签，而是：

1. 在求解器中直接保存 PSD factor；或
2. 显式投影后重新导出 `X`；
3. 用独立验证器重新计算并保存投影后解的所有指标。

报告因此保留双口径：`35 DIMACS-tolerance feasible` 与
`24 transport-strict feasible`。

## 9. 对“C++/CUDA 重写是否更快”的判断

答案是肯定的，而且证据来自三层：

1. **共同成功问题的逐题证据**：16/16 solver time 更快；
2. **相同迭代数证据**：`neu3`、`chs_5000`、`theta102` 在完全相同迭代
   次数下仍快 4.44–15.81x，直接说明每迭代实现效率提高；
3. **鲁棒性证据**：`NUMERICAL_ERROR` 从 11 降到 0，`p/d/g` 可行端点从
   16 增到 35。

端点 wall-time 加速尤其大，部分来自消除 Julia 进程/JIT 约几十秒的固定
开销；solver time 的 1.22–15.81x 提升则不能只用 JIT 解释。

严格而言，这是一组“完整实现 A 对完整实现 B”的实验，不是只改变语言、
保持每个 kernel 和每次数值操作完全一致的微基准。C++ 版本同时改变了内存
生命周期、sparse product 复用、projection batching、kernel fusion 和部分
数值执行路径。因此可确定的是**当前 C++/CUDA 产品实现优于当前 Julia
产品实现**；不能把每一部分提升都单独归因于 C++ 语言本身。

## 10. 局限与下一步建议

### 10.1 当前局限

- 只有 35/75 达到 `p/d/g <= 1e-4`，40 题超时；
- 没有新增解决 Cardal 未解决实例；
- 两轮运行不是在同一时刻、同一物理 GPU 上交替重复的严格 paired
  microbenchmark，系统负载可能带来小幅扰动；
- 两轮 manifest 记录的独立 `dimacs.py` SHA 不同（Julia 为
  `145281...`，C++ 为 `81b87f...`）。两个 wrapper 对 `p/d/g` 使用相同
  定义，但为了最高等级的 A/B 可比性，仍应使用同一份冻结 validator
  重新验证两组 checkpoint；
- C++ full campaign 中途从同步验证改为异步 CPU 验证，但 solver time
  仍由原生可执行文件内部记录；
- C++ `decoupled_status.json` 的最终快照仍显示 `running`，这是监控状态
  写回的小 bug。权威 `campaign_status.json` 为 `complete`，75/75 完成，
  PID 已退出，结果不受影响。

### 10.2 优先工作

1. **修复 PSD-preserving export**：直接输出 factor 或投影后的正式解，
   重新验证当前 11 个 transport-rejected 端点。
2. **做重复 paired benchmark**：选择共同成功的 16 题，在同一张 H100
   上分别 warm Julia 和 C++，每题至少重复 3 次，报告中位数与波动。
3. **建立 portfolio 路由**：优先识别 `chs`、`ros`、`theta`、部分 `neu`
   和 `1zc` 类结构；短时 probe 不收敛则切换 Cardal。
4. **分析 40 个超时问题**：尤其是 2–10 cone、超大 dense cone 和
   `mater-6`，区分 projection、SpMV、步长控制和 residual stagnation。
5. **参数化 campaign driver**：移除冻结脚本中的 workspace/Python 绝对
   路径，并在最终退出时同步更新所有状态文件。

## 11. 结果文件索引

Julia：

- [manifest](julia-original/benchmark-summary/manifest.json)
- [comparison CSV](julia-original/benchmark-summary/comparison.csv)
- [final progress](julia-original/benchmark-summary/progress.json)
- [campaign status](julia-original/benchmark-summary/campaign_status.json)
- [campaign log](julia-original/benchmark-summary/campaign.log)
- [parallelization amendment](julia-original/benchmark-summary/PARALLELIZATION_AMENDMENT.md)

C++/CUDA：

- [manifest](cpp-cuda/benchmark-summary/manifest.json)
- [comparison CSV](cpp-cuda/benchmark-summary/comparison.csv)
- [final progress](cpp-cuda/benchmark-summary/progress.json)
- [campaign status](cpp-cuda/benchmark-summary/campaign_status.json)
- [initial campaign log](cpp-cuda/benchmark-summary/campaign.log)
- [decoupled campaign log](cpp-cuda/benchmark-summary/campaign_decoupled.log)
