# 读懂 Profiler Trace：中文要点

> 英文原文：[Reading Profiler Traces](./profiling.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
profiler trace 用一条时间线展示每个 CPU 线程和每个 GPU stream 在做什么，回答“时间都花到哪里去了”。拿到 trace 很容易，难的是读懂它：知道哪些模式意味着“GPU 在挨饿”“通信没有被 overlap”或“这个 kernel 慢”，然后选对下一层深入分析的工具。这一章介绍最常用的 4 个工具，以及如何阅读它们的输出。

## 核心概念
- **`torch.profiler`**：看哪些 PyTorch op 和 kernel 在运行、各用多长时间、从哪里调用；开销中等；输出 Chrome trace 格式的 JSON，用 Perfetto 查看。
- **Nsight Systems (`nsys`)**：整个系统的时间线，包括所有进程、线程、CUDA API、kernel、NCCL、内存拷贝、OS 调用和 GPU 硬件指标；开销比 `torch.profiler` 低，不需要改代码。
- **Nsight Compute (`ncu`)**：分析某个具体 kernel 为什么慢：compute bound 还是 memory bound、occupancy、cache 命中率；开销很高，要按 kernel 挑选。
- **HTA (Holistic Trace Analysis)**：离线对多个 rank 的 `torch.profiler` trace 计算汇总统计，在 Jupyter 里使用。
- **Perfetto**：在浏览器里运行的 trace 查看器，trace 在本地处理、不会上传，取代了旧的 `chrome://tracing`。
- **NVTX range / `record_function`**：给代码区域命名，让时间线上出现 "forward"、"backward"、"optimizer"、"dataloader" 这样的标签，而不是一大片 kernel 名。`nsys` 显示 NVTX range，`torch.profiler` 显示 `record_function` 标签，两个工具都用就两种都加。
- **launch-bound**：CPU 发射 kernel 的速度赶不上 GPU 执行完的速度。
- **exposed communication**：NCCL kernel 运行期间 compute stream 处于空闲，这段时间就损失在通信上了。

## 关键要点
- **自上而下**：先看时间线（`torch.profiler` 或 `nsys`）找出时间丢在哪里，确定了哪个 kernel 值得关注之后才用 `ncu`。profile 之前：只看稳态，前几个 iteration 被 CUDA context 创建、`torch.compile`、cuDNN/Triton autotuning、caching allocator 扩张等一次性开销主导；只录几个 step，大模型几个 step 的 trace 就有几百 MB；先知道不开 profiler 时的基准数字（step time 或 tokens/sec），才能判断 profiler 本身带来多少失真；给代码区域命名。[详见](./profiling.md#before-you-profile)
- **`torch.profiler`**：用 `schedule` 跳过前几个 step、预热、再录几个 step；`record_shapes=True` 记录每个 op 的输入形状，`with_stack=True` 记录 python 调用栈（开销更大、trace 更大）。多 GPU job 中每个 rank 各写一个 trace，通常看 1-2 个 rank，再加上怀疑慢的那个 rank 就够了。[详见](./profiling.md#torchprofiler)
- **Perfetto 操作**：把 trace 文件拖进 [Perfetto UI](https://ui.perfetto.dev)；`W`/`S` 以鼠标为中心缩放，`A`/`D` 左右平移；点击 slice 在底部面板看详情；框选一段区域得到按 slice 名字聚合的时间；搜索框按名字查找（如 kernel 名或 `nccl`）。浏览器内存上限约 2GB，而 trace 占用的内存是文件大小的几倍，大 trace 打不开时用本地的 native trace processor，UI 会自动连接它。[详见](./profiling.md#perfetto)
- **trace 里有什么**：CPU track 是 Python 主线程及嵌套的 PyTorch op（`aten::mm`、`aten::copy_` 等）和它们发出的 CUDA runtime 调用（`cudaLaunchKernel`、`cudaMemcpyAsync`、`cudaStreamSynchronize`），backward 通常在单独的 autograd 线程上；GPU track 每个 CUDA stream 一条，compute 通常在一个 stream 上，NCCL 通信在它自己的 stream 上；flow arrow 从 `cudaLaunchKernel` 指向它发射的 kernel，水平距离就是 CPU 领先 GPU 多少。CUDA 是异步的，CPU track 上一个 op 的时长是*发射*它的时间，不是运行它的时间。[详见](./profiling.md#what-you-see-in-a-torchprofiler-trace)
- **最重要的信号是 GPU stream 上的空隙**：空隙意味着 GPU 闲着在等 CPU，看空隙期间 CPU 在做什么：同步操作（`cudaStreamSynchronize`、`cudaDeviceSynchronize`、`aten::item` / `aten::_local_scalar_dense`，典型元凶是 `.item()`、`.cpu()`、`print(tensor)` 和对 tensor 值做 Python `if`，要移出热路径或每 N 步做一次）；data loader 跟不上；或大量细小的 Python 级 op（如遍历几千个参数的 optimizer step）。[详见](./profiling.md#1-gaps-on-the-gpu-stream)
- **小 kernel 与通信**：每个几微秒的 kernel 之间隔着差不多大的空隙，说明是 launch-bound，常见于小 batch 推理和有很多小 op 的模型，对策是更大的 batch、融合 kernel、`torch.compile`（更少更大的 kernel）、CUDA graphs（一次调用发射一整串 kernel）；健康的反面是 CPU 远远领先、GPU stream 排满，即 GPU-bound，进一步提速只能靠更快的 kernel。NCCL kernel（名字以 `nccl` 开头）关键看是否和 compute overlap；很长的 NCCL kernel 不一定说明网络慢，因为 collective 要等所有 rank 都加入，如果某个 rank 的 collective 一直很短而其他 rank 都很长，那个短的就是大家在等的 straggler。[详见](./profiling.md#3-communication)
- **内存操作与“GPU 忙但慢”**：compute stream 上紧挨着要用数据的 kernel 之前出现 `Memcpy HtoD`，说明拷贝没有和计算 overlap，名字里带 `Pageable -> Device` 说明源内存没有 pin，更慢而且是同步的；稳态下每个 step 都有 `cudaMalloc` / `cudaFree`，说明有东西在和 caching allocator 作对（如调用 `torch.cuda.empty_cache()`，或接近显存上限时的碎片）。GPU 没有空隙但慢时，选中整个 step 按总时长排序：是不是预期的 kernel（matmul 应该以训练用的 dtype 运行：看 matmul op 的输入 dtype（开了 `record_shapes=True` 就在 op 详情里），或 kernel 名里的输入类型，注意 bf16 GEMM kernel 名通常也带表示累加器的 `f32`（如 `bf16bf16_bf16f32`），所以要看的是输入类型是 `bf16bf16` 还是 `f32f32`；attention 应该是 flash attention kernel，而不是 `bmm` + `softmax` + `bmm`），有没有占比很大的意外 kernel（如大量 `elementwise` 或 `copy` kernel，来自非连续 tensor 或 dtype 转换）。[详见](./profiling.md#5-the-gpu-is-busy-but-slow)
- **改动前后要量化**：在 trace 中测量改动前后的 step time 和 GPU busy time（在一个 step 的 wall time 内，compute stream 上所有 kernel 时长的总和）。trace 里看起来更好、但不开 profiler 时 step time 没有改善的改动，不算改进。[详见](./profiling.md#6-quantify-before-and-after)
- **nsys**：在代码里用 `torch.cuda.profiler.start()` / `stop()` 标出范围，配合 `--capture-range=cudaProfilerApi` 只录几个稳态 step；或者不改代码，用 `--delay=SECONDS --duration=SECONDS` 录一个时间窗口。`-o report-%h` 中的 `%h` 会替换成主机名，多节点 job 每个 node 一份报告；`nsys` 会跟随子进程，放在 `torchrun` 前面就能在一份报告里拿到本 node 的所有 rank；`--gpu-metrics-devices=all` 额外采样 SM、tensor core、显存带宽、NVLink 和 PCIe 等硬件指标。NGC PyTorch 镜像自带 `nsys` 和 `ncu` 命令行工具；GUI 可在 Linux、Windows 和 macOS 上运行，在 GPU node 上 profile，在笔记本上看报告。[详见](./profiling.md#nsight-systems)
- **ncu、HTA 与 AMD**：`ncu` 把每个选中的 kernel 重放很多次、收集几百个硬件计数器，被 profile 的 kernel 会慢 10-100 倍，所以只选少数几个；重点看 GPU Speed Of Light（compute 或 memory 吞吐有一个接近 100%，说明受它限制，不改算法收益不大；两个都低说明 kernel 没把 GPU 喂饱）、Roofline、Launch Statistics/Occupancy、Memory Workload Analysis。对大多数 ML 工程师来说，它的实际用途是确认慢 kernel 是否真的受硬件限制，或者给 kernel 作者提供证据。HTA 汇总一次运行的所有 rank，数字明显异常的 rank 就是要在 Perfetto 里细看的。AMD 上 `torch.profiler` 用法相同，Nsight 对应的是 rocprofv3 和 ROCm Systems Profiler（前身 Omnitrace）。[详见](./profiling.md#nsight-compute)

## 常用命令 / 配置
```python
# 给代码区域命名
with torch.cuda.nvtx.range("forward"):  # 在 nsys 里显示
    loss = model(batch)
with torch.profiler.record_function("backward"):  # 在 torch.profiler 里显示
    loss.backward()
```

```bash
# nsys：只在 torch.cuda.profiler.start()/stop() 之间录制，每个 node 一份报告
nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-memory-usage=true \
    -o report-%h --force-overwrite=true \
    python train.py
nsys stats --report cuda_gpu_kern_sum report.nsys-rep   # 不开 GUI，按总时长列出 kernel
nsys stats --report cuda_api_sum report.nsys-rep        # 每种 CUDA API 调用的时间（找同步）
# ncu：跳过前 50 次匹配的（预热）发射，只分析第 1 个 matmul kernel
ncu --kernel-name regex:gemm --launch-skip 50 --launch-count 1 --set full -o gemm-report python train.py
./trace_processor server http trace.json   # Perfetto 打不开大 trace 时用本地 trace processor；旧版本用 ./trace_processor --httpd trace.json
kubectl cp POD:/workspace/report-node1.nsys-rep .   # k8s 上在 pod 里 profile，再把报告拷出来
```

## 常见坑
- GPU stream 上到处是空隙 -> `.item()`、`.cpu()`、`print(tensor)` 或对 tensor 值做 `if` 等操作强制同步 -> 移出热路径，或每 N 步做一次。
- 把 CPU track 上 op 的时长当成运行时间 -> CUDA 是异步的，那只是发射时间 -> 看 GPU track 上 kernel 的实际执行。
- 大 trace 在 Perfetto 里打不开 -> 浏览器内存上限约 2GB，trace 占用的内存是文件大小的几倍 -> 用本地 native trace processor。
- 看到很长的 NCCL kernel 就认为网络慢 -> 它的时长包含了等待最慢 rank 的时间 -> 找出 collective 一直很短的那个 straggler rank；网络本身用 [network benchmarks](../../network/benchmarks/) 测。
- `ncu` 或 `nsys --gpu-metrics-devices` 报 `ERR_NVGPUCTRPERM` -> driver 默认只允许 admin 访问 GPU 性能计数器 -> 由 node 管理员设置内核模块参数 `NVreg_RestrictProfilingToAdminUsers=0`，或容器加 `SYS_ADMIN` capability（共享集群上常常不允许）。
- 用 `ncu` 后程序慢得离谱 -> 每个选中的 kernel 都被重放很多次，慢 10-100 倍 -> 用 `--kernel-name`、`--launch-skip`、`--launch-count` 只挑少数几个 kernel。

## 相关章节
- [Profilers](../../debug/pytorch.md#profilers)（`torch.profiler` 表格输出的基础）、[Memory profiler tools](./README.md#memory-profiler-tools)
- [DataLoader](./README.md#dataloader)、[Pinned memory and non-blocking device copy](./README.md#pinned-memory-and-non-blocking-device-copy)、[Tile and wave quantization](./README.md#tile-and-wave-quantization)
- [Model parallelism](../model-parallelism/)、[network benchmarks](../../network/benchmarks/)
- [Module parameters](../../orchestration/containers/drivers.md#module-parameters)（性能计数器权限）
- 外部：[Perfetto UI](https://ui.perfetto.dev)、[Nsight Systems](https://developer.nvidia.com/nsight-systems)、[Nsight Compute](https://developer.nvidia.com/nsight-compute)、[Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis)、[rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/)、[ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/)
