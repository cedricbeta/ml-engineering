# Reading Profiler Traces

A profiler trace answers the question "where does the time go?" with a timeline of what every CPU thread and every GPU stream was doing. Getting a trace is easy; the skill is in reading it - knowing which patterns mean "the GPU is starved", "communication isn't overlapped", or "this kernel is slow" - and then picking the right tool for the next level of detail.

This chapter covers the 4 tools you're most likely to use and how to read what they produce:

| Tool | Answers | Overhead | View with |
| :--- | :------ | :------- | :-------- |
| [`torch.profiler`](#torchprofiler) | which PyTorch ops and kernels run, for how long, called from where | moderate | [Perfetto](#perfetto) |
| [Nsight Systems](#nsight-systems) (`nsys`) | the whole system timeline: all processes, threads, CUDA API, kernels, NCCL, memory copies, OS calls, GPU hardware metrics | low | Nsight Systems GUI |
| [Nsight Compute](#nsight-compute) (`ncu`) | why one specific kernel is slow: compute vs memory bound, occupancy, cache hits | very high, per kernel | Nsight Compute GUI |
| [HTA](#holistic-trace-analysis) | summary statistics over many ranks' `torch.profiler` traces | none (offline) | Jupyter |

The workflow is top-down: start with the timeline (`torch.profiler` or `nsys`) to find where the time is lost, and only go to `ncu` once you know which kernel deserves the attention.

For the basics of `torch.profiler`'s tabular output see [Profilers](../../debug/pytorch.md#profilers), and for memory profiling see [Memory profiler tools](./README.md#memory-profiler-tools).

## Before you profile

- **Profile the steady state.** The first iterations are dominated by one-time costs - CUDA context creation, `torch.compile`, cuDNN/Triton autotuning, the caching allocator growing. Skip them.
- **Profile a few steps, not the whole run.** A trace of a few steps of a large model is already hundreds of MBs. Use the profiler's schedule or capture range (shown below).
- **Know your baseline number** - the step time or the tokens/sec without the profiler, so you can tell how much the profiler itself distorts things.
- **Name your code regions.** `nsys` shows NVTX ranges and `torch.profiler` shows `record_function` labels on the timeline (use both if you use both tools), which turns a sea of kernel names into "forward", "backward", "optimizer", "dataloader":

```python
import torch
with torch.cuda.nvtx.range("forward"):  # shows up in nsys
    loss = model(batch)
with torch.profiler.record_function("backward"):  # shows up in torch.profiler
    loss.backward()
```

## torch.profiler

To trace a few steady-state steps of a training loop:

```python
from torch.profiler import profile, schedule, ProfilerActivity

def trace_handler(prof):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    prof.export_chrome_trace(f"trace-rank{rank}-step{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=2, active=3, repeat=1),  # skip 5 steps, warm up 2, record 3
    on_trace_ready=trace_handler,
    record_shapes=True,  # the input shapes of each op
    with_stack=True,     # the python call stack of each op - more overhead and a bigger trace
) as prof:
    for batch in dataloader:
        train_step(batch)
        prof.step()  # tells the profiler a step has ended
```

In a multi-GPU job every rank writes its own trace. Usually it's enough to look at one or 2 ranks, plus the rank that is suspected of being slow.

The output is a JSON file in the Chrome trace format, which you open in [Perfetto](#perfetto).

## Perfetto

[Perfetto UI](https://ui.perfetto.dev) is a trace viewer that runs in the browser - drag the trace file onto the page. The trace is processed locally and is not uploaded anywhere. It replaces the older `chrome://tracing` viewer.

Navigation:
- `W`/`S` zoom in/out around the mouse pointer, `A`/`D` pan left/right
- click a slice to see its details (duration, arguments, shapes, python stack) in the bottom panel
- drag over a region to get the aggregated time per slice name within it
- the search box finds slices by name (e.g., a kernel name or `nccl`)

The browser limits the memory to about 2GB, and a trace takes a few times its file size in memory, so big traces fail to open. Then use the native trace processor, which the UI connects to automatically:

```bash
curl -LO https://get.perfetto.dev/trace_processor
chmod +x trace_processor
./trace_processor server http trace.json   # older versions: ./trace_processor --httpd trace.json
# now open https://ui.perfetto.dev - it offers to use the local trace processor
```

### What you see in a torch.profiler trace

- **CPU tracks** - the Python main thread with the nested PyTorch ops (`aten::mm`, `aten::copy_`, ...) and the CUDA runtime calls they make (`cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSynchronize`). The backward pass usually runs on a separate autograd thread.
- **GPU tracks** - one per CUDA stream: the kernels and memory copies as they actually executed on the GPU. Compute typically runs on one stream, and NCCL communication on its own streams.
- **Flow arrows** - selecting a `cudaLaunchKernel` shows an arrow to the kernel it launched. The horizontal distance between them is how far ahead of the GPU the CPU is.

Remember that CUDA is asynchronous: the CPU only enqueues the kernels, and the GPU executes them later. So the time an op takes on the CPU track is the time to *launch* it, not to run it.

## How to read a timeline

These patterns apply equally to `torch.profiler` traces in Perfetto and to `nsys` timelines.

### 1. Gaps on the GPU stream

The single most important thing to look at: is the compute stream busy all the time? Gaps mean the GPU is idle, waiting for the CPU. Look at what the CPU threads do during the gap:

- `cudaStreamSynchronize`, `cudaDeviceSynchronize` or `aten::item` / `aten::_local_scalar_dense` - something forces the CPU to wait for the GPU, after which the GPU waits for the CPU to launch more work. Typical culprits are `.item()`, `.cpu()`, `print(tensor)`, and Python `if` on a tensor value. Move them out of the hot path, or do them every N steps.
- the data loader - `next(iter)` or your collate function in the main thread: the input pipeline isn't keeping up, see [DataLoader](./README.md#dataloader).
- many small Python-level ops - e.g., an optimizer step looping over thousands of parameters, or a model with lots of tiny layers.

### 2. Many tiny kernels

Kernels of a few microseconds each, separated by gaps of similar size, mean the work is *launch-bound*: the CPU can't launch kernels as fast as the GPU finishes them. Typical in inference with small batches and in models with many small ops. Remedies: larger batches, fused kernels, `torch.compile` (fewer, bigger kernels), CUDA graphs (launch a whole sequence of kernels with a single call).

The healthy opposite: the CPU runs far ahead of the GPU (the flow arrows are long) and the GPU stream is packed - the job is GPU-bound, and further gains have to come from faster kernels.

### 3. Communication

NCCL kernels (names starting with `nccl`, e.g., `ncclDevKernel_AllReduce...`) run on their own streams. What matters is whether they overlap with compute:

- **overlapped** - the compute stream keeps running while the NCCL kernel runs on the other stream. Good.
- **exposed** - the compute stream is idle while the NCCL kernel runs. This is time lost to communication. In DDP/FSDP this can be improved with bucket sizes, prefetching settings or a different parallelism strategy, see [Model parallelism](../model-parallelism/).

A very long NCCL kernel doesn't necessarily mean a slow network: a collective can only finish when all ranks join it, so its duration includes the time waiting for the slowest rank. If one rank's collectives are consistently short and all the others are long, the short one is the straggler - everyone waits for it, and its trace shows what it was busy with. To measure the network itself, use the [network benchmarks](../../network/benchmarks/).

### 4. Memory operations

- `Memcpy HtoD` on the compute stream right before the kernels that need the data - the copy isn't overlapped with compute. `Pageable -> Device` in its name means the source memory isn't pinned, which is slower and synchronous, see [Pinned memory and non-blocking device copy](./README.md#pinned-memory-and-non-blocking-device-copy).
- `cudaMalloc` / `cudaFree` in the steady state - normally the caching allocator reuses memory and these are rare after the first steps. If you see them every step, something is fighting the allocator (e.g., calling `torch.cuda.empty_cache()`, or memory fragmentation near the memory limit).

### 5. The GPU is busy, but slow

When the GPU stream has no gaps, look at which kernels take the time (in Perfetto: select a whole step and sort by the total duration):

- Are they the kernels you expect? E.g., matmuls should run in the dtype you train with. Check the input dtypes of the matmul ops (with `record_shapes=True` they are in the op's details), or the input types in the kernel name - but note that bf16 GEMM kernel names usually also contain `f32` for the accumulator (e.g., `bf16bf16_bf16f32`), so it's the input types (`bf16bf16` vs `f32f32`) that matter. Attention should be a flash attention kernel, not a chain of `bmm` + `softmax` + `bmm`.
- Are there unexpected kernels taking a big share - e.g., lots of `elementwise` or `copy` kernels (non-contiguous tensors, dtype conversions)?

Once you find a legitimate kernel that is slower than it should be, that's the time for [Nsight Compute](#nsight-compute).

### 6. Quantify before and after

Measure the step time and the GPU busy time (the sum of kernel durations on the compute stream over the wall time of a step) in the trace before and after a change. A change that looks better in the trace but doesn't improve the step time without the profiler isn't an improvement.

## Nsight Systems

[Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) traces the whole system, not just PyTorch: every process (including the data loader workers and all the ranks on the node), the CUDA API, the kernels, NCCL, cuBLAS/cuDNN, the OS calls, and optionally GPU hardware metrics. Its overhead is lower than `torch.profiler`'s, and it doesn't need any changes to the code.

The NGC PyTorch images include the `nsys` and `ncu` command line tools. Otherwise install the Nsight Systems CLI from NVIDIA's package repository. The GUI for viewing the reports runs on Linux, Windows and macOS - you profile on the GPU node and look at the report on your laptop.

### Profiling a few steps

To avoid huge reports, only record a few steady-state steps. Mark the range in the code:

```python
for step, batch in enumerate(dataloader):
    if step == 10:
        torch.cuda.profiler.start()
    with torch.cuda.nvtx.range(f"step {step}"):
        train_step(batch)
    if step == 13:
        torch.cuda.profiler.stop()
```

and tell `nsys` to record only within this range:

```bash
nsys profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-memory-usage=true \
    -o report-%h --force-overwrite=true \
    python train.py
```

(`%h` in the output name is replaced by the hostname, so each node of a multi-node job writes its own report.)

`nsys` follows child processes, so you can put it in front of `torchrun` and get all the ranks of the node in a single report. Alternatively, without changing the code, use `--delay=SECONDS --duration=SECONDS` to record a time window.

Add `--gpu-metrics-devices=all` to also sample GPU hardware metrics (SM activity, tensor core activity, memory bandwidth, NVLink and PCIe throughput) - this, like Nsight Compute, needs access to the GPU performance counters, see [Permissions](#permissions).

### Reading an nsys report

Open the `.nsys-rep` file in the Nsight Systems GUI. The timeline has a row per process and thread with their CUDA API calls, NVTX ranges and OS calls, and a "CUDA HW" section per GPU with the kernels and memory copies per stream. All the [patterns above](#how-to-read-a-timeline) apply. What `nsys` adds:

- **NVTX ranges** give the structure - your `step N`, `forward`, `backward` labels, and NCCL and other libraries also emit their own ranges.
- **GPU Metrics rows** show the hardware utilization over time. Low SM activity during a stretch of kernels means the kernels themselves don't keep the GPU busy (e.g., too small to fill all the SMs).
- **All the processes** - e.g., whether the data loader worker processes are busy or idle, or whether some other process on the node competes for the GPU.

For a quick text summary without the GUI:

```bash
nsys stats --report cuda_gpu_kern_sum report.nsys-rep   # the kernels sorted by total time
nsys stats --report nvtx_sum report.nsys-rep            # time per NVTX range
nsys stats --report cuda_api_sum report.nsys-rep        # time per CUDA API call (look for synchronizations)
```

## Nsight Compute

[Nsight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`) profiles individual kernels in depth: it replays each selected kernel many times, collecting hundreds of hardware counters. This makes the profiled kernels 10-100x slower, so always select only a few kernels:

```bash
# the 1st matmul kernel after skipping the first 50 (warmup) launches of matching kernels
ncu --kernel-name regex:gemm --launch-skip 50 --launch-count 1 --set full -o gemm-report python train.py
```

Open the `.ncu-rep` file in the Nsight Compute GUI. Where to look:

- **GPU Speed Of Light** - the kernel's compute (SM) throughput and memory throughput as a percentage of the hardware's peak. If one of them is near 100%, the kernel is bound by it and there is little to gain without changing the algorithm. If both are low, the kernel doesn't keep the GPU busy - latency-bound, too few blocks to fill the SMs, or stalls - and the next sections tell why.
- **Roofline** - the same story as a chart: where the kernel sits relative to the compute and memory bandwidth limits.
- **Launch Statistics and Occupancy** - the grid size vs the number of SMs (see [Tile and wave quantization](./README.md#tile-and-wave-quantization)), and how many warps can be active per SM.
- **Memory Workload Analysis** - cache hit rates and the traffic between the memory levels.

Nsight Compute is the tool for kernel developers. For most ML engineers its practical use is to confirm that a slow kernel is genuinely limited by the hardware, or to give the kernel's authors the evidence that it isn't.

### Permissions

`ncu` and `nsys --gpu-metrics-devices` need access to the GPU performance counters, which by default the driver allows only to admins. Without it you get `ERR_NVGPUCTRPERM`. Either the node's admin sets the NVIDIA kernel module parameter `NVreg_RestrictProfilingToAdminUsers=0` (see [Module parameters](../../orchestration/containers/drivers.md#module-parameters)), or the container runs with the `SYS_ADMIN` capability (`--cap-add=SYS_ADMIN` with Docker, `securityContext.capabilities.add` on k8s) - the latter is often not allowed on shared clusters.

On k8s, run the profiler inside the pod, then copy the report out with `kubectl cp POD:/workspace/report-node1.nsys-rep .` and open it on your machine.

## Holistic Trace Analysis

When you have `torch.profiler` traces from many ranks, looking at each one in Perfetto doesn't scale. [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) (HTA) loads all the traces of a run and computes summaries per rank:

```python
# pip install HolisticTraceAnalysis
from hta.trace_analysis import TraceAnalysis

analyzer = TraceAnalysis(trace_dir="./traces")
analyzer.get_temporal_breakdown()     # per rank: % of time in compute, communication and idle
analyzer.get_comm_comp_overlap()      # per rank: how much of the communication overlaps with compute
analyzer.get_idle_time_breakdown()    # why the GPU was idle
analyzer.get_gpu_kernel_breakdown()   # which kernels take the time
```

A rank whose numbers stand out from the others is where to look in Perfetto.

## AMD

The same concepts apply on AMD GPUs: `torch.profiler` works the same and produces the same traces for Perfetto, and the ROCm equivalents of Nsight are [rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/) for tracing and kernel counters, and [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) (formerly Omnitrace), whose traces are viewed in Perfetto as well.
