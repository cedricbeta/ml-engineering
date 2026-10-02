# Snapshots: Restoring Instead of Starting

An inference replica can't answer a single request until it's fully initialized: the CUDA context created, the weights loaded onto the GPU, the kernels autotuned, `torch.compile` done and the CUDA graphs captured. For a large model this [cold start](./inference.md#cold-start) takes minutes, and it's paid again by every new replica, every scale-up from zero and every restart.

The idea behind snapshots is to pay that cost once: save the initialized state, and have every later replica restore it instead of re-creating it. Approaches differ in how much of the state they save, from the weights only, up to the whole running process including its GPU memory - the latter being what NVIDIA's Dynamo Snapshot does on k8s.

## What "snapshot" can mean

The word is used for several unrelated things in ML infrastructure, so first a disambiguation:

| Term | What is saved | What for | See |
| :--- | :------------ | :------- | :-- |
| training checkpoint | model weights, optimizer states, RNG, data position | resuming training | [Checkpoints](../../training/checkpoints/), [Fault Tolerance on Kubernetes](./fault-tolerance.md) |
| PyTorch memory snapshot | the history of GPU memory allocations | debugging memory usage | [PyTorch memory profiler](../../debug/pytorch.md#pytorch-memory-profiler) |
| k8s `VolumeSnapshot` | the content of a PVC at a point in time | backing up or cloning volumes, e.g., a pre-populated model cache | [Volume Snapshots](https://kubernetes.io/docs/concepts/storage/volume-snapshots/) |
| process checkpoint/restore | a whole running process, its CPU and GPU memory | starting replicas in seconds, migrating processes | this chapter |

Also note the name clash: NVIDIA Dynamo, the inference serving framework whose Dynamo Snapshot is covered below, has nothing to do with TorchDynamo (`torch._dynamo`), the frontend of `torch.compile`.

## The ladder of startup optimizations

Each step of the ladder saves more of the initialized state, skips more of the cold start, and comes with more constraints:

| # | What's saved | What it skips | Tools | Constraints |
| :-: | :--------- | :------------ | :---- | :---------- |
| 1 | the weights, in a fast place and format | slow downloads and reads | local NVMe, fast shared storage, [Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer), pre-sharded weights, image volumes | the weights still need to be loaded, and everything else initialized |
| 2 | compilation artifacts | recompiling | persistent `torch.compile`, Triton and vLLM caches | must match the code, the library versions and the GPU |
| 3 | the whole process, kept alive with the GPU memory released | everything, but only on the same node | vLLM sleep mode, SGLang memory saver | the process must keep running; vLLM's sleep level 1 holds the weights in CPU RAM |
| 4 | the whole process, including its GPU memory, as a file | everything, on any compatible node | `cuda-checkpoint` + CRIU, Dynamo Snapshot, Modal GPU memory snapshots | must match the image, the driver and the GPU type; a privileged agent; Dynamo Snapshot supports only single-GPU workloads for now |

Steps 1 and 2 are cheap, robust and worth doing in any case - see [Data and Model Loading](./storage.md#model-weights) and [Caches that make restarts faster](./storage.md#caches-that-make-restarts-faster). Steps 3 and 4 are covered below.

Before choosing, [measure where your cold start goes](./inference.md#cold-start). No snapshot helps with pulling the container image or getting a node from the autoscaler - if those dominate, fix them first.

### Compile caches as an artifact

Beyond pointing the cache directories to persistent storage, PyTorch can export the `torch.compile` cache artifacts as bytes, so you can ship them alongside the model, e.g., produce them in a build step and load them at startup:

```python
import torch

# after running the compiled model once:
result = torch.compiler.save_cache_artifacts()  # None if nothing was compiled
if result is not None:
    artifacts, cache_info = result
    open("compile-cache.bin", "wb").write(artifacts)

# in a new process, before running the compiled model:
torch.compiler.load_cache_artifacts(open("compile-cache.bin", "rb").read())
```

The model still has to be compiled - but from the cache, without generating and autotuning the kernels again.

## Keeping the process: vLLM sleep mode

If the goal is to free the GPUs between uses (e.g., several models sharing the same GPUs, or an RL loop alternating between training and generation), you don't need to save the process at all - just keep it running and release its GPU memory. vLLM's [sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html) does that:

- level 1 - the weights are offloaded to CPU RAM and the KV cache is discarded. Waking up copies the weights back - much faster than a cold start.
- level 2 - both the weights and the KV cache are discarded, for when the weights will be replaced anyway (e.g., an RL weight update) or there isn't enough CPU RAM.

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --port 8000
curl -X POST 'http://localhost:8000/sleep?level=1'
curl -X POST 'http://localhost:8000/wake_up'
```

`VLLM_SERVER_DEV_MODE=1` enables development endpoints, which must not be exposed to users. The limitation is inherent: the process lives on, so a sleeping model still holds its node and CPU memory, and it doesn't help to start replicas on other nodes. SGLang has the equivalent `release_memory_occupation` / `resume_memory_occupation`.

## Process checkpoint/restore

To save a running process to a file and later recreate it - possibly on another machine - 2 tools work together:

- **[CRIU](https://criu.org/Main_Page)** (Checkpoint/Restore In Userspace) saves and restores Linux process trees: their memory, threads, open files, sockets and pipes - everything the Linux kernel knows about. It knows nothing about GPUs.
- **[cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)** handles the GPU side of a process. Suspending a process's CUDA state locks the CUDA APIs, waits for the already submitted work to finish, copies the device memory to host memory and releases the GPU. After that the process has no GPU resources left, and CRIU can save it like any other process. Resuming re-acquires the GPU, copies the memory back to the original virtual addresses and restores the streams and contexts.

The flow on a single machine (as root):

```bash
cuda-checkpoint --toggle --pid $PID                                  # suspend: GPU memory -> host memory
criu dump --shell-job --images-dir ckpt --tree $PID                  # save the process to disk (it exits)
criu restore --shell-job --restore-detached --images-dir ckpt        # recreate the process
cuda-checkpoint --toggle --pid $PID                                  # resume: host memory -> GPU
```

The capabilities live in the NVIDIA driver, and `cuda-checkpoint` just exposes them, so what's possible depends on the driver version: 550 introduced it, 570 added the integration with CRIU 4.0+ and a CUDA driver API (`cuCheckpoint*`) with the same functionality, 580 added GPU migration (restoring onto a different GPU - of the same type), 595 Arm support (Dynamo Snapshot itself doesn't support Arm yet), and 610 support for `cuIpcGetMemHandle`-based CUDA IPC. Some things can't be checkpointed - notably UVM (managed) memory and IPC memory created with `cuMemExportToShareableHandle()` - and the checkpoint fails if the process has them.

## Dynamo Snapshot on Kubernetes

Running CRIU against the processes inside pods, storing the results, and restoring them into new pods on other nodes needs orchestration. The kubelet has a [checkpoint API](https://kubernetes.io/docs/reference/node/kubelet-checkpoint-api/) (beta since k8s 1.30), but it only creates a checkpoint archive on the node - k8s has no native restore.

[Dynamo Snapshot](https://github.com/ai-dynamo/snapshot), announced by NVIDIA in 2026, fills this gap for GPU inference pods: it checkpoints a fully initialized pod - its process with its CPU and GPU memory - and restores it into new pods on any compatible node. It's integrated into [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) ([docs](https://docs.nvidia.com/dynamo/v1.2.1/kubernetes-deployment/advanced-platform/snapshot)), and is also available as a standalone project for use with your own serving stack. As of 2026-10 the standalone version is 0.1.0, and its authors say the APIs may still change, so it's not yet recommended for production-critical workloads.

### The components

- a control-plane operator and a privileged node agent (a DaemonSet), installed with a single Helm chart. The agent does the actual CRIU and `cuda-checkpoint` work.
- a `ReadWriteMany` volume where the checkpoint artifacts are stored.
- the resources you work with:

| Resource | Role |
| :------- | :--- |
| `PodSnapshot` | requests a checkpoint of a running pod, and is what restores refer to |
| `PodSnapshotContent` | the record of the stored artifact, managed by the operator |
| `SnapshotJob` | starts a pod from a template, checkpoints it once ready and removes it - for pipelines |
| the `nvidia.com/restore-from` pod annotation | makes a new pod restore from the named `PodSnapshot` instead of starting from scratch |

Checkpointing a running replica:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: PodSnapshot
metadata:
  name: vllm-snapshot
spec:
  source:
    podRef:
      name: vllm-source-<pod-id>
      containers:
        - main
```

```bash
kubectl apply -f vllm-snapshot.yaml
kubectl wait --for=condition=Ready podsnapshot/vllm-snapshot --timeout=30m
```

Restoring: a Deployment with the same pod spec, the `nvidia.com/restore-from: vllm-snapshot` annotation, and the container command replaced with an inert `sleep infinity` - the agent restores the checkpointed process into that container. The project's guides have the complete, ready to use manifests for vLLM, SGLang and TensorRT-LLM.

### The workload has to cooperate

A process can't be checkpointed at an arbitrary moment - e.g., in the middle of a generation, or before it has warmed up. So the inference server must follow a [workload contract](https://github.com/ai-dynamo/snapshot/blob/main/docs/reference/workload-contract.md), coordinating with the agent through files in a shared directory:

1. **Capture**: start the engine, run at least one real generation (so the lazy initialization, autotuning and CUDA graph capture are part of the snapshot), stop the in-flight work, release the GPU memory that doesn't need saving, and only then write the `ready-for-snapshot` file. The agent checkpoints the pod once it reports ready.
2. **Restore**: the new pod's entrypoint must stay idle instead of initializing (otherwise it would load a second copy of the model next to the restored one). The agent restores the process into the container, remaps its pod IP (via CRIU's inet-remap plugin) and the saved GPU UUID onto this pod's IP and this node's GPU, and signals `restore-complete`. The restored process then brings the GPU memory back, resumes generation, checks its health, and signals that it's serving.

The engines already have the calls needed for this - for vLLM `pause_generation()`, `sleep()`, `wake_up()` and `resume_generation()`, for SGLang `pause_generation()`, `release_memory_occupation()`, `resume_memory_occupation()` and `continue_generation()`.

The memory release step matters a lot: vLLM pre-allocates most of the GPU's memory for the KV cache, and checkpointing that empty cache would be pure waste. NVIDIA reports that releasing it shrank the checkpoint of Qwen3-0.6B on a B200 from about 190GiB to about 6GiB.

### How fast is it

From the project's [benchmarks](https://github.com/ai-dynamo/snapshot/blob/main/docs/development/benchmarks.md) (single B200, vLLM 0.20, driver 595, checkpoints on a VAST NFS volume, image pull and container start excluded from both columns):

| Model | Weights | Cold start | Restore |
| :---- | ------: | ---------: | ------: |
| Qwen3 0.6B | 1.5 GB | 52.4s | 3.5s |
| Qwen3 8B | 16.4 GB | 58.3s | 8.1s |
| GPT-OSS 120B | 65.3 GB | 85.3s | 31.1s |
| Qwen3 32B | 65.5 GB | 79.3s | 19.4s |
| Llama 3.3 70B FP8 | 72.7 GB | 102.1s | 22.4s |
| Qwen2.5 72B | 145.4 GB | 97.8s | 40.9s |

What determines the restore time:
- **the size of the checkpoint**, not the number of parameters - e.g., the 70B FP8 model restores in about half the time of a bf16 model of similar parameter count. The checkpoint holds more than the weights (the CUDA context, compiled kernels, workspace buffers, the process's CPU memory), which is why 2 models with the same weight size, GPT-OSS 120B and Qwen3 32B, restore in 31s and 19s.
- **the storage throughput** - the CRIU part, reading the process image from the shared volume, took 49-67% of the total in these runs. A slower storage moves every number up.
- **the PCIe bandwidth** - for copying the GPU memory back onto the GPU.

A report from [Photoroom](https://www.photoroom.com/inside-photoroom/how-we-cut-gpu-cold-starts-from-minutes-to-seconds-with-memory-checkpointing), who run Dynamo Snapshot's node agent with their own deployment tooling, shows the other side: on nodes that already had the image, startup went from ~220s to 35-45s, but on fresh nodes, where the image had to be pulled first, only from ~430s to ~195s.

### Requirements and limitations (as of 2026-10)

- containerd or CRI-O, x86_64 nodes, NVIDIA GPU Operator 26.3+ with driver 580+, MIG disabled, no vGPU.
- single-GPU workloads only - multi-GPU and multi-node are on the roadmap (they need hooks to quiesce and resume NCCL and other communication libraries).
- a `ReadWriteMany` storage class for the artifacts.
- the node agent runs privileged, with `hostPID`, `hostIPC` and `hostNetwork`, so its namespace must allow privileged pods - this needs a security review on shared clusters. The workload pods themselves are not privileged, but run with a seccomp profile that blocks `io_uring`, which CRIU can't checkpoint (the Helm chart installs the profile).
- no tools that intercept `libcuda.so` calls inside the workload (e.g., some GPU monitoring agents).

## What can go wrong with process snapshots

These apply to any process checkpoint/restore approach, not just Dynamo Snapshot:

- **A snapshot is pinned to its environment.** It can only be restored with the same container image, the same driver version and the same GPU type. Every image rebuild or driver upgrade requires new snapshots - so creating them has to be automated as part of the deployment pipeline (e.g., with a `SnapshotJob`), and the stale ones cleaned up.
- **Everything resolved at startup is frozen.** Apart from what the restore tooling explicitly remaps (like the pod IP and the GPU in Dynamo Snapshot), the restored process believes it's still where it was captured: the node's hostname, the addresses of other services resolved at startup, established connections to them, values derived from the environment, anything random generated at startup (identical in every replica restored from the same snapshot). Photoroom ran into this with a monitoring agent's host address. Defer such things until after the restore, or redo them in the code that runs once the restore is complete (after `restore-complete` in Dynamo Snapshot's contract).
- **Secrets end up in the snapshot.** The checkpoint contains the process's entire memory, including any tokens and credentials it loaded. Protect the snapshot storage accordingly.
- **Only the expected processes may touch the GPU.** Everything with a CUDA context must be checkpointable. E.g., Photoroom had to make all GPU initialization happen in the worker process, and to disable `torch.compile`'s parallel compile workers (`TORCHINDUCTOR_COMPILE_THREADS=1`), whose memory couldn't be checkpointed.
- **Warm up before capturing.** A snapshot taken before the first generation misses the lazy initialization, the autotuning and the CUDA graphs - the restored replica then pays for them on its first requests.
- **Storage costs.** One snapshot per model, configuration, image and driver version, each roughly the size of the GPU memory in use plus the process's CPU memory - which is why releasing the KV cache before capturing matters so much.

For training, application-level checkpoints remain the way to go: a multi-GPU or multi-node job would need its NCCL communicators and network connections quiesced and re-established around the snapshot, which Dynamo Snapshot doesn't support yet, and the training's own checkpoint is much smaller than a dump of its whole memory.

## When to use what

- The image pull or the node provisioning dominates the cold start -> fix those first, see [Cold start](./inference.md#cold-start). No snapshot helps with them.
- Loading the weights dominates -> faster storage, streaming loaders, a local NVMe copy.
- Compilation and CUDA graph capture dominate -> persistent compile caches.
- Several models take turns on the same GPUs -> sleep mode.
- You need replicas that start in seconds - for fast autoscaling or scale-to-zero - and your model fits on a single GPU -> process snapshots, e.g., Dynamo Snapshot, keeping in mind it's a preview. Managed platforms offer the same idea, e.g., [Modal's GPU memory snapshots](https://modal.com/blog/gpu-mem-snapshots).
