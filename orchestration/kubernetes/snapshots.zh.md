# 快照：恢复而不是冷启动：中文要点

> 英文原文：[Snapshots: Restoring Instead of Starting](./snapshots.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
推理副本要完全初始化之后才能回答第一个请求：创建 CUDA context、把权重加载到 GPU、kernel autotuning、`torch.compile`、capture CUDA graphs。大模型的这个 [冷启动](./inference.md#cold-start) 要好几分钟，而且每个新副本、每次从零扩容、每次重启都要再付一次。快照的思路是只付一次：把初始化好的状态保存下来，之后的副本直接恢复它，而不是重新创建。各种方法保存的状态多少不同，从只保存权重，一直到保存包括 GPU 显存在内的整个运行中的进程，NVIDIA 的 Dynamo Snapshot 在 k8s 上做的就是后者。

## 核心概念
"snapshot" 在 ML 基础设施里指好几种互不相关的东西，先分清楚：

| 术语 | 保存了什么 | 用来做什么 |
| :--- | :--------- | :--------- |
| training checkpoint | 模型权重、优化器状态、RNG、数据位置 | 恢复训练 |
| PyTorch memory snapshot | GPU 显存分配的历史 | 调试显存使用 |
| k8s `VolumeSnapshot` | 某一时刻 PVC 的内容 | 备份或克隆卷，例如一个预先放好模型的缓存卷 |
| process checkpoint/restore | 整个运行中的进程，包括它的 CPU 和 GPU 内存 | 几秒内启动副本、迁移进程（本章主题） |

- **NVIDIA Dynamo ≠ TorchDynamo**：只是重名，毫无关系。NVIDIA Dynamo 是推理服务框架，下文的 Dynamo Snapshot 属于它；TorchDynamo（`torch._dynamo`）是 `torch.compile` 的前端。
- **CRIU (Checkpoint/Restore In Userspace)**：保存和恢复 Linux 进程树，包括内存、线程、打开的文件、socket 和 pipe，也就是 Linux 内核知道的一切，但它对 GPU 一无所知。
- **cuda-checkpoint**：负责进程的 GPU 部分。suspend 时锁住 CUDA API、等已提交的工作完成、把显存拷到主机内存并释放 GPU，之后进程不再占有任何 GPU 资源，CRIU 就能像保存普通进程一样保存它；resume 时重新获取 GPU，把内存拷回原来的虚拟地址，并恢复 stream 和 context。
- **vLLM sleep mode**：进程不退出，只释放 GPU 显存。level 1 把权重卸载到 CPU 内存并丢弃 KV cache，唤醒时把权重拷回，比冷启动快得多；level 2 把权重和 KV cache 都丢弃，用于权重反正要被替换（如 RL 权重更新）或 CPU 内存不够的情况。
- **Dynamo Snapshot 的资源**：`PodSnapshot` 请求对一个运行中的 pod 做 checkpoint，恢复时引用的也是它；`PodSnapshotContent` 是已存储 artifact 的记录，由 operator 管理；`SnapshotJob` 从模板启动 pod，就绪后做 checkpoint 再删掉它，适合流水线；pod annotation `nvidia.com/restore-from` 让新 pod 从指定的 `PodSnapshot` 恢复，而不是从头启动。

## 关键要点
启动优化的“阶梯”：每往上一级，保存的初始化状态越多，跳过的冷启动越多，约束也越多。[详见](./snapshots.md#the-ladder-of-startup-optimizations)

| # | 保存什么 | 跳过什么 | 工具 | 约束 |
| :-: | :------- | :------- | :--- | :--- |
| 1 | 权重，放在快的位置、用快的格式 | 慢速下载和读取 | 本地 NVMe、快速共享存储、Run:ai Model Streamer、预先切分的权重、image volumes | 权重仍要加载，其他一切仍要初始化 |
| 2 | 编译产物 | 重新编译 | 持久化的 `torch.compile`、Triton 和 vLLM 缓存 | 必须与代码、库版本和 GPU 匹配 |
| 3 | 整个进程，保持运行但释放 GPU 显存 | 全部，但只能在同一个 node 上 | vLLM sleep mode、SGLang memory saver | 进程必须一直运行；vLLM 的 sleep level 1 把权重放在 CPU 内存里 |
| 4 | 整个进程，包括 GPU 显存，存成文件 | 全部，可在任何兼容的 node 上 | `cuda-checkpoint` + CRIU、Dynamo Snapshot、Modal GPU memory snapshots | 必须与镜像、driver 和 GPU 类型匹配；需要特权 agent；Dynamo Snapshot 目前只支持单 GPU 负载 |

- **第 1、2 级便宜又稳妥，无论如何都值得做**（见 [Model weights](./storage.md#model-weights) 和 [Caches that make restarts faster](./storage.md#caches-that-make-restarts-faster)）。PyTorch 还能把 `torch.compile` 的缓存导出成 bytes 随模型一起发布，例如在构建步骤里生成、启动时加载；模型仍要编译，但是从缓存编译，不必重新生成和 autotune kernel。选方案之前先 [测量冷启动的时间花在哪](./inference.md#cold-start)：拉镜像、从 autoscaler 拿 node，任何快照都帮不上，如果它们占大头就先解决它们。[详见](./snapshots.md#compile-caches-as-an-artifact)
- **第 3 级 sleep mode**：适合在两次使用之间释放 GPU（多个模型共享同一组 GPU，或 RL 循环在训练和生成之间交替），根本不需要保存进程。`VLLM_SERVER_DEV_MODE=1` 会开启开发用 endpoint，不能暴露给用户。局限是固有的：进程还活着，所以睡眠中的模型仍占着 node 和 CPU 内存，也帮不了在其他 node 上启动副本。SGLang 对应的是 `release_memory_occupation` / `resume_memory_occupation`。[详见](./snapshots.md#keeping-the-process-vllm-sleep-mode)
- **第 4 级的能力在 NVIDIA driver 里**，`cuda-checkpoint` 只是把它暴露出来，所以能做什么取决于 driver 版本：550 引入；570 加入与 CRIU 4.0+ 的集成，以及功能相同的 CUDA driver API（`cuCheckpoint*`）；580 支持 GPU 迁移（恢复到另一块同类型的 GPU）；595 支持 Arm（但 Dynamo Snapshot 本身还不支持 Arm）；610 支持基于 `cuIpcGetMemHandle` 的 CUDA IPC。UVM（managed）内存和用 `cuMemExportToShareableHandle()` 创建的 IPC 内存无法 checkpoint，进程里有它们时 checkpoint 会失败。[详见](./snapshots.md#process-checkpointrestore)
- **k8s 没有原生的 restore**：kubelet 有 checkpoint API（k8s 1.30 起 beta），但它只在 node 上生成一个 checkpoint 压缩包。NVIDIA 在 2026 年发布的 Dynamo Snapshot 填补了这个空白：对完全初始化的 GPU 推理 pod（进程及其 CPU 和 GPU 内存）做 checkpoint，再恢复到任何兼容 node 上的新 pod 里。它集成在 NVIDIA Dynamo 中，也可以作为独立项目配合你自己的推理栈使用。截至 2026-10 独立版本是 0.1.0，作者表示 API 还可能变化，目前不建议用于生产关键负载。它由一个 control-plane operator 加一个特权 node agent（DaemonSet），用一个 Helm chart 安装，实际的 CRIU 和 `cuda-checkpoint` 工作由 agent 完成；再加一个存放 checkpoint artifact 的 `ReadWriteMany` 卷组成。恢复时用一个 pod spec 相同的 Deployment，加上 `nvidia.com/restore-from: vllm-snapshot` annotation，并把容器命令换成什么也不做的 `sleep infinity`，由 agent 把保存的进程恢复进这个容器。项目的 guide 里有 vLLM、SGLang 和 TensorRT-LLM 的完整 manifest。[详见](./snapshots.md#the-components)
- **workload 必须配合**：进程不能在任意时刻做 checkpoint（比如生成到一半，或还没 warm up），所以推理服务要遵守 [workload contract](https://github.com/ai-dynamo/snapshot/blob/main/docs/reference/workload-contract.md)，通过共享目录里的文件和 agent 协调。capture：启动引擎，至少跑一次真实的生成（让 lazy initialization、autotuning 和 CUDA graph capture 都进入快照），停掉进行中的工作，释放不需要保存的显存，然后才写 `ready-for-snapshot` 文件。restore：新 pod 的 entrypoint 必须保持空闲而不是初始化（否则会在恢复出的模型旁边再加载一份），agent 把进程恢复进容器，把它的 pod IP（通过 CRIU 的 inet-remap 插件）和保存的 GPU UUID 重新映射到本 pod 的 IP 和本 node 的 GPU，再发出 `restore-complete`；恢复的进程接着把显存拿回来、恢复生成、检查健康，并报告已在服务。所需的调用引擎都已提供：vLLM 是 `pause_generation()`、`sleep()`、`wake_up()`、`resume_generation()`，SGLang 是 `pause_generation()`、`release_memory_occupation()`、`resume_memory_occupation()`、`continue_generation()`。释放显存这一步非常重要：vLLM 预分配大部分显存给 KV cache，把空的 cache 存下来纯属浪费，NVIDIA 报告释放它之后 B200 上 Qwen3-0.6B 的 checkpoint 从约 190GiB 缩小到约 6GiB。[详见](./snapshots.md#the-workload-has-to-cooperate)
- **有多快**：项目 [benchmark](https://github.com/ai-dynamo/snapshot/blob/main/docs/development/benchmarks.md)（单块 B200、vLLM 0.20、driver 595、checkpoint 放在 VAST NFS 卷上，两列都不含拉镜像和容器启动）中，冷启动 / 恢复分别为：Qwen3 0.6B 52.4s / 3.5s，Qwen3 8B 58.3s / 8.1s，GPT-OSS 120B 85.3s / 31.1s，Qwen3 32B 79.3s / 19.4s，Llama 3.3 70B FP8 102.1s / 22.4s，Qwen2.5 72B 97.8s / 40.9s。恢复时间取决于 checkpoint 的大小而不是参数量（例如 70B FP8 模型的恢复时间约为参数量相近的 bf16 模型的一半；checkpoint 不只有权重，还包括 CUDA context、编译好的 kernel、workspace buffer 和进程的 CPU 内存，所以权重大小相同的 GPT-OSS 120B 和 Qwen3 32B 分别要 31s 和 19s）、存储吞吐（CRIU 从共享卷读取进程镜像的部分占总时间的 49-67%，存储越慢所有数字越大）和 PCIe 带宽（把显存拷回 GPU）。Photoroom（用 Dynamo Snapshot 的 node agent 加上自己的部署工具）的报告展示了另一面：在已有镜像的 node 上，启动从约 220s 降到 35-45s；但在要先拉镜像的新 node 上，只从约 430s 降到约 195s。[详见](./snapshots.md#how-fast-is-it)
- **要求与限制（截至 2026-10）**：containerd 或 CRI-O，x86_64 node，NVIDIA GPU Operator 26.3+ 且 driver 580+，关闭 MIG，不支持 vGPU；只支持单 GPU 负载（多 GPU、多节点在 roadmap 上，需要能让 NCCL 等通信库暂停和恢复的 hook）；artifact 需要 `ReadWriteMany` storage class；node agent 以特权运行，并带 `hostPID`、`hostIPC` 和 `hostNetwork`，它所在的 namespace 必须允许特权 pod，在共享集群上要做安全评审；workload pod 本身不需要特权，但运行时带一个阻止 `io_uring` 的 seccomp profile（CRIU 无法 checkpoint `io_uring`；这个 profile 由 Helm chart 安装）；workload 里不能有拦截 `libcuda.so` 调用的工具（例如某些 GPU 监控 agent）。[详见](./snapshots.md#requirements-and-limitations-as-of-2026-10)
- **怎么选**：拉镜像或准备 node 占大头 -> 先解决它们，快照帮不上；加载权重占大头 -> 更快的存储、流式加载、本地 NVMe 副本；编译和 CUDA graph capture 占大头 -> 持久化编译缓存；多个模型轮流使用同一组 GPU -> sleep mode；需要几秒内启动的副本（快速扩缩容或 scale-to-zero）且模型放得进单块 GPU -> 进程快照，如 Dynamo Snapshot（记住它还是预览版），托管平台也有同样的思路，如 Modal 的 GPU memory snapshots。训练仍应使用应用层 checkpoint：多 GPU 或多节点 job 需要在快照前后让 NCCL communicator 和网络连接暂停并重新建立，Dynamo Snapshot 还不支持；而且训练自己的 checkpoint 比整个内存 dump 小得多。[详见](./snapshots.md#when-to-use-what)

## 常用命令 / 配置
```python
# 构建阶段：编译后的模型跑过一次之后，导出 torch.compile 缓存
result = torch.compiler.save_cache_artifacts()  # 没有编译过任何东西时返回 None
if result is not None:
    artifacts, cache_info = result
    open("compile-cache.bin", "wb").write(artifacts)

# 新进程里，在运行编译后的模型之前加载
torch.compiler.load_cache_artifacts(open("compile-cache.bin", "rb").read())
```

```yaml
# Dynamo Snapshot：对一个运行中的副本做 checkpoint（保存为 vllm-snapshot.yaml）
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
# vLLM sleep mode（dev endpoint 不能暴露给用户）
VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --port 8000
curl -X POST 'http://localhost:8000/sleep?level=1'   # level 1：权重卸载到 CPU 内存，丢弃 KV cache
curl -X POST 'http://localhost:8000/wake_up'         # 唤醒：把权重拷回 GPU
# 单机上的进程 checkpoint/restore（以 root 运行）
cuda-checkpoint --toggle --pid $PID                                  # suspend：GPU 显存 -> 主机内存
criu dump --shell-job --images-dir ckpt --tree $PID                  # 把进程保存到磁盘（进程随之退出）
criu restore --shell-job --restore-detached --images-dir ckpt        # 重新创建进程
cuda-checkpoint --toggle --pid $PID                                  # resume：主机内存 -> GPU
# Dynamo Snapshot：提交上面的 PodSnapshot，并等 checkpoint 完成
kubectl apply -f vllm-snapshot.yaml
kubectl wait --for=condition=Ready podsnapshot/vllm-snapshot --timeout=30m
```

## 常见坑
- （以下各条适用于任何进程快照方案，不只是 Dynamo Snapshot）重建镜像或升级 driver 后快照用不了 -> 快照绑定环境，只能用同一个容器镜像、同一个 driver 版本、同一种 GPU 恢复 -> 把生成快照自动化，放进部署流水线（如用 `SnapshotJob`），并清理过期快照；每个模型、配置、镜像和 driver 版本各一份，每份大约是所用显存加上进程的 CPU 内存，存储成本要算进去（这也是 capture 前释放 KV cache 如此重要的原因）。
- 恢复出的副本还在用旧 node 的 hostname 或旧地址，或各副本的“随机”值都一样 -> 启动时确定的一切都被冻结了：除了恢复工具明确重新映射的东西（如 Dynamo Snapshot 中的 pod IP 和 GPU），恢复出的进程仍以为自己在被 capture 的地方，包括 node 的 hostname、启动时解析的其他服务地址、与它们已建立的连接、从环境推导出的值、启动时生成的随机数（Photoroom 就踩到了监控 agent 的 host 地址） -> 推迟到恢复之后再做，或在恢复完成后运行的代码里重做（在 Dynamo Snapshot 的 contract 中，即 `restore-complete` 之后）。
- 快照存储成了凭据泄露点 -> checkpoint 包含进程的全部内存，包括加载过的 token 和凭据 -> 相应地保护快照存储。
- checkpoint 失败 -> 有不该碰 GPU 的进程也建了 CUDA context，或进程里有无法 checkpoint 的内存 -> 只让预期的进程碰 GPU；Photoroom 把所有 GPU 初始化都放进 worker 进程，并用 `TORCHINDUCTOR_COMPILE_THREADS=1` 关掉 `torch.compile` 的并行编译 worker。
- 恢复出的副本头几个请求很慢 -> 快照是在第一次生成之前拍的，缺少 lazy initialization、autotuning 和 CUDA graphs -> capture 之前先 warm up。
- checkpoint 体积巨大、恢复很慢 -> 把预分配但为空的 KV cache 也存了进去 -> capture 前释放不需要保存的显存。

## 相关章节
- [Inference on Kubernetes](./inference.md)，尤其是 [Cold start](./inference.md#cold-start)
- [Data and Model Loading](./storage.md)：[Model weights](./storage.md#model-weights)、[Caches that make restarts faster](./storage.md#caches-that-make-restarts-faster)
- 训练 checkpoint：[Checkpoints](../../training/checkpoints/)、[Fault Tolerance on Kubernetes](./fault-tolerance.md)；显存快照：[PyTorch memory profiler](../../debug/pytorch.md#pytorch-memory-profiler)
- 外部：[Dynamo Snapshot](https://github.com/ai-dynamo/snapshot)、[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)、[cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint)、[CRIU](https://criu.org/Main_Page)、[kubelet checkpoint API](https://kubernetes.io/docs/reference/node/kubelet-checkpoint-api/)、[Volume Snapshots](https://kubernetes.io/docs/concepts/storage/volume-snapshots/)、[vLLM sleep mode](https://docs.vllm.ai/en/latest/features/sleep_mode.html)、[Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer)、[Photoroom 的报告](https://www.photoroom.com/inside-photoroom/how-we-cut-gpu-cold-starts-from-minutes-to-seconds-with-memory-checkpointing)、[Modal's GPU memory snapshots](https://modal.com/blog/gpu-mem-snapshots)
