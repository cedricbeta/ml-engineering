# 新集群第一天检查清单：中文要点

> 英文原文：[Day One on a New Cluster](./new-cluster-checklist.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
拿到一个新的 GPU 集群（或新的 node pool、新的镜像）后，在上面跑任何昂贵的任务之前，先花一天验证它确实提供了你付钱买的东西。现在发现的每个问题，比如坏 GPU、退回 TCP 的 NIC、慢存储、无法恢复的 job，否则都会在大规模训练的中途才暴露，代价高得多。这份清单按依赖顺序把其他章节的工具串起来（每一步都依赖上一步正常），每一步都给出要运行什么、好的结果是什么样，以及结果不好时去哪里查。它是为 k8s 写的，但同样适用于 SLURM：用 `srun` 而不是 pod 来启动同样的脚本。

## 核心概念
- **baseline（基线）**：把测到的数字记下来，以后觉得变慢时拿来对比；任何 driver、镜像或集群变更之后都要重跑这些检查。
- **[driver-report.sh](../containers/driver-report.sh)**：一次性输出驱动栈的完整报告。
- **torch-distributed-gpu-test.py**：检查一个 node 上所有 GPU 能否互相通信。
- **[mamf-finder-all-gpus.py](../../compute/accelerator/benchmarks/mamf-finder-all-gpus.py)**：在 GPU0 上运行 [mamf-finder.py](../../compute/accelerator/benchmarks/mamf-finder.py) 的自动搜索，再在其他 GPU 都满载计算的情况下逐块测量，得到每块 GPU 的 MAMF（瞬时峰值）和 MSMF（持续算力）。两个脚本要放在同一目录。
- **[evaluate-cluster](../../skills/evaluate-cluster/SKILL.md) skill**：书里更完整的硬件验收流程（DCGM 诊断、每块 GPU 的 MAMF/MSMF、节点内外 all-reduce、fio，并生成带日期的报告），可以让 AI agent 代跑；这份清单补充的是 k8s 特有的部分：pod 里的驱动栈、pod 里的 RDMA、PVC 和容错演练。
- **[all_reduce_bench.py](../../network/benchmarks/all_reduce_bench.py) / `busbw`**：测量跨节点网络带宽。
- **[fio-scan](../../storage/fio-scan)**：用 fio 测量存储，靠 `fio-json-extract.py` 汇总结果，所以两个文件都要下载。
- **容错演练（fault tolerance drill）**：在训练运行中故意触发保存、删除 pod，确认 job 能恢复。

## 关键要点
- **在哪里运行**：使用本仓库 YAML 和脚本的命令（`dev-pod.yaml`、`multi-node-job.yaml`、`../containers/driver-report.sh`）假设你在 [仓库](https://github.com/stas00/ml-engineering) checkout 的 `orchestration/kubernetes/` 目录下运行，并且先按你的集群改好；在 pod 里运行的命令会用 `curl` 下载所需的文件。
- **第 0 步，访问权限与清点**：确认能创建 pod，node 数量符合预期，所有 node 的 GPU 型号和 driver 版本都一样。留意 GPU 比其他 node 少的 node（有 GPU 掉卡了），以及 driver 版本不同的 node（在不同时间配置的，行为可能不同）。[详见](./new-cluster-checklist.md#0-access-and-inventory)
- **第 1 步，驱动栈**：在 [dev pod](./dev-pod.yaml) 里运行 driver-report.sh。好的结果是：内核 driver 版本与 `libcuda.so` 版本相同；driver 支持的 CUDA 版本 >= PyTorch 构建所用的版本，`torch.cuda.is_available()` 为 `True` 且所有 GPU 可见；GPU 架构（如 `sm_90`）在 torch 编译的 arch 列表里；NVSwitch 系统上 Fabric 状态为 `Completed` / `Success`；RDMA 设备存在、端口 `ACTIVE` 且速率符合预期、memlock 限制为 `unlimited`。否则查 [Symptom to cause](../containers/drivers.md#symptom-to-cause)。顺便记下 pod 就绪花了多久，其中大部分是拉镜像，每个新 node 都要付这个代价。[详见](./new-cluster-checklist.md#1-the-driver-stack)
- **第 2 步，单个 node**：torch-distributed-gpu-test.py 应该每个 rank 都报 OK，卡住就看 [network debug](../../network/debug/)。再用 mamf-finder-all-gpus.py 测算力，MAMF 和 MSMF 应接近 [对比表](../../compute/accelerator/README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table) 里这款 GPU 的值，并且各块 GPU 之间差距要小，因为一块慢 GPU 就会拖慢整个同步训练 job；在多个 node 上重复。[详见](./new-cluster-checklist.md#2-a-single-node)
- **第 3 步，多节点连通性**：在 2 个 node 上跑 [multi-node-job.yaml](./multi-node-job.yaml)，NCCL 日志里应看到 `NET/IB`（AWS 上是 `NET/OFI`）、每个 GPU 列出一个 NIC、channel 行里有 `GDRDMA`。看到 `NET/Socket` 说明高速网络没用上。[详见](./network.md#2-does-nccl-use-them)
- **第 4 步，网络带宽**：用同一个 job 跑 all_reduce_bench.py，先 2 个 node，再所有要用的 node。`busbw` 应符合 [这类硬件的预期](../../network/benchmarks/README.md#all_reduce-benchmark)，并且从 2 个 node 扩到全部 node 时下降不多；如果下降明显就二分排查：两两配对跑，找出慢的 node、NIC 或链路。[详见](./new-cluster-checklist.md#4-network-bandwidth)
- **第 5 步，存储**：在挂载了共享 PVC 的 pod 里，按你的工作负载的访问模式测量存储。另外检查：PVC 能否被 2 个不同 node 上的 pod 同时挂载（`ReadWriteMany`），一边写的文件另一边能否看到；写入并读回一个 checkpoint 大小的文件要多久（这决定了 checkpoint 的频率）；从它加载模型权重要多久。[详见](./new-cluster-checklist.md#5-storage)
- **第 6 步，容错演练**：用小模型通过 [jobset-train.yaml](./jobset-train.yaml) 跑你真正的训练代码，运行中：① `touch /tmp/save-and-exit`，job 应保存 checkpoint、退出、重启并从这个 checkpoint 恢复；② 删除其中一个 pod，job 应通过 `preStop` hook 保存、重启所有 pod 并恢复，注意直接删除在 k8s 看来不算 disruption（pod 不会得到 `DisruptionTarget` condition），所以这次重启会计入 `maxRestarts`，这一点和真正的抢占不同；③ 检查 loss 曲线在多次重启之间是否平滑，跳变说明丢了优化器状态或数据顺序变了。部署推理服务的话也做对应的演练：在负载下删掉一个副本，确认没有请求失败。[详见](./new-cluster-checklist.md#6-fault-tolerance-drill)
- **第 7 步，可观测性**：第一次正式运行之前确认：能看到每块 GPU 的利用率、显存和温度（通常是 DCGM exporter + Grafana，向管理员要 dashboard）；pod 删除后日志还在不在（有没有日志收集系统，还是要自己写到 PVC）；一天后去哪里看失败 job 的事件（`kubectl get events` 默认只保留大约一小时）。[详见](./new-cluster-checklist.md#7-observability)
- **记录基线**：每个集群一张表，记录 driver/CUDA/NCCL 版本、新 node 上的拉镜像时间、MAMF / MSMF bf16 TFLOPS（每块 GPU 的最小值/中位数）、2 节点和 N 节点的 all-reduce busbw、存储顺序读/写 GB/s、checkpoint 保存/加载时间、模型权重加载时间、从重启到恢复训练的时间。[详见](./new-cluster-checklist.md#record-the-baseline)

## 常用命令 / 配置
```bash
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count,nvidia.com/cuda.driver-version.full   # 每个 node 的 GPU 型号、数量和 driver 版本，用来找出异类
kubectl exec -i dev-pod -- bash -s < ../containers/driver-report.sh   # 在 dev pod 里跑驱动栈报告
NCCL_DEBUG=INFO torchrun --nproc-per-node=8 torch-distributed-gpu-test.py   # 单个 node 内所有 GPU 的通信测试（脚本先用 curl 下载）
python mamf-finder-all-gpus.py   # 测量每块 GPU 的 matmul 算力（mamf-finder.py 和它放在同一目录）
kubectl logs -l job-name=gpu-test --prefix --tail=-1 | grep -E "NET/|via NET"   # 多节点：看 NCCL 用的是哪种网络
bash fio-scan /shared   # 测量共享存储（需要先装 fio，并下载 fio-scan 和 fio-json-extract.py）
kubectl exec POD -- touch /tmp/save-and-exit   # 容错演练：触发保存-退出-重启-恢复
kubectl delete pod POD   # 容错演练：删除一个 pod，应通过 preStop hook 保存后整体重启（计入 maxRestarts）
```

## 常见坑
- 有的 node 上 GPU 比别的 node 少 -> 某块 GPU 掉卡了 -> 第 0 步用 `kubectl get nodes -L ...` 列出 GPU 数量就能发现。
- 整个同步训练 job 都慢 -> 某一块 GPU 比其他的慢 -> 在多个 node 上跑 mamf-finder-all-gpus.py，看每块 GPU 的 MSMF 和最慢的那块。
- NCCL 日志里是 `NET/Socket` -> 高速网络没被使用 -> 按 [Fast Inter-node Networking](./network.md) 排查。
- 从 2 个 node 扩到全部 node 时 `busbw` 明显下降 -> 有慢的 node、NIC 或链路 -> 两两配对跑 benchmark，二分定位。
- 重启前后 loss 曲线出现跳变 -> 丢了优化器状态或数据顺序变了 -> 保存完整的训练状态，见 [Checkpoint and resume](./fault-tolerance.md#checkpoint-and-resume)。
- 第二天想查失败 job 的事件却查不到 -> `kubectl get events` 默认只保留大约一小时 -> 第一次正式运行前就确认事件和日志在哪里能看到。

## 相关章节
- [Getting connected](./users.md#getting-connected)、[Finding the GPUs](./users.md#finding-the-gpus)
- [Drivers: Symptom to cause](../containers/drivers.md#symptom-to-cause)
- [Not all accelerators are created equal](../../compute/accelerator/README.md#not-all-accelerators-are-created-equal)
- [Is the bandwidth what it should be?](./network.md#3-is-the-bandwidth-what-it-should-be)、[network debug](../../network/debug/)
- [fio](../../storage/README.md#fio)、[Data and Model Loading](./storage.md)、[Frequent checkpoint saving](../../training/fault-tolerance/README.md#frequent-checkpoint-saving)
- [Fault Tolerance on Kubernetes](./fault-tolerance.md)、推理侧的演练见 [Updates and shutdowns](./inference.md#updates-and-shutdowns)
