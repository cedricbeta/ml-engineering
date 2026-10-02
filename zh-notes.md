# 中文要点索引：容器、Kubernetes 与性能分析

这里汇总了容器、Kubernetes 和 profiling 相关章节的中文要点页。每一页大约一页纸，帮助快速抓住重点；细节、完整示例和解释以英文原文为准。

## 建议的阅读顺序

先打基础，再上集群，最后讲运维和性能：

| # | 中文要点 | 英文原文 | 一句话 |
| :-: | :------- | :------- | :----- |
| 1 | [容器基础](orchestration/containers/README.zh.md) | [Containers](orchestration/containers/README.md) | 镜像、容器、GPU 容器怎么跑，怎么写 Dockerfile |
| 2 | [驱动：内核、宿主机与容器](orchestration/containers/drivers.zh.md) | [Drivers](orchestration/containers/drivers.md) | 哪些来自宿主机、哪些来自镜像，三个"CUDA 版本"，报错对照表 |
| 3 | [Kubernetes 概念](orchestration/kubernetes/README.zh.md) | [Kubernetes](orchestration/kubernetes/README.md) | k8s 与 SLURM 的术语对照，新手最容易踩的坑 |
| 4 | [Kubernetes 用户指南](orchestration/kubernetes/users.zh.md) | [Kubernetes for Users](orchestration/kubernetes/users.md) | 日常 `kubectl`、pod spec 必备配置、状态排错表 |
| 5 | [新集群第一天](orchestration/kubernetes/new-cluster-checklist.zh.md) | [Day One on a New Cluster](orchestration/kubernetes/new-cluster-checklist.md) | 拿到新集群后按步骤验证 |
| 6 | [数据与模型加载](orchestration/kubernetes/storage.zh.md) | [Data and Model Loading](orchestration/kubernetes/storage.md) | 数据集、权重、checkpoint、缓存放哪里 |
| 7 | [高速网络](orchestration/kubernetes/network.zh.md) | [Fast Inter-node Networking](orchestration/kubernetes/network.md) | 让 pod 用上 RDMA，并确认 NCCL 真的在用 |
| 8 | [容错](orchestration/kubernetes/fault-tolerance.zh.md) | [Fault Tolerance on Kubernetes](orchestration/kubernetes/fault-tolerance.md) | checkpoint/续训、整体重启、优雅退出 |
| 9 | [推理部署](orchestration/kubernetes/inference.zh.md) | [Inference on Kubernetes](orchestration/kubernetes/inference.md) | 健康检查、滚动更新、多节点副本、冷启动、扩缩容 |
| 10 | [快照：恢复而不是冷启动](orchestration/kubernetes/snapshots.zh.md) | [Snapshots](orchestration/kubernetes/snapshots.md) | 编译缓存、sleep mode、进程级 checkpoint/restore（Dynamo Snapshot） |
| 11 | [CI 构建镜像](orchestration/containers/ci.zh.md) | [Building Images in CI](orchestration/containers/ci.md) | 构建、测试、打标签、推送 |
| 12 | [读懂 profiler trace](training/performance/profiling.zh.md) | [Reading Profiler Traces](training/performance/profiling.md) | `torch.profiler`、Perfetto、Nsight Systems/Compute |

## 可以直接用的文件

| 文件 | 用途 |
| :--- | :--- |
| [driver-report.sh](orchestration/containers/driver-report.sh) | 一键输出驱动栈信息，宿主机和 pod 里各跑一次对比 |
| [dev-pod.yaml](orchestration/kubernetes/dev-pod.yaml) | 交互式 GPU pod（相当于 SLURM 的 `salloc`） |
| [multi-node-job.yaml](orchestration/kubernetes/multi-node-job.yaml) | 最小的多节点 `torchrun` 测试 |
| [jobset-train.yaml](orchestration/kubernetes/jobset-train.yaml) | 带容错的多节点训练 |
| [vllm-deployment.yaml](orchestration/kubernetes/vllm-deployment.yaml) | vLLM 推理服务 |
| [build-image.yml](orchestration/containers/build-image.yml) | GitHub Actions 构建镜像的工作流 |

注意：这些文件都还没有在真实集群上验证过，使用前请按自己集群的情况修改（镜像版本、PVC 名称、RDMA 资源名等）。
