# Kubernetes 概览：中文要点

> 英文原文：[Kubernetes](./README.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
作者认为 k8s 并不适合做训练：[SLURM](../slurm/) 构建在 Unix 之上，多节点协调更简单；k8s 则完全重新造了轮子，复杂度高，还缺少很多基本的 Unix 功能（比如没有 Unix 用户名，所有用户都用同一个 Unix 用户名运行），需要很大的支持团队。这一章不打算全面讲 k8s，只讲最重要的坑：给 SLURM/Unix 背景的新人一张术语对照表，说明 GPU 是怎么进入 pod 的，并解决 CPU OOM 导致整个 job 被重置的问题。它也是 k8s 各子章节的入口。

## 核心概念
- **Cluster / Node**：Cluster 是由同一个 control plane 管理的一组机器；Node 是其中的一台机器（VM 或物理机）。
- **Pod**：调度的最小单位。一个或多个容器一起被调度到同一个 node 上，共享网络和 volume，类似 SLURM 中在一个节点上运行的 job step。
- **Container**：从容器镜像启动、运行在隔离文件系统中的进程，类似 chroot 里的进程。
- **Job / Deployment**：Job 把 pod 运行到结束，可以重试，类似 `sbatch`；Deployment 始终保持 N 个相同的 pod 运行，用于推理服务，不用于训练（见 [Inference on Kubernetes](./inference.md)）。
- **Namespace**：集群对象和配额的命名分区，类似 account/partition。
- **Service**：一组 pod 前面的稳定 DNS 名（可选负载均衡），类似 `/etc/hosts` 里的一条记录。
- **PVC (PersistentVolumeClaim)**：申请一块持久存储并挂载到 pod 里，类似共享文件系统的挂载。
- **ConfigMap / Secret**：注入 pod 的小型配置文件或凭据，类似 `~/.config` 里的文件、环境变量。
- **Taint / Toleration**：node 上有 taint 时会“排斥”pod，除非 pod 明确声明能容忍（toleration），类似分区访问限制。
- **`kubectl`**：与集群 API server 通信的命令行客户端，相当于把 `squeue`、`sbatch`、`scancel` 等合成一个工具。
- **NVIDIA GPU Operator**：GPU 集群的管理员通常会安装它，在每个 GPU node 上部署 NVIDIA driver（若 node 镜像没有预装）、NVIDIA Container Toolkit、device plugin（向调度器通告 `nvidia.com/gpu`）、GPU Feature Discovery（添加 `nvidia.com/gpu.product=...` 这类 node label）和 DCGM exporter（GPU 指标）。

## 关键要点
- **一切都是声明式的**：不是在 node 上执行命令，而是提交一份 YAML 描述（`kubectl apply -f job.yaml`），由 k8s 去实现。要修改就改 YAML 再重新 apply，或者删除后重建。[详见](./README.md#key-concepts-for-newcomers)
- **一切都在容器里**：没有“登录计算节点运行 `python train.py`”这回事，代码、python 环境和 CUDA 库都必须在镜像里（或挂载进去）。[详见](../containers/)
- **pod 的文件系统是临时的**：没写到持久卷（PVC、NFS 等）上的东西在 pod 结束时都会消失，包括 checkpoint 和日志。checkpoint 一定要存到持久存储或云对象存储。
- **Pods are cattle, not pets**：k8s 会因为 OOM、node drain、抢占、健康检查失败等原因随时杀掉并重建 pod，训练必须能从最近的 checkpoint 恢复。[详见](./fault-tolerance.md)
- **多节点不是内置功能**：core k8s 没有“一次给我 16 个 node 并告诉每个进程它的 rank”的概念，要么自己搭（见 [multi-node-job.yaml](./multi-node-job.yaml)），要么用专门的 operator。[详见](./users.md#multi-node-training)
- **GPU 是 "extended resource"**：在 pod 的 resource limits 里用 `nvidia.com/gpu: 8` 申请。GPU 不能在 pod 之间共享或超卖（除非管理员配置了 MIG 或 time-slicing），没申请 GPU 的 pod 本不应看到 GPU（但视集群配置有可能看到，见 [这个坑](../containers/drivers.md#how-gpus-get-into-a-container)）。AMD 对应的是 AMD GPU Operator 和 `amd.com/gpu` 资源。
- **driver 在 node 上，CUDA 用户态库在镜像里**：CUDA runtime、cuDNN、NCCL 都在容器镜像里，所以镜像的 CUDA 版本必须被 node 的 driver 支持。[详见](../containers/README.md#cuda-and-the-nvidia-driver)
- **高速跨节点网络要单独配置**：IB/RoCE 用 NVIDIA Network Operator，云厂商有各自的方案（AWS 的 EFA、GCP 的 GPUDirect-TCPXO/RDMA）。RDMA 设备如何暴露给 pod 因集群而异，要问管理员 pod 需要哪些 resource 和 annotation，否则 NCCL 会悄悄退回到很慢的 TCP socket。[详见](./network.md)
- **集群一般不用自己建**：通常来自云厂商（EKS、GKE、AKS）或 neocloud（CoreWeave、Nebius、Lambda 等），管理团队会给你一个 kubeconfig 文件和一个 namespace。
- **CPU OOM 会重置整个 job**：k8s v1.28 起默认 `memory.oom.group = 1`，任何一个进程 CPU OOM 都会把你踢出去、整个 job 重置，连日志都可能没有。这对推理服务合理，对交互式训练调参却是大问题。请管理员在 node pool 或集群级别改成 `memory.oom.group = 0`，这样只杀掉引起 OOM 的那个进程；Kubernetes 1.32 引入的 kubelet 参数 `singleProcessOOMKill` 可以做到这一点。[详见](./README.md#overcoming-job-reset-on-cpu-oom-event)

## 常用命令 / 配置
```bash
kubectl apply -f job.yaml   # 提交 YAML 描述，由 k8s 负责把它变成现实
```

```
# node pool 配置示例：开启 singleProcessOOMKill，让 memory.oom.group = 0
compute:
  additionalNodePools:
    - name: foo
      kubeletConfig:
        singleProcessOOMKill: true
```

```bash
# 在运行中的 node 上检查实际设置：0 表示修复已生效（只杀出问题的进程），1 是会杀掉整个 job 进程组的默认值
$ cat /sys/fs/cgroup/memory.oom.group
```

## 常见坑
- 某个进程 CPU OOM 后整个 job 被踢掉重置、日志也没了 -> 默认 `memory.oom.group = 1` -> 请管理员设置 `singleProcessOOMKill: true`，再用 `cat /sys/fs/cgroup/memory.oom.group` 确认值为 `0`。
- pod 结束后 checkpoint 和日志不见了 -> 写在了 pod 的临时文件系统上 -> 写到 PVC、NFS 或云对象存储。
- 多节点训练能跑但很慢，没有任何报错 -> pod 没拿到 RDMA 设备，NCCL 悄悄退回 TCP socket -> 向管理员确认需要的 resource 和 annotation，按 [network.md](./network.md) 验证。
- 镜像里的 CUDA 用不了 -> 镜像的 CUDA 版本不被 node 的 driver 支持 -> 见 [CUDA and the driver](../containers/README.md#cuda-and-the-nvidia-driver) 和 [Drivers](../containers/drivers.md)。
- pod 里看不到 GPU -> pod 没有申请 GPU -> 在 resource limits 里加上 `nvidia.com/gpu`。反过来，没申请 GPU 的 pod 却能用 GPU，见 [Drivers](../containers/drivers.md#how-gpus-get-into-a-container)。

## 相关章节
- [Day One on a New Cluster](./new-cluster-checklist.md)：在新集群上跑任何昂贵任务之前，逐步验证集群的检查清单。
- [Kubernetes for Users](./users.md)：做 ML 工作需要的 `kubectl` 命令、pod spec 设置和常见故障诊断。
- [Fault Tolerance](./fault-tolerance.md)：应对硬件故障、抢占和 node 升级。
- [Fast Inter-node Networking](./network.md)：把 InfiniBand/RoCE/EFA 接进 pod 并验证 NCCL 确实在用。
- [Data and Model Loading](./storage.md)：数据集、模型权重、checkpoint 和缓存放在哪里。
- [Inference on Kubernetes](./inference.md)：部署 LLM 推理服务：health probe、更新、多节点副本、冷启动、自动扩缩容。
- [Containers](../containers/) 和 [如何在 CI 中构建镜像](../containers/ci.md)；完整驱动栈见 [Drivers: Kernel, Host and Container](../containers/drivers.md)。
- 示例文件：[dev-pod.yaml](./dev-pod.yaml)（交互式 GPU pod，相当于 SLURM 的 `salloc`）、[multi-node-job.yaml](./multi-node-job.yaml)（最小的多节点 `torchrun` job）、[jobset-train.yaml](./jobset-train.yaml)（容错的多节点训练 job）、[vllm-deployment.yaml](./vllm-deployment.yaml)（vLLM 推理服务）。
- 对比：[SLURM](../slurm/)。
- 入门资料：[Kubernetes Basics tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)、在笔记本上跑玩具集群练习 `kubectl` 的 [kind](https://kind.sigs.k8s.io/) / [minikube](https://minikube.sigs.k8s.io/)、[kubectl cheatsheet](https://kubernetes.io/docs/reference/kubectl/quick-reference/)、终端 UI [k9s](https://k9scli.io/)。
