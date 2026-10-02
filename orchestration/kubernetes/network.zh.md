# Kubernetes 上的高速跨节点网络：中文要点

> 英文原文：[Fast Inter-node Networking on Kubernetes](./network.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
多节点训练和推理需要后端网络（InfiniBand、RoCE、AWS EFA 或 GCP 的 GPUDirect）在 pod 里可用。陷阱在于：网络没接进来时什么都不会报错，NCCL 会悄悄退回到 pod 网络上的 TCP socket，job 照样运行，只是慢几倍甚至一个数量级。所以在新集群上、或换了新镜像时，一定要验证高速网络确实在被使用。这一章给出 pod 需要什么的清单，以及自下而上的验证步骤。

## 核心概念
- **pod network / `eth0`**：默认情况下 pod 只有一个连到集群 pod 网络的虚拟以太网接口 `eth0`。node 上的其他东西（`/dev/infiniband` 下的 RDMA 设备、额外的网卡、内核模块）pod 都看不到，除非集群专门暴露出来。
- **device plugin（或 DRA driver）**：把 RDMA 设备暴露成可调度的资源，例如用于 IB/RoCE 的 NVIDIA Network Operator，或 AWS 上的 EFA device plugin。
- **Multus**：在 RoCE/SR-IOV 环境中给 pod 附加第二网络，让 pod 多出 `net1`、`net2` 等接口。
- **GPUDirect RDMA**：网卡直接读写 GPU 显存。需要 node 上有 `nvidia-peermem` 内核模块或 DMA-BUF 支持。
- **NCCL plugin**：部分云厂商需要的插件，如 AWS 的 `aws-ofi-nccl`、GCP 用于 GPUDirect-TCPXO 和 RDMA 的插件。
- **`hostNetwork: true`**：让 pod 直接拿到 node 的所有网络接口。更简单，但放弃了 pod 之间的网络隔离，两个 pod 跑在同一个 node 上会端口冲突，所以通常只允许整节点的 job 使用。
- **`busbw`**：all-reduce benchmark 报告的总线带宽，用来和硬件能力对比。

## 关键要点
- pod 需要申请的东西因集群而异：具体的 resource 名称和 annotation 要找管理员或看云厂商文档，这一章只给检查清单和验证方法。[详见](./network.md#why-its-harder-on-k8s)
- pod spec 清单：申请 GPU 和集群特定的 RDMA 资源（例如 NVIDIA Network Operator 的 `rdma/rdma_shared_device_a`、AWS EFA 的 `vpc.amazonaws.com/efa`，EFA 还需要 huge pages）；只有使用 Multus 的集群才需要 `k8s.v1.cni.cncf.io/networks` annotation；加 `IPC_LOCK` capability（RDMA 要 pin 内存）；挂载 `/dev/shm`。[详见](./network.md#what-a-pod-needs)
- 镜像里还必须有对应 fabric 的用户态库：IB/RoCE 需要 `rdma-core`（`libibverbs`、`librdmacm` 和 `libmlx5` provider，NGC 镜像已包含）；AWS EFA 需要 `libfabric` 和 `aws-ofi-nccl`（用 EFA installer 安装，或用 AWS 的 Deep Learning Containers）；GCP 需要 Google 针对机型提供的 NCCL plugin 和 NCCL 环境变量设置。
- 多节点训练总是申请一个 node 的全部 GPU 和全部 NIC：只要 4 个 GPU 的 pod 可能拿到分属不同 PCIe switch 或 NUMA node 的 GPU 和 NIC，影响性能。pod 里用 `nvidia-smi topo -m` 查看哪些 NIC 离哪些 GPU 近。
- 验证第 1 步，pod 能否看到设备：`/dev/infiniband` 下应有 `uverbs0`、`uverbs1` 等，`ulimit -l` 应为 `unlimited`，否则内存注册会失败。AWS EFA 上用 `fi_info -p efa`（在 `/opt/amazon/efa/bin/` 下）。[详见](./network.md#1-does-the-pod-see-the-devices)
- 验证第 2 步，NCCL 是否在用：跑一个 2 节点 job（[multi-node-job.yaml](./multi-node-job.yaml) 已设置 `NCCL_DEBUG=INFO`），日志里应看到 `NET/IB`（RoCE 设备会带 `/RoCE`；AWS 上是 `NET/OFI` 和 `efa` provider；GCP 上是 Google 插件的名字），每个 GPU 对应一个设备（只列出部分 NIC 就只能用上部分带宽），channel 上有 `GDRDMA`（没有它数据要经 CPU 内存中转，更慢）；`OOB eth0` 表示 bootstrap 走 pod 网络，这是正常的。[详见](./network.md#2-does-nccl-use-them)
- 看到 `NET/Socket` 就说明 NCCL 没找到 RDMA 设备，退回了 pod 网络上的 TCP。
- 验证第 3 步，带宽是否达标：日志正确时性能也可能很差（比如交换机配置或 adaptive routing 有问题），所以要实测。最简单的是把 [multi-node-job.yaml](./multi-node-job.yaml) 里的脚本换成 [all_reduce_bench.py](../../network/benchmarks/all_reduce_bench.py)，把得到的 `busbw` 和 [预期值](../../network/benchmarks/README.md#all_reduce-benchmark) 对比；TCP 回退通常表现为跨节点 `busbw` 比硬件能力低一个数量级。先测 2 个节点，再测所有要用的节点，因为一个坏 NIC 或坏线缆就会拖慢整个 job。[详见](./network.md#3-is-the-bandwidth-what-it-should-be)
- pod 放置：同一个 leaf switch 下的 node 之间跳数更少，而原生 k8s 调度不了解拓扑，可能把 pod 撒满整个数据中心。可用 Kueue 的 Topology Aware Scheduling，或云厂商自己的机制（GCP 的 compact placement policy，AWS 的 placement group 和 capacity block），具体问管理员。[详见](./network.md#pod-placement)

## 常用命令 / 配置
```bash
ls /dev/infiniband            # IB/RoCE/EFA：应看到 uverbs0, uverbs1, ...
ibv_devinfo | grep -E "hca_id|state|link_layer"   # 查看 RDMA 设备、端口状态和链路层
ulimit -l                     # 应为 `unlimited`，否则内存注册失败
kubectl describe node NODE | grep -A15 Allocatable   # 检查 node 是否通告了 RDMA 资源
kubectl logs gpu-test-0-xxxxx | grep -E "NET/|via NET"   # 从 NCCL 初始化日志里看用的是哪种传输
```

```
# 期望看到的 NCCL 日志：NET/IB、每个 GPU 一个设备、GDRDMA
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB ... [7]mlx5_7:1/IB [RO]; OOB eth0:10.4.2.17<0>
NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA
# 说明高速网络没用上（退回 TCP）的日志
NCCL INFO NET/Socket : Using [0]eth0:10.4.2.17<0>
```

```yaml
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]  # RDMA 需要 pin 内存
```

## 常见坑
- job 能跑但慢了几倍到一个数量级，没有报错 -> NCCL 日志里是 `NET/Socket`，没找到 RDMA 设备而退回 TCP -> 按清单检查 pod 的 RDMA 资源、annotation 和镜像里的库。
- pod 里没有 `/dev/infiniband` -> pod 没拿到 RDMA 资源 -> 检查 pod 的 `resources`，并确认 node 通告了该资源（`kubectl describe node NODE | grep -A15 Allocatable`）。
- NCCL 初始化时卡住 -> pod 有 Multus 附加的额外接口，NCCL 选错了 bootstrap 用的接口 -> 用 `NCCL_SOCKET_IFNAME=eth0` 显式指定，见 [`NCCL_SOCKET_IFNAME`](../../network/debug/README.md#nccl_socket_ifname)。
- 日志里只列出了部分 NIC -> 只能用上部分带宽 -> 申请 node 的全部 GPU 和全部 NIC。
- 日志看起来没问题，但带宽还是低 -> 交换机配置、adaptive routing，或某个节点的 NIC/线缆有问题 -> 跑 benchmark，先 2 节点再全部节点，逐步排查。

## 相关章节
- 网络背景：[Network](../../network/)；NCCL 通用调试：[Network debug](../../network/debug/)，以及 [NCCL 日志的解释](../../network/debug/README.md#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues)
- 示例：[dev-pod.yaml](./dev-pod.yaml)、[multi-node-job.yaml](./multi-node-job.yaml)
- benchmark：[all_reduce_bench.py](../../network/benchmarks/all_reduce_bench.py)、[Network benchmarks](../../network/benchmarks/README.md#all_reduce-benchmark)；也可以通过 [MPI Operator](https://github.com/kubeflow/mpi-operator) 跑 [nccl-tests](https://github.com/NVIDIA/nccl-tests)
- 云厂商指南：AWS [EFA on EKS](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html)、GCP [GPUDirect on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx)、NVIDIA [Network Operator](https://github.com/Mellanox/network-operator)
- 拓扑感知调度：[Kueue's Topology Aware Scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/)
