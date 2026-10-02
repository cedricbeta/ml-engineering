# 驱动：内核、宿主机与容器：中文要点

> 英文原文：[Drivers: Kernel, Host and Container](./drivers.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
GPU 集群上很多令人困惑的故障，归根结底是驱动栈的各层彼此不匹配：“节点 A 上能跑、节点 B 上不行”，“宿主机上 `nvidia-smi` 正常、容器里不行”，“管理员升级了什么之后就坏了”。这一章讲清楚涉及哪些驱动、每个做什么、谁依赖谁、哪些来自宿主机、哪些来自容器镜像，以及如何逐一检查。想一次拿到完整报告，可以在宿主机或容器/pod 里运行 [driver-report.sh](./driver-report.sh)。

## 核心概念
- **kernel module（内核模块）**：内核驱动，如 `nvidia`、`nvidia_uvm`、`nvidia_peermem`，以及 RDMA 的 `ib_core`、`ib_uverbs`、`mlx5_core`、`mlx5_ib`、`efa` 等，永远来自宿主机。
- **driver 用户态库**：`libcuda.so`（CUDA driver API）、`libnvidia-ml.so`（NVML，`nvidia-smi`、DCGM、k8s device plugin 等都用它）、`libnvidia-ptxjitcompiler.so`（运行时把 PTX 编译成机器码）和 `nvidia-smi` 等二进制，必须与已加载的内核模块版本完全一致。
- **GSP firmware**：从 Turing 起，GPU 有一个 GPU System Processor，driver 把部分初始化和管理工作交给它。固件随 driver 包发布，必须来自同一 driver 版本。
- **宿主机服务**：`nvidia-persistenced`（让 driver 在无进程使用 GPU 时也保持初始化）、`nvidia-fabricmanager`（配置 NVSwitch 和 NVLink fabric；在 B200/B300 系统上还会启动 NVLink Subnet Manager（NVLSM））、`nvidia-imex`（GB200/GB300 NVL72 这类机架级系统上让不同节点的 GPU 通过 NVLink 共享内存）、DCGM（健康检查、诊断和指标）。
- **CDI (Container Device Interface)**：较新的 toolkit 用一个 spec 文件（如 `/etc/cdi/nvidia.yaml`，由 `nvidia-ctk cdi generate` 生成）列出要注入的设备文件和库。
- **compute capability**：GPU 架构编号，如 H100/H200 是 `sm_90`，B200/GB200 是 `sm_100`。
- **driver container**：GPU Operator 和 Network Operator 用的特权容器，负责编译并把内核模块加载进宿主机内核，结果仍是整个 node 共享的宿主机级 driver。

## 关键要点
- **唯一的规则：容器共享宿主机的内核**（容器里 `uname -r` 显示的是宿主机内核版本）。因此内核模块总是来自宿主机；容器只能看到传给它的设备（k8s 上通过 device plugin）；用户态库来自镜像，唯一例外是 NVIDIA driver 自己的用户态库，它们在容器启动时从宿主机注入；模块参数、IOMMU、huge pages、cgroups 等内核设置属于宿主机，容器里大多能读到，但修改要由 node 的管理员来做。[详见](./drivers.md#the-one-rule-containers-share-the-hosts-kernel)
- **driver 用户态库必须与内核模块版本完全一致**，否则报 `Failed to initialize NVML: Driver/library version mismatch`。通常发生在 driver 升级后：磁盘上的库被替换了，但旧的内核模块还在运行，需要重启（或停掉所有用 GPU 的进程后重新加载模块）。在容器里出现这个错误有两种可能：同一个宿主机问题（注入的是宿主机磁盘上的新库，而旧的内核模块还在运行），或者镜像里有自己的一份 driver 库（这时要从镜像里删掉）。如果宿主机上的 `nvidia-smi` 正常，就是后者。[详见](./drivers.md#driver-user-space-libraries)
- **三个“CUDA 版本”**：① driver 支持的最新 CUDA 版本（`nvidia-smi` 右上角）；② 程序实际使用的 CUDA runtime（来自镜像，PyTorch 通常打包在 pip wheel 里）；③ CUDA toolkit（`nvcc`，只在 `devel` 镜像里）。规则：①必须 >= ②；同一大版本内有 minor version compatibility（CUDA 12.x runtime 配 >= 525 的 driver，13.x 配 >= 580，但有限制，比如 JIT 编译比 driver 新的 PTX 不行）；新 driver 总能跑旧 runtime，所以升级 driver 不必重建镜像；数据中心 GPU 可以用 `cuda-compat` 包做 forward compatibility；`nvcc` 版本只在编译时重要（构建 PyTorch 扩展时，它要与 PyTorch 构建所用的 CUDA 版本兼容）。[详见](./drivers.md#the-three-cuda-versions)
- **第 4 个维度是 GPU 架构**，与 driver 无关：PyTorch 或 flash-attention 等里的 CUDA kernel 必须为你的 GPU 编译过，否则报 `no kernel image is available for execution on the device`。为 `sm_80` 编译的二进制能在 `sm_86` 上运行，但不能在 `sm_90` 上运行；新一代 GPU 需要足够新的 CUDA（如 Blackwell 需要 CUDA 12.8+）。
- **GPU 如何进入容器**：NVIDIA Container Toolkit 传入所申请 GPU 的设备文件以及 `/dev/nvidiactl`、`/dev/nvidia-uvm`，挂载宿主机的 driver 用户态库和 `nvidia-smi`，并更新容器的库缓存。注入内容由 `NVIDIA_VISIBLE_DEVICES`（哪些 GPU）和 `NVIDIA_DRIVER_CAPABILITIES`（哪些库，未设置时默认 `utility,compute`）控制。`CUDA_VISIBLE_DEVICES` 是另一回事：它由进程内的 CUDA runtime 读取，只能隐藏容器已经拿到的 GPU。[详见](./drivers.md#how-gpus-get-into-a-container)
- **没申请 GPU 的 pod 也可能看到 GPU**：NVIDIA 的 CUDA 基础镜像设置了 `NVIDIA_VISIBLE_DEVICES=all`，视集群配置，toolkit 可能照此给没申请 GPU 的 pod 访问别人的 GPU。NVIDIA 建议让 toolkit 在非特权容器中忽略这个变量，并让 device plugin 改用 volume mounts 传递设备（`ACCEPT_NVIDIA_VISIBLE_DEVICES_ENVVAR_WHEN_UNPRIVILEGED=false` 和 `DEVICE_LIST_STRATEGY=volume-mounts`）；发现没申请 GPU 的 pod 在用 GPU，要告诉管理员。特权容器无论申请了什么都能看到 node 的所有设备。
- **Fabric Manager 是最常见的麻烦来源**：它没运行，或版本与 driver 不一致时，`nvidia-smi` 里 GPU 看起来正常，但所有 CUDA 程序都报 `cudaErrorSystemNotReady`（error 802）。[详见](./drivers.md#host-services)
- **open vs proprietary 内核模块**：R560 起默认是 open 模块；Grace Hopper 和 Blackwell 只支持 open，Maxwell、Pascal、Volta 只支持 proprietary，Turing 到 Hopper 两者都行（NVIDIA 推荐 open）；DMA-BUF 方式的 GPUDirect RDMA 需要 open 模块。`/proc/driver/nvidia/version` 第一行有 `NVIDIA UNIX Open Kernel Module` 就是 open。[详见](./drivers.md#open-vs-proprietary-kernel-modules)
- **RDMA**：与 GPU driver 不同，RDMA 用户态（`rdma-core`）来自镜像、不注入，而 verbs 的内核-用户态接口是稳定的，所以通常能配合宿主机上任意版本的内核驱动。NIC 固件是独立的一层，不同节点固件不同会导致性能不一致。GPUDirect RDMA 有两种内核桥接：DMA-BUF（NVIDIA 推荐，需要 open GPU 模块、Linux 5.12+、CUDA 11.7+、Turing 或更新的数据中心/RTX GPU，可配合 inbox NIC 驱动）和旧的 `nvidia_peermem`（需要 MLNX_OFED/DOCA-OFED）。[详见](./drivers.md#gpudirect-rdma)
- **升级 driver 时要一起动**：内核模块和用户态库、GSP 固件、Fabric Manager（NVSwitch 服务器上，版本要与 driver 兼容，实际上就是同一版本，因为它们一起发布；NVL72 这类机架上 Fabric Manager 在 switch tray 上，而 compute tray 上的 `nvidia-imex` 按 driver 分支打包，所有 tray 上必须是同一版本）、CDI spec（若使用）要同步，并且重启（或完整重新加载模块）。先 `kubectl drain NODE --ignore-daemonsets` 再升级，全集群升到同一版本。AMD 的分工不同：宿主机上只有 `amdgpu` 内核模块，整个 ROCm 用户态都来自镜像，不注入任何东西。[详见](./drivers.md#upgrading-drivers)

## 多节点 NVLink 系统（GB200/GB300 NVL72）
以上内容适用于任何 NVIDIA GPU 服务器。GB200 NVL72、GB300 NVL72 和 GH200 NVL32 这类机架级系统的 NVLink 跨越多台机器，称为 multi-node NVLink（MNNVL）；只用常见 8 卡 HGX/DGX 服务器（H100、H200、B200、B300）的话可以跳过这一节。[详见](./drivers.md#multi-node-nvlink-systems-gb200gb300-nvl72)

**B200 vs GB200**：B200 是一块 GPU；GB200 是一个 "superchip"，由一个 Grace CPU（NVIDIA 的 Arm CPU，即名字里的 "G"）加 2 块 B200 GPU 组成。同理，GB300 用的是 B300，GH200 是 Grace CPU 加一块 Hopper GPU。GPU 本身的架构相同（两者都需要 open 内核模块，所有 Blackwell GPU 都是如此），区别在于怎么组装成机器：

| | HGX/DGX B200 | GB200 NVL72 |
| :- | :----------- | :---------- |
| 单位 | 一台服务器：8 块 B200 + 2 个 x86 CPU | 一个机架：18 个 compute tray + 9 个 NVLink switch tray |
| 每台机器（OS 实例）的 GPU 数 | 8 | 每个 compute tray 4 块（2 个 GB200 = 2 个 Grace CPU + 4 块 GPU） |
| NVLink 域 | 一台服务器里的 8 块 GPU | 整个机架的 72 块 GPU，跨 18 台机器 |
| 机器之间 | 网卡（InfiniBand/RoCE）：数据经网络拷贝 | NVLink：GPU 直接读写彼此的显存 |
| CPU 架构 | x86_64 | Arm（aarch64），需要 `arm64` 镜像 |
| CPU-GPU 连接 | PCIe | NVLink-C2C，内存是 coherent 的（见 [coherent memory 的坑](../../debug/pytorch.md#overcoming-the-coherent-memory-uncertain-behavior)） |
| fabric 管理 | Fabric Manager 和 NVLink Subnet Manager 跑在服务器本机上 | Fabric Manager 和 NVLink Subnet Manager 跑在 switch tray 上；IMEX 跑在每个 compute tray 上 |

实际影响：job spec 里每个 node 是 4 块 GPU（`--nproc-per-node=4`、`nvidia.com/gpu: 4`），镜像要构建成 `arm64`（见 [CPU architecture](./ci.md#cpu-architecture)），跨节点 NVLink 还需要下面这些软件。

**谁负责什么**：NVLink switch tray（运行 NVIDIA 的交换机 OS NVOS）上，NVLink Subnet Manager 发现 NVLink 拓扑并配置交换机的转发表，Fabric Manager 把交换机配置成 GPU 之间统一的内存 fabric，用户不直接接触它们，但它们不健康时下面的一切都不工作；每个 compute tray 上是 GPU driver（open 内核模块）、`nvidia-persistenced` 和 `nvidia-imex` daemon；你的进程里是 CUDA（`libcuda.so`）和 NCCL、NVSHMEM 等通信库，跨节点 NVLink 可用时它们会自动使用。每块 GPU 在 `nvidia-smi -q` 的 `Fabric` 部分报告自己在 fabric 中的位置：`State: Completed` 且 `Status: Success` 表示这块 GPU 的 fabric 已建好；`ClusterUUID` 相同的 GPU 在同一个 NVLink 域，其中 `CliqueId` 也相同的 GPU 能通过 NVLink 互访，NCCL 就据此决定哪些 rank 之间走 NVLink。k8s 上 GPU Feature Discovery 把它发布为 node label `nvidia.com/gpu.clique`（`<ClusterUUID>.<CliqueId>`）。

**显存如何跨机器共享**：同一台机器里，进程之间用 CUDA IPC handle 共享显存，但 handle 只对本机 driver 有意义，node B 的 driver 对 node A 上的分配一无所知。MNNVL 用 fabric handle 加 IMEX 服务解决这个问题。一块要被另一个 node 上的 GPU 读写的 buffer，一生是这样的：
1. **分配**：用 CUDA 虚拟内存管理 API 并指定 fabric handle 类型：`cuMemCreate` + `CU_MEM_HANDLE_TYPE_FABRIC`（CUDA 12.4+；`CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED` 表示 GPU 是否支持）。
2. **导出**：`cuMemExportToShareableHandle` 返回一个 fabric handle，就是一小段字节；它和文件描述符不同，不绑定某个进程或机器，可以用任何方式传递。
3. **发送**：通过普通网络连接把这些字节发给另一个进程，例如 NCCL 用它的 TCP bootstrap 连接，MPI 程序用 MPI。
4. **导入**：在另一个 node 上调用 `cuMemImportFromShareableHandle`。那边的 driver 询问本地 IMEX daemon，后者向导出方 node 的 IMEX daemon 确认这次导入是允许的，并拿到映射内存所需的信息。
5. **映射**：映射进进程的地址空间（`cuMemMap`、`cuMemSetAccess`）。此后 node B 上的 kernel 直接通过 NVLink 读写 node A 那块 GPU 上的显存，数据经过 NVLink 交换机，不经过网卡或 CPU。
6. **释放**：用完后两边都 unmap 并释放 handle。

IMEX daemon 之间只交换这些簿记信息（通过以太网上的 TCP），数据本身始终走 NVLink。[详见](./drivers.md#how-gpu-memory-is-shared-across-machines)

**IMEX：daemon、domain 和 channel**：`nvidia-imex` daemon 跑在每个 compute tray 上，daemon 彼此认识的那些 tray 组成一个 IMEX domain，即可以共享显存的 node 集合；手动配置时 peer 列在 `/etc/nvidia-imex/nodes_config.cfg` 里（每行一个 IP 地址），daemon 自己的设置在 `/etc/nvidia-imex/config.cfg`。channel 是设备文件 `/dev/nvidia-caps-imex-channels/channelN`：进程必须能访问某个 channel 才能导出、导入 fabric 内存，两个进程只有用同一个 channel 才能共享内存，同一个 NVLink 域里不同用户或 job 的显存就是靠它隔离的，但前提是每个用户或 job 独占自己的 channel（driver 会使用进程能访问的编号最小的 channel）；`channel0` 可以用模块参数 `NVreg_CreateImexChannel0=1` 在 driver 加载时创建，也可以用 `mknod` 手动创建，而一个所有人都能访问的 channel0 是单租户系统的简单做法，等于放弃了这种隔离。单台机器不需要这些，同一台机器上的进程不用 IMEX 也能共享显存。容器里要像 GPU 一样把 channel 设备传进去：NVIDIA Container Toolkit 用 `NVIDIA_IMEX_CHANNELS=0`（channel ID 的逗号分隔列表）；k8s 上由 NVIDIA 的 [DRA driver for GPUs](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu)（k8s 1.32+）处理：创建一个 `ComputeDomain`，driver 为你的 workload 所在的 node 运行 IMEX daemon，并创建一个申请 channel 用的 `ResourceClaimTemplate`，pod 再去 claim 它。ComputeDomain 随 workload 创建、随它消失，并把这个 workload 的显存与同一机架上的其他 workload 隔离。YAML 里的 `numNodes` 已经废弃，`0` 不是“没有 node”，而是当前版本推荐的值。DRA driver 的指南还给 pod 加了基于 `nvidia.com/gpu.clique` label 的 node affinity，确保 pod 只落在 MNNVL node 上，并提醒 pod 处于 `Running` 并不能证明 IMEX 正常，要检查 pod 里存在 `/dev/nvidia-caps-imex-channels/channel0`。[详见](./drivers.md#imex-daemons-domains-and-channels)

**NCCL 如何使用**：由 `NCCL_MNNVL_ENABLE` 控制：`2`（默认）在 communicator 跨多个 node 时检查 MNNVL，`1` 总是检查，`0` 从不使用。检查时 NCCL 先确认每个 rank 的 GPU 支持 fabric handle、fabric `State` 为 `Completed`、并且报告的 `ClusterUUID` 非零，再按 `ClusterUUID` 和 `CliqueId` 给 rank 分组，找出共享 NVLink 域的 rank，最后在每个进程内部做自测：分配一小块 fabric buffer，导出再导入回来（即上面的第 1、2、4 步）。自测能验证进程可以使用 fabric 内存，但因为数据不出进程，查不出只在跨节点时才出现的问题。全部通过时 init 日志里有一行 `MNNVL 1 cliqueId 7f cliqueSize 8 cliqueRank 5 nvlDomainSize 8`，其中 `cliqueSize` 和 `nvlDomainSize` 数的是这个 job 的 rank，不是机架里的 GPU，2 个 tray 的测试就显示 8；此后 NCCL 与同一 clique 的 rank 共享的 buffer（包括 NVLink SHARP（NVLS）的 multicast 对象）都用 fabric 内存。[详见](./drivers.md#how-nccl-uses-it)

**小心悄悄回退**：只要任何一个 rank 上第一步的检查失败（比如某一个 tray 的 fabric 状态不是 `Completed`），NCCL 就会对整个 job 悄悄不用 MNNVL，tray 之间的流量全部走网卡，没有任何报错；在默认值下，如果禁用了 P2P，MNNVL 也会被跳过。所以一定要找 `MNNVL 1` 这一行。只有这些检查都通过、自测却失败时，NCCL 才会报错退出，这时 `NCCL_MNNVL_ENABLE=0` 能让 job 先跑起来，但 compute tray 之间的流量就改走网卡了。

**验证与快照**：先用下面“常用命令”里的 MNNVL 检查命令，再实测：[nvbandwidth](https://github.com/NVIDIA/nvbandwidth) 有多节点测试（如 `multinode_device_to_device_memcpy_read_ce`），或者在 2 个 compute tray 上跑 [all_reduce_bench.py](../../network/benchmarks/all_reduce_bench.py) 并和单个 tray 比较，MNNVL 正常时跨 tray 不应该像跨服务器走网卡那样慢一个数量级。`cuda-checkpoint` 无法 checkpoint 用 `cuMemExportToShareableHandle` 导出的内存，而 fabric 内存正是这种内存，所以要对使用 MNNVL 的进程（如多节点训练进程）做 checkpoint，必须先释放所有 fabric 分配（通常是销毁 NCCL communicator），恢复后再重建，见 [Process checkpoint/restore](../kubernetes/snapshots.md#process-checkpointrestore)。

## 常用命令 / 配置
```bash
kubectl exec -i POD -- bash -s < driver-report.sh   # 在 pod 里跑完整报告，看容器实际拿到了什么
cat /proc/driver/nvidia/version                    # 已加载的内核 driver 版本（容器里也能看）
grep -E "^nvidia" /proc/modules | cut -d" " -f1    # 加载了哪些 nvidia 模块
nvidia-smi -q | grep -A2 Fabric          # 任何地方都能跑：期望 State: Completed, Status: Success
nvidia-smi --query-gpu=name,compute_cap --format=csv   # GPU 的 compute capability
python -c "import torch; print(torch.cuda.get_arch_list())"   # PyTorch 构建里包含哪些架构
kubectl get nodes -L nvidia.com/cuda.driver-version.full,nvidia.com/cuda.runtime-version.full   # 不登录 node 查看各 node 的 driver 版本
cat /sys/class/infiniband/*/fw_ver                # NIC 固件版本，总能用
# MNNVL（GB200/GB300 NVL72）检查
nvidia-smi -q | grep -A5 Fabric        # 每块 GPU：State Completed、Status Success、ClusterUUID 相同
nvidia-smi nvlink --status             # 所有 NVLink 都 active 且速率符合预期
systemctl status nvidia-imex           # 每个 compute tray 上（用 DRA driver 时看它的 daemon pod）
nvidia-imex-ctl -N                     # 整个 IMEX domain 的状态
ls -l /dev/nvidia-caps-imex-channels/  # 容器/pod 里也要能看到 channel
```

```yaml
# k8s 上为多节点 NVLink workload 创建 ComputeDomain，pod 再 claim imex-channel-0
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: my-compute-domain
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: imex-channel-0
```

## 常见坑
- `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` -> 内核模块没加载：内核升级后没重建模块、`nouveau` 被加载、或 Secure Boot 拒绝了未签名模块 -> 在宿主机上修复。
- `Failed to initialize NVML: Driver/library version mismatch` -> driver 升级后 node 没重启（宿主机和容器里都会出现），或镜像自带了 driver 库（只在容器里出现） -> 重启 node；或从镜像里删掉 driver 库（不要装 `nvidia-driver-*`、`libnvidia-compute-*` 这类包）。
- 容器里找不到 `nvidia-smi`，或 `torch.cuda.device_count()` 为 0 -> 没有申请/注入 GPU（没加 `--gpus`，pod 里没有 `nvidia.com/gpu`），或 `NVIDIA_DRIVER_CAPABILITIES` 缺少 `utility` -> 修改 docker 命令或 pod spec。
- GPU 显示正常但 CUDA 程序报 `cudaErrorSystemNotReady` / `system not yet initialized`（802） -> Fabric Manager 没运行或与 driver 版本不符 -> 在宿主机上修复。
- NCCL 报 `ibv_reg_mr` / `Cannot allocate memory` -> memlock 限制不是 `unlimited` -> 在宿主机或容器 runtime 上修改。
- NVL72 这类机架上多节点 job 很慢，NCCL 日志里没有 `MNNVL 1` 那一行 -> MNNVL 被悄悄跳过了（例如某块 GPU 的 fabric `State` 不是 `Completed`），tray 之间的流量走了网卡 -> 在每个 tray 上跑 `nvidia-smi -q` 检查。如果 NCCL 直接报 `MNNVL ... is available but not working on this system`：提示查 `/dev/nvidia-caps-imex-channels` 的是连 fabric 内存都分配不了，通常是容器里没有 IMEX channel；提示 `nvidia-imex-ctl -N` 的是本地导出/导入失败，通常是 IMEX daemon 有问题 -> 给容器传入 channel（`NVIDIA_IMEX_CHANNELS` 或 k8s 的 ComputeDomain）、检查 IMEX domain；`NCCL_MNNVL_ENABLE=0` 能让 job 先跑起来，但跨 tray 流量会走网卡。

## 相关章节
- [Containers](./README.md)、[driver-report.sh](./driver-report.sh)、arm64 镜像：[CPU architecture](./ci.md#cpu-architecture)、MNNVL 与快照：[Snapshots](../kubernetes/snapshots.md)
- [Fast Inter-node Networking](../kubernetes/network.md)（尤其是 [2. Does NCCL use them?](../kubernetes/network.md#2-does-nccl-use-them)）、[Overcoming job reset on CPU OOM event](../kubernetes/README.md#overcoming-job-reset-on-cpu-oom-event)
- [NVIDIA GPU debug: Running diagnostics](../../compute/accelerator/nvidia/debug.md#running-diagnostics)、[Xid Errors](../../compute/accelerator/nvidia/debug.md#xid-errors)、[Troubleshooting AMD GPUs](../../compute/accelerator/amd/debug.md)
- [Overcoming the coherent memory uncertain behavior](../../debug/pytorch.md#overcoming-the-coherent-memory-uncertain-behavior)、[AMD/ROCm hangs or slow with IOMMU](../../debug/pytorch.md#amdrocm-hangs-or-slow-with-iommu)、[Disable Access Control Services](../../network/benchmarks/README.md#disable-access-control-services)
- 外部：[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)、[GPU Operator](https://github.com/NVIDIA/gpu-operator)、[Network Operator](https://github.com/Mellanox/network-operator)、[DRA driver for GPUs](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu)、[ComputeDomain guide](https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/blob/main/site/content/docs/guides/compute-domain-workloads.md)、[CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
