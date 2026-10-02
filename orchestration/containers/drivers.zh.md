# 驱动：内核、宿主机与容器：中文要点

> 英文原文：[Drivers: Kernel, Host and Container](./drivers.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
GPU 集群上很多令人困惑的故障，归根结底是驱动栈的各层彼此不匹配：“节点 A 上能跑、节点 B 上不行”，“宿主机上 `nvidia-smi` 正常、容器里不行”，“管理员升级了什么之后就坏了”。这一章讲清楚涉及哪些驱动、每个做什么、谁依赖谁、哪些来自宿主机、哪些来自容器镜像，以及如何逐一检查。想一次拿到完整报告，可以在宿主机或容器/pod 里运行 [driver-report.sh](./driver-report.sh)。

## 核心概念
- **kernel module（内核模块）**：内核驱动，如 `nvidia`、`nvidia_uvm`、`nvidia_peermem`，以及 RDMA 的 `ib_core`、`ib_uverbs`、`mlx5_core`、`mlx5_ib`、`efa` 等，永远来自宿主机。
- **driver 用户态库**：`libcuda.so`（CUDA driver API）、`libnvidia-ml.so`（NVML，`nvidia-smi`、DCGM、k8s device plugin 等都用它）、`libnvidia-ptxjitcompiler.so`（运行时把 PTX 编译成机器码）和 `nvidia-smi` 等二进制，必须与已加载的内核模块版本完全一致。
- **GSP firmware**：从 Turing 起，GPU 有一个 GPU System Processor，driver 把部分初始化和管理工作交给它。固件随 driver 包发布，必须来自同一 driver 版本。
- **宿主机服务**：`nvidia-persistenced`（让 driver 在无进程使用 GPU 时也保持初始化）、`nvidia-fabricmanager`（配置 NVSwitch 和 NVLink fabric）、`nvidia-imex`（GB200/GB300 NVL72 这类机架级系统上让不同节点的 GPU 通过 NVLink 共享内存）、DCGM（健康检查、诊断和指标）。
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
- **升级 driver 时要一起动**：内核模块和用户态库、GSP 固件、Fabric Manager（NVSwitch 系统）、CDI spec（若使用）要同步，并且重启（或完整重新加载模块）。先 `kubectl drain NODE --ignore-daemonsets` 再升级，全集群升到同一版本。AMD 的分工不同：宿主机上只有 `amdgpu` 内核模块，整个 ROCm 用户态都来自镜像，不注入任何东西。[详见](./drivers.md#upgrading-drivers)

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
```

## 常见坑
- `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` -> 内核模块没加载：内核升级后没重建模块、`nouveau` 被加载、或 Secure Boot 拒绝了未签名模块 -> 在宿主机上修复。
- `Failed to initialize NVML: Driver/library version mismatch` -> driver 升级后 node 没重启（宿主机和容器里都会出现），或镜像自带了 driver 库（只在容器里出现） -> 重启 node；或从镜像里删掉 driver 库（不要装 `nvidia-driver-*`、`libnvidia-compute-*` 这类包）。
- 容器里找不到 `nvidia-smi`，或 `torch.cuda.device_count()` 为 0 -> 没有申请/注入 GPU（没加 `--gpus`，pod 里没有 `nvidia.com/gpu`），或 `NVIDIA_DRIVER_CAPABILITIES` 缺少 `utility` -> 修改 docker 命令或 pod spec。
- GPU 显示正常但 CUDA 程序报 `cudaErrorSystemNotReady` / `system not yet initialized`（802） -> Fabric Manager 没运行或与 driver 版本不符 -> 在宿主机上修复。
- `nvidia-smi` 正常但报 `CUDA unknown error`（999） -> 常见原因是 `nvidia_uvm` 没加载或缺 `/dev/nvidia-uvm`，或 GPU 处于异常状态 -> 在宿主机上修复。
- NCCL 报 `ibv_reg_mr` / `Cannot allocate memory` -> memlock 限制不是 `unlimited` -> 在宿主机或容器 runtime 上修改。

## 相关章节
- [Containers](./README.md)、[driver-report.sh](./driver-report.sh)
- [Fast Inter-node Networking](../kubernetes/network.md)（尤其是 [2. Does NCCL use them?](../kubernetes/network.md#2-does-nccl-use-them)）、[Overcoming job reset on CPU OOM event](../kubernetes/README.md#overcoming-job-reset-on-cpu-oom-event)
- [NVIDIA GPU debug: Running diagnostics](../../compute/accelerator/nvidia/debug.md#running-diagnostics)、[Xid Errors](../../compute/accelerator/nvidia/debug.md#xid-errors)、[Troubleshooting AMD GPUs](../../compute/accelerator/amd/debug.md)
- [Overcoming the coherent memory uncertain behavior](../../debug/pytorch.md#overcoming-the-coherent-memory-uncertain-behavior)、[AMD/ROCm hangs or slow with IOMMU](../../debug/pytorch.md#amdrocm-hangs-or-slow-with-iommu)、[Disable Access Control Services](../../network/benchmarks/README.md#disable-access-control-services)
- 外部：[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)、[GPU Operator](https://github.com/NVIDIA/gpu-operator)、[Network Operator](https://github.com/Mellanox/network-operator)、[DRA driver for GPUs](https://github.com/NVIDIA/k8s-dra-driver-gpu)、[CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
