# Drivers: Kernel, Host and Container

Many of the confusing failures on GPU clusters come down to the layers of the driver stack not matching each other: "it works on node A but not on node B", "`nvidia-smi` works on the host but not in the container", "it worked until the admins upgraded something". This chapter explains which drivers are involved, what each one does, what depends on what, which parts come from the host and which from your container image, and how to check each of them.

To get a full report of the stack in one go, run [driver-report.sh](./driver-report.sh) - on the host, or inside a container/pod to see what the container actually got:

```bash
bash driver-report.sh
kubectl exec -i POD -- bash -s < driver-report.sh
```

## The one rule: containers share the host's kernel

A container is not a VM - its processes run on the host's kernel. Run `uname -r` inside any container and you get the host's kernel version, regardless of the image's OS. This has important consequences:

- **Kernel drivers (kernel modules) always come from the host.** You can't bring your own NVIDIA kernel driver in an image, and installing driver packages in the image doesn't load anything into the kernel.
- **A container only sees the devices passed to it.** The GPUs (`/dev/nvidia*`) and the RDMA NICs (`/dev/infiniband/*`) must be explicitly exposed by the container runtime - on k8s via device plugins.
- **User-space libraries come from the image**, with one important exception: the NVIDIA driver's own user-space libraries (`libcuda.so`, `libnvidia-ml.so`), which must exactly match the host's kernel driver and are therefore injected from the host at container start.
- **Kernel settings are the host's**: module parameters, IOMMU, huge pages, cgroups, etc. You can read most of them from inside a container, but changing them is a job for the node's admin.

The one exception to "kernel modules come from the host" are driver containers, like those of the [GPU Operator](#the-gpu-operator) and the Network Operator: they are privileged containers that compile and load the kernel modules into the host's kernel - but the result is still a host-level driver that all containers on that node share.

## The layers

From top to bottom, for an NVIDIA GPU node with an InfiniBand/RoCE network:

```
  ┌─ from the container image ──────────────────────────────────────────────────────┐
  │ your code, PyTorch                                                              │
  │ NCCL, cuDNN, cuBLAS, CUDA runtime (libcudart)  - often bundled in the torch wheel│
  │ NCCL network plugins (aws-ofi-nccl, ...), libfabric                             │
  │ rdma-core: libibverbs, librdmacm, the libmlx5/libefa providers                  │
  ├─ injected from the host by the NVIDIA Container Toolkit ────────────────────────┤
  │ libcuda.so, libnvidia-ml.so, libnvidia-ptxjitcompiler.so, nvidia-smi            │
  │ device files: /dev/nvidia0..7, /dev/nvidiactl, /dev/nvidia-uvm                  │
  ├─ passed in by the RDMA device plugin ───────────────────────────────────────────┤
  │ device files: /dev/infiniband/uverbs0..N, /dev/infiniband/rdma_cm               │
  └─────────────────────────────────────────────────────────────────────────────────┘
  ┌─ host only ─────────────────────────────────────────────────────────────────────┐
  │ services: nvidia-persistenced, nvidia-fabricmanager, nvidia-imex, DCGM          │
  │ kernel modules: nvidia, nvidia_uvm, nvidia_peermem, gdrdrv                      │
  │                 ib_core, ib_uverbs, rdma_ucm, mlx5_core, mlx5_ib (or efa)       │
  │ firmware: GPU VBIOS + GSP firmware, NIC firmware, NVSwitch firmware             │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

The rest of this chapter goes through these bottom up.

## The NVIDIA GPU driver

### Kernel modules

| Module | What it does | Without it |
| :----- | :----------- | :--------- |
| `nvidia` | the core driver - talks to the GPU hardware, manages its memory, creates `/dev/nvidia*` | no GPUs at all |
| `nvidia_uvm` | Unified Virtual Memory - required by CUDA, creates `/dev/nvidia-uvm` | CUDA fails to initialize, while `nvidia-smi` still works |
| `nvidia_modeset`, `nvidia_drm` | display and graphics | nothing for compute, often loaded anyway |
| `nvidia_peermem` | GPUDirect RDMA via the legacy peer memory API, see [GPUDirect RDMA](#gpudirect-rdma) | NCCL can't use GPUDirect RDMA unless DMA-BUF is available |
| `gdrdrv` | [GDRCopy](https://github.com/NVIDIA/gdrcopy) - low latency CPU access to GPU memory, used by NVSHMEM/UCX-based communication libraries | only needed if such a library asks for it |

Check:

```bash
cat /proc/driver/nvidia/version                    # the loaded kernel driver's version
grep -E "^nvidia" /proc/modules | cut -d" " -f1    # which nvidia modules are loaded
ls -l /dev/nvidia*                                  # the device files
```

All of these work inside a container too - `/proc/driver/nvidia` and `/proc/modules` describe the host's kernel.

#### Open vs proprietary kernel modules

NVIDIA ships 2 flavors of the kernel modules: the original proprietary ones and the [open-source GPU kernel modules](https://github.com/NVIDIA/open-gpu-kernel-modules). Since the R560 driver release the open ones are the default. Grace Hopper and Blackwell are supported only by the open modules, while Maxwell, Pascal and Volta are supported only by the proprietary ones. For Turing, Ampere, Ada and Hopper both work, and NVIDIA recommends the open ones. Some features require the open modules, e.g., GPUDirect RDMA via DMA-BUF.

To tell which one is loaded, look at the first line of `/proc/driver/nvidia/version` - the open modules say `NVIDIA UNIX Open Kernel Module`.

#### GSP firmware

Since Turing, GPUs have a GPU System Processor (GSP), to which the driver offloads part of the GPU initialization and management. Its firmware is shipped as part of the driver package (under `/lib/firmware/nvidia/<driver-version>/`) and must come from the same driver version. `nvidia-smi -q | grep "GSP Firmware"` shows its version.

#### Module parameters

The kernel driver's behavior can be changed with module parameters, set in `/etc/modprobe.d/*.conf` and applied on module load (typically a reboot). To see the current values:

```bash
cat /proc/driver/nvidia/params
```

An example where this matters is the coherent memory mode on Grace-based systems, see [Overcoming the coherent memory uncertain behavior](../../debug/pytorch.md#overcoming-the-coherent-memory-uncertain-behavior).

### Driver user-space libraries

Besides the kernel modules, the driver package installs user-space libraries:

- `libcuda.so` - the CUDA driver API. Everything that uses CUDA (the CUDA runtime, PyTorch) goes through it.
- `libnvidia-ml.so` - NVML, the management library used by `nvidia-smi`, DCGM, `pynvml`, the k8s device plugin, etc.
- `libnvidia-ptxjitcompiler.so` - compiles PTX code to the GPU's machine code at run time.
- binaries like `nvidia-smi`.

**These must be exactly the same version as the loaded kernel module.** The classic symptom of a mismatch is:

```
$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch
```

It typically happens right after a driver upgrade: the package manager replaced the libraries on disk, but the old kernel module is still loaded. The fix is to reboot (or unload and reload the modules, which requires stopping everything that uses the GPUs). Inside a container the same error has 2 possible causes: the same host-side problem (the injected libraries are the new ones on the host's disk, while the old kernel module is still loaded), or the image contains its own copy of the driver libraries - in which case remove them from the image. If `nvidia-smi` on the host works, it's the latter.

To compare the versions - the library file names carry the full version:

```bash
head -1 /proc/driver/nvidia/version
readlink -f $(ldconfig -p | awk '$1=="libcuda.so.1" {print $NF; exit}')
```

### Host services

| Service | What it does | Needed on |
| :------ | :----------- | :-------- |
| `nvidia-persistenced` | keeps the driver initialized even when no process uses the GPUs. Without it, every new process pays several seconds of driver initialization, and some settings don't stick | all compute nodes |
| `nvidia-fabricmanager` | configures the NVSwitches and the NVLink fabric between the GPUs | NVSwitch-based systems: HGX/DGX A100, H100, H200, B200, B300 |
| `nvidia-imex` | lets GPUs on different nodes share memory over NVLink | rack-scale multi-node NVLink systems (GB200/GB300 NVL72) |
| DCGM (`nv-hostengine`) | GPU health checks, diagnostics and metrics | everywhere you want monitoring, see [Running diagnostics](../../compute/accelerator/nvidia/debug.md#running-diagnostics) |

The Fabric Manager one is the most common source of trouble: if it's not running, or it's a different version than the driver (it checks on start and refuses to run if they don't fit), the GPUs show up fine in `nvidia-smi` but every CUDA program fails with `cudaErrorSystemNotReady` (error 802, "system not yet initialized"). Check:

```bash
systemctl status nvidia-fabricmanager    # on the host
nvidia-smi -q | grep -A2 Fabric          # anywhere - wants: State: Completed, Status: Success
```

On rack-scale NVL72 systems the NVLink domain spans many nodes and is managed from the NVLink switch trays, and the compute nodes run the IMEX service. On k8s this is orchestrated by NVIDIA's [DRA driver for GPUs](https://github.com/NVIDIA/k8s-dra-driver-gpu) via ComputeDomains.

## The three CUDA versions

There are 3 different things called "CUDA version", which is the source of endless confusion:

| What | Where it comes from | How to see it |
| :--- | :------------------ | :------------ |
| the newest CUDA version the driver supports | the host's driver | `nvidia-smi` - `CUDA Version` in the top right corner |
| the CUDA runtime your program uses | the image - for PyTorch usually bundled in the pip wheel (the `nvidia-*` packages) | `python -c "import torch; print(torch.version.cuda)"` |
| the CUDA toolkit (`nvcc`, headers) | the image, only in `devel` images | `nvcc --version` |

The rules:

1. **The driver must support the CUDA runtime** - the 1st number must be >= the 2nd one. If it's not, you get `CUDA driver version is insufficient for CUDA runtime version` or `torch.cuda.is_available()` returns `False`. Within the same major version there is [minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html): any CUDA 12.x runtime works with a driver >= 525, and any CUDA 13.x runtime with a driver >= 580 - with some limitations, most notably JIT-compiling PTX code that is newer than the driver doesn't work.
2. **A newer driver always works with an older CUDA runtime.** So upgrading the driver never requires rebuilding the images, while a too new image does require upgrading the driver.
3. **On datacenter GPUs, [forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)** lets a newer CUDA run on an older driver: the `cuda-compat` package provides a newer `libcuda.so` that works with the older kernel module, placed in `/usr/local/cuda-X.Y/compat/` and used via `LD_LIBRARY_PATH`. Some images, including NGC's, ship it and enable it at start when needed.
4. **The `nvcc` version only matters when compiling** - it must be compatible with the CUDA version PyTorch was built with when you build PyTorch extensions.

And there is a 4th compatibility dimension which has nothing to do with the driver: **the GPU architecture**. Each GPU has a compute capability (`sm_90` for H100/H200, `sm_100` for B200/GB200, etc.), and the CUDA kernels in your PyTorch build (or flash-attention, etc.) must have been compiled for it. Otherwise you get `no kernel image is available for execution on the device`. Compare:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
python -c "import torch; print(torch.cuda.get_arch_list())"
```

A binary compiled for `sm_80` also runs on `sm_86` (same major version, higher minor), but not on `sm_90` - a different major version needs either its own binary or PTX code to JIT-compile from. New GPU generations need new enough CUDA (e.g., Blackwell needs CUDA 12.8+) and new enough builds of all the libraries with CUDA kernels.

## How GPUs get into a container

The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) is installed on the host and hooks into the container runtime (Docker, containerd, CRI-O). When a container asks for GPUs, it:

1. passes in the device files of the requested GPUs plus `/dev/nvidiactl` and `/dev/nvidia-uvm`
2. mounts the host's driver user-space libraries (`libcuda.so`, `libnvidia-ml.so`, ...) and binaries (`nvidia-smi`) into the container
3. updates the container's library cache so that they are found

What gets injected is controlled by 2 env vars, which you will find set in the NVIDIA images:

- `NVIDIA_VISIBLE_DEVICES` - which GPUs: `all`, `none`, or a list of indices/UUIDs.
- `NVIDIA_DRIVER_CAPABILITIES` - which driver libraries: `compute` (CUDA), `utility` (`nvidia-smi` and NVML), `video`, `graphics`, etc. The default when unset is `utility,compute`. If `nvidia-smi` is missing in a container that otherwise has GPUs, this var lacks `utility`.

Newer toolkit versions describe the devices using the [Container Device Interface](https://github.com/cncf-tags/container-device-interface) (CDI) - a spec file (e.g., `/etc/cdi/nvidia.yaml`, generated with `nvidia-ctk cdi generate`) listing the exact device files and libraries to inject. If the spec isn't regenerated after a driver upgrade, it points to the old library files and containers fail to start - newer versions of the toolkit regenerate it automatically.

With Docker you ask for the GPUs with `docker run --gpus all` (or `--device nvidia.com/gpu=all` with CDI). On k8s the flow is:

```
pod requests nvidia.com/gpu: 8
  -> the scheduler picks a node that the device plugin reported as having 8 free GPUs
  -> the device plugin tells the kubelet which 8 GPUs to give to the container
  -> the container runtime + NVIDIA Container Toolkit inject those GPUs and the driver libraries
```

Gotchas:

- **Never install the NVIDIA driver in the image** (packages like `nvidia-driver-*`, `libnvidia-compute-*`). The injected libraries must match the host's kernel module, and a copy in the image can shadow them and cause the version mismatch error above. The CUDA toolkit and runtime are fine - they are not the driver.
- **A pod that didn't request GPUs may still see them.** NVIDIA's CUDA base images set `NVIDIA_VISIBLE_DEVICES=all`, and depending on how the cluster is configured, the toolkit may honor it even for a pod that didn't request any GPUs - giving it access to GPUs allocated to someone else. NVIDIA recommends configuring the toolkit to ignore this variable in unprivileged containers and the device plugin to pass the devices as volume mounts instead (`ACCEPT_NVIDIA_VISIBLE_DEVICES_ENVVAR_WHEN_UNPRIVILEGED=false` and `DEVICE_LIST_STRATEGY=volume-mounts`). If you see a GPU-less pod using GPUs, tell your admins.
- **Privileged containers see all the devices** of the node, including all GPUs, regardless of what they requested.
- `CUDA_VISIBLE_DEVICES` is a different thing: it's read by the CUDA runtime inside your process and only hides GPUs that the container already has. It's what launchers use to give each process its own GPU.

### The GPU Operator

On k8s the [GPU Operator](https://github.com/NVIDIA/gpu-operator) manages the whole host side as pods on each GPU node: the driver (as a driver container that compiles and loads the kernel modules), the Container Toolkit, the device plugin, DCGM and DCGM exporter, the MIG manager, and optionally `nvidia-peermem` and GDRCopy. Some node images come with the driver pre-installed, in which case the operator is configured not to install it (`driver.enabled=false`). The kernel module flavor is selected with `driver.kernelModuleType` (`auto`, `open` or `proprietary`).

To see the versions on a k8s cluster without logging into the nodes:

```bash
kubectl get pods -n gpu-operator -o wide            # the namespace may differ
kubectl get nodes -L nvidia.com/cuda.driver-version.full,nvidia.com/cuda.runtime-version.full
```

(the node labels are set by GPU Feature Discovery - and they are a quick way to find nodes that ended up with a different driver version than the rest.)

## RDMA network drivers

### Kernel modules

| Module | What it does |
| :----- | :----------- |
| `ib_core` | the core of the kernel's RDMA subsystem, used by all RDMA drivers |
| `ib_uverbs` | gives user-space direct access to the NIC (verbs) - creates `/dev/infiniband/uverbsN`. This is what NCCL uses |
| `rdma_ucm` | RDMA connection manager for user-space - creates `/dev/infiniband/rdma_cm` |
| `ib_umad` | management datagrams, used by the subnet manager tools (`ibstat`, `ibnetdiscover`, ...) - creates `/dev/infiniband/umadN` |
| `mlx5_core` | the NVIDIA (Mellanox) ConnectX NIC driver - Ethernet and the shared core |
| `mlx5_ib` | the RDMA (InfiniBand/RoCE) part of the ConnectX driver |
| `efa` | the AWS Elastic Fabric Adapter driver |

These come either with the Linux kernel itself (the "inbox" drivers) or from NVIDIA's packaged driver stack - MLNX_OFED, now succeeded by DOCA-OFED - which has newer drivers and tools, and is required for some features (e.g., the legacy `nvidia-peermem`). On k8s the [Network Operator](https://github.com/Mellanox/network-operator) can install it as a driver container.

The NIC firmware is a separate layer with its own version - different firmware versions on different nodes can result in inconsistent performance. Check:

```bash
ibv_devinfo | grep -E "hca_id|fw_ver"            # if ibverbs-utils is installed
cat /sys/class/infiniband/*/fw_ver                # always works
ethtool -i <interface>                            # driver and firmware of an Ethernet interface
```

### User-space

Unlike the GPU driver, the RDMA user-space comes **from the image** and isn't injected: `rdma-core` (`libibverbs`, `librdmacm` and the device-specific providers like `libmlx5` and `libefa`). The kernel-user interface of verbs is stable, so the image's `rdma-core` generally works with whatever kernel driver version the host has. NGC images already include it.

On top of it, NCCL has the IB transport built-in, while some fabrics need a NCCL network plugin in the image - e.g., on AWS EFA [aws-ofi-nccl](https://github.com/aws/aws-ofi-nccl) with `libfabric`.

### GPUDirect RDMA

GPUDirect RDMA lets the NIC read and write GPU memory directly, instead of staging the data through CPU memory. It requires a kernel-level bridge between the GPU driver and the NIC driver, of which there are 2:

- **DMA-BUF** - the standard Linux kernel mechanism, which NVIDIA recommends. Requires the open GPU kernel modules, Linux kernel 5.12+, CUDA 11.7+, and a Turing or newer datacenter/RTX GPU. Works with the inbox NIC drivers.
- **`nvidia_peermem`** - the legacy kernel module. Works with any driver version, but requires MLNX_OFED/DOCA-OFED.

You can tell it's working from NCCL's logs - the channels say `via NET/IB/.../GDRDMA`, see [Fast Inter-node Networking](../kubernetes/network.md#2-does-nccl-use-them).

For how the RDMA devices get into a pod, and how to verify the whole path, see [Fast Inter-node Networking](../kubernetes/network.md).

## AMD GPUs

The split is different from NVIDIA's: only the `amdgpu` kernel module is on the host (either the kernel's inbox version or the newer `amdgpu-dkms` from ROCm), and the entire ROCm user-space - including the parts that would correspond to `libcuda.so` - comes from the image. Nothing is injected; the container just needs the devices `/dev/kfd` and `/dev/dri/renderD*` (`--device=/dev/kfd --device=/dev/dri` with Docker, the [AMD GPU Operator](https://github.com/ROCm/gpu-operator) on k8s). The compatibility question is therefore between the host's `amdgpu` driver version and the image's ROCm version - check the ROCm compatibility matrix. See also [Troubleshooting AMD GPUs](../../compute/accelerator/amd/debug.md).

## Kernel-level things that matter

- **The driver must be built for the running kernel.** The NVIDIA kernel modules are compiled for a specific kernel version, either via DKMS (recompiled automatically on kernel upgrades - `dkms status` shows the state) or as pre-compiled packages. If the kernel gets upgraded and the module isn't rebuilt, after the reboot there is no driver and you get: `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver`. This is why GPU nodes often have kernel upgrades pinned.
- **`nouveau`**, the open-source community driver for NVIDIA GPUs, conflicts with NVIDIA's driver and must be blacklisted. **Secure Boot** rejects unsigned kernel modules, so the modules must be signed or Secure Boot disabled.
- **IOMMU and PCIe ACS settings** affect GPU-to-GPU and GPU-to-NIC performance, see [Disable Access Control Services](../../network/benchmarks/README.md#disable-access-control-services) and [AMD/ROCm hangs or slow with IOMMU](../../debug/pytorch.md#amdrocm-hangs-or-slow-with-iommu).
- **Memory locking** (`ulimit -l` must be `unlimited` for RDMA) and **huge pages** (needed by EFA) are set on the host and inherited by the containers.
- **cgroups v2** decides how OOM kills behave in containers, see [Overcoming job reset on CPU OOM event](../kubernetes/README.md#overcoming-job-reset-on-cpu-oom-event).

## Upgrading drivers

What must move together on an NVIDIA GPU node:

1. the kernel modules and the driver user-space libraries - always the same version
2. the GSP firmware - it comes with the driver package
3. Fabric Manager (on NVSwitch systems) - the same version as the driver
4. the CDI spec, if used - regenerated
5. a reboot (or a full module reload) - until then the old kernel module keeps running

So upgrade a node only after draining it (`kubectl drain NODE --ignore-daemonsets`), and upgrade all nodes of a cluster to the same version - a multi-node job running on mixed driver versions can behave inconsistently. On k8s the GPU Operator can do rolling driver upgrades node by node.

The container images don't need to be rebuilt after a driver upgrade (rule 2 of [the three CUDA versions](#the-three-cuda-versions)), but when you move the images to a newer CUDA, check that the driver supports it first.

## Symptom to cause

| Symptom | Likely cause | Where to fix |
| :------ | :----------- | :----------- |
| `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver` | the kernel module isn't loaded - a kernel upgrade without rebuilding the module, `nouveau` loaded, Secure Boot | host |
| `Failed to initialize NVML: Driver/library version mismatch` | driver upgraded but the node not rebooted (shows on the host and in containers), or the image contains its own driver libraries (shows only in containers) | host / image |
| `nvidia-smi` not found, or `torch.cuda.device_count()` is 0 in a container | no GPUs requested or injected (no `--gpus`, no `nvidia.com/gpu` in the pod), or `NVIDIA_DRIVER_CAPABILITIES` lacks `utility` | docker command / pod spec |
| `CUDA driver version is insufficient for CUDA runtime version` | the image's CUDA is newer than the driver supports | use an older image or upgrade the driver |
| `cudaErrorSystemNotReady` / `system not yet initialized` (802) | Fabric Manager is not running or doesn't match the driver | host |
| `no kernel image is available for execution on the device` | the PyTorch (or extension) build doesn't include your GPU's architecture | image |
| `CUDA unknown error` (999) while `nvidia-smi` works | often `nvidia_uvm` isn't loaded or `/dev/nvidia-uvm` is missing, or the GPU is in a bad state | host |
| NCCL logs `NET/Socket` instead of `NET/IB` | no RDMA devices in the container | pod spec, see [network](../kubernetes/network.md) |
| NCCL fails with `ibv_reg_mr` / `Cannot allocate memory` errors | the memlock limit isn't `unlimited` | host / container runtime |
| NCCL works, but no `GDRDMA` in its logs | neither DMA-BUF nor `nvidia_peermem` is available | host |
| GPU-related errors in `dmesg`, e.g., `Xid` | hardware or driver problems | see [Xid Errors](../../compute/accelerator/nvidia/debug.md#xid-errors) |
