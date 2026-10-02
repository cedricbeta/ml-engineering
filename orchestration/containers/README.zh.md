# 容器：中文要点

> 英文原文：[Containers](./README.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
无论用 [Kubernetes](../kubernetes/)、[SLURM](../slurm/) 还是单台租来的 GPU 机器，迟早都会在容器里跑 ML 工作负载：k8s 上别无选择，SLURM 上也越来越常见，因为这样就不必依赖节点上装了什么软件。这一章只讲 ML 特有的部分：CUDA 与 NVIDIA driver 的关系、如何选基础镜像、用 Docker 跑 GPU 容器、构建自己的镜像，以及在 SLURM 上使用容器。通用入门见 Docker 的 [Get started guide](https://docs.docker.com/get-started/)。

## 核心概念
- **Image**：文件系统的只读快照（OS 库、python、CUDA 库、你的包和代码），加上默认命令等元数据。镜像按层构建，`Dockerfile` 里每条指令产生一层。
- **Container**：镜像的运行实例。它的进程跑在宿主机的内核上（不是 VM），但只能看到镜像的文件系统和明确挂载或暴露给它的东西；写到容器自身文件系统的内容在容器删除后就没了。
- **Registry**：存放镜像的服务器，如 Docker Hub、NVIDIA NGC（`nvcr.io`）、GitHub（`ghcr.io`）或云上的私有仓库（ECR、Artifact Registry、ACR）。镜像以 `registry/repository:tag` 的形式引用，如 `nvcr.io/nvidia/pytorch:25.08-py3`。
- **Runtime**：运行容器的软件：工作站上是 Docker 和 Podman，k8s node 上是 containerd，HPC（SLURM）集群上是 enroot/Apptainer。它们都能使用同样的 OCI（Docker）镜像。
- **NVIDIA Container Toolkit**：装在宿主机上，容器启动时把匹配的 driver 用户态库（`libcuda.so`、`nvidia-smi` 等）和 `/dev/nvidia*` 设备挂进容器。
- **`runtime` vs `devel` 镜像**：`devel` 包含 CUDA 编译器 `nvcc` 和头文件，编译 CUDA 扩展（如从源码构建 flash-attention）时需要；`runtime` 更小，只安装预编译 wheel 时就够了。

## 关键要点
- **最重要的一条：NVIDIA 内核 driver 来自宿主机，其余一切来自镜像。** 镜像提供 CUDA runtime、cuDNN、NCCL 和基于它们构建的框架。所以永远不要在镜像里安装 NVIDIA driver。[详见](./README.md#cuda-and-the-nvidia-driver)
- **镜像的 CUDA 版本必须被宿主机 driver 支持**：宿主机上 `nvidia-smi` 右上角的 `CUDA Version` 是这个 driver 支持的最新 CUDA 版本。镜像的 CUDA 更新时会报 `CUDA driver version is insufficient for CUDA runtime version`，或 `torch.cuda.is_available()` 返回 `False`；解决办法是换旧一点的镜像，或升级宿主机 driver。AMD 同理：`amdgpu` 内核 driver 在宿主机上，ROCm 用户态在镜像里。
- **选基础镜像**：NGC PyTorch（`nvcr.io/nvidia/pytorch:YY.MM-py3`）每月发布，各组件一起构建测试，开箱即用、性能好，但体积大，且自带 PyTorch 构建，`pip install` 一个依赖其他 `torch` 版本的包可能会把它替换成 PyPI 版本，装完依赖后要 `pip list | grep torch` 检查；PyTorch 官方镜像更小，更接近原生 `pip install torch`；NVIDIA CUDA 镜像加自建 python 环境最可控也最费事；AMD 用 ROCm 镜像；推理可直接用 `vllm/vllm-openai`、`lmsysorg/sglang` 等镜像。[详见](./README.md#choosing-a-base-image)
- `pip install torch` 的 wheel 自带 CUDA runtime 库（`nvidia-*` pip 包），所以只是跑 PyTorch 的话不需要 CUDA 基础镜像，普通 `python` 或 `ubuntu` 镜像加上宿主机的 NVIDIA Container Toolkit 就够了。
- **Docker 跑训练的常用参数**：`--gpus all` 暴露 GPU；`--ipc=host` 使用宿主机共享内存（否则 `/dev/shm` 只有 64MB，也可以用 `--shm-size=64g`）；`--ulimit memlock=-1` 允许无限制 pin 内存（RDMA 和 pinned memory 传输需要）；`--network=host` 是多节点训练最简单的方式；`-v` 挂载数据、checkpoint 和缓存；`-e HF_TOKEN` 传环境变量又不把值写在命令行上；`--rm` 退出后删除容器。IB/RoCE 还需要 `--device=/dev/infiniband --cap-add=IPC_LOCK`；用 `py-spy`/`gdb` 调试需要 `--cap-add=SYS_PTRACE`。[详见](./README.md#running-gpu-containers-with-docker)
- **构建镜像**：层按“很少变 -> 经常变”的顺序排列（系统包 -> python 依赖 -> 代码），因为某层一变，之后的层全部重建；用 `.dockerignore` 排除 `.git`、数据集、checkpoint、`__pycache__`、虚拟环境等，否则 build context 可能有几百 GB；不要把 secret 放进镜像；不要使用 `latest` tag，用日期、版本号或 git commit 做 tag，要完全可复现可以用 digest（`image@sha256:...`）。[详见](./README.md#building-your-own-image)
- **CPU 架构与 GPU 架构**：在 Apple Silicon Mac 上默认构建出 `arm64` 镜像，在 `x86_64` GPU node 上会报 `exec format error`，要用 `docker buildx build --platform linux/amd64 ...`；反过来，NVIDIA 基于 Grace 的系统（GH200、GB200、GB300）是 `arm64`。编译 CUDA 扩展需要 `devel` 镜像，并且因为 `docker build` 时没有 GPU，要用 `TORCH_CUDA_ARCH_LIST` 指定目标架构。
- **镜像要小**：ML 镜像通常有 10-25GB，每个 node 都要先拉完镜像 job 才能启动。`pip` 加 `--no-cache-dir`，在创建 `apt` 列表的同一个 `RUN` 里删除它们（后面的层删除文件，前面的层仍占空间）；用 multi-stage build；不要把数据集和权重放进镜像；重而少变的层放在底部；问管理员是否支持预拉取或懒加载（如 GKE image streaming、SOCI/stargz snapshotter）。[详见](./README.md#keep-the-images-small)
- **SLURM 上的容器**：Docker 需要 root daemon，所以 HPC 集群用 rootless runtime。enroot + pyxis 给 `srun` 加上容器参数，注意 registry 和镜像名之间用 `#` 分隔，可以先 import 成 squashfs 文件避免每个 job 都导入；Apptainer（前身 Singularity）用 `--nv` 让容器能用宿主机的 NVIDIA driver 和 GPU（AMD 用 `--rocm`）。两者都以你自己的用户运行、使用宿主机网络。[详见](./README.md#containers-on-slurm)

## 常用命令 / 配置
```bash
# 新容器里的快速检查
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
docker run --rm --gpus all ubuntu nvidia-smi   # 检查 Docker 能否看到 GPU
docker run --user $(id -u):$(id -g) ...        # 以自己的身份运行，避免写出 root 拥有的文件
docker history IMAGE                           # 查看哪些层比较大
```

```bash
# 典型的训练用 docker run
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network=host \
    -v /data:/data \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    nvcr.io/nvidia/pytorch:25.08-py3 \
    bash
```

```bash
# SLURM 上：enroot 先导入成 squashfs，再交给 pyxis 使用；或用 Apptainer
enroot import -o /shared/images/pytorch-25.08.sqsh docker://nvcr.io#nvidia/pytorch:25.08-py3
srun --container-image=/shared/images/pytorch-25.08.sqsh ...
apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:25.08-py3
srun apptainer exec --nv --bind /data:/data pytorch.sif python train.py
```

## 常见坑
- DataLoader worker 报 `bus error` 或 `No space left on device`，NCCL 也可能失败 -> `/dev/shm` 只有默认的 64MB -> 加 `--ipc=host` 或 `--shm-size=64g`；k8s 上用 [memory-backed emptyDir](../kubernetes/users.md#shared-memory)。
- 挂载目录里的文件属于 root，普通用户删不掉 -> 容器内进程默认以 `root` 运行 -> `docker run --user $(id -u):$(id -g)`；有程序因 `$HOME` 不可写而失败时，加 `-e HOME=/tmp` 或挂载一个 home 目录。
- 在 GPU node 上报 `exec format error` -> 在 Apple Silicon Mac 上构建出了 `arm64` 镜像 -> `docker buildx build --platform linux/amd64 ...`。
- 每改一行代码都要重装所有依赖 -> `COPY . /workspace` 写在了 `pip install` 之前 -> 按变化频率排列层，代码放最后。
- `docker build` 要上传几百 GB -> build context 包含了 `.git`、数据集、checkpoint 等 -> 写 `.dockerignore`。
- 装完依赖后 NGC 镜像里的 PyTorch 变了 -> 某个包依赖不同的 `torch` 版本，被替换成了 PyPI 构建 -> 装完后 `pip list | grep torch` 检查。

## 相关章节
- [Drivers: Kernel, Host and Container](./drivers.md)：GPU 和网络的驱动栈、依赖关系及其与容器的关系。
- [Building Images in CI](./ci.md)：自动构建、测试、打 tag 和推送镜像。
- [Kubernetes](../kubernetes/)：大规模运行这些容器；[SLURM](../slurm/)。
- [NCCL with docker containers](../../network/debug/README.md#nccl-with-docker-containers)：容器相关的网络设置。
- [diagnosing multi-gpu hanging](../../debug/pytorch.md#approaches-to-diagnosing-multi-gpu-hanging--deadlocks)
- 外部：[CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)、[NGC PyTorch release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)、[dive](https://github.com/wagoodman/dive)、[Dev Containers](https://containers.dev/)
