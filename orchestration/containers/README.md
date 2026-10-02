# Containers

Whether you use [Kubernetes](../kubernetes/), [SLURM](../slurm/) or a single rented GPU box, sooner or later you will run your ML workloads inside containers. On k8s there is no other way, and on SLURM it's increasingly common because it frees you from depending on whatever software is installed on the nodes.

This chapter covers the ML-specific parts. For the general introduction see Docker's [Get started guide](https://docs.docker.com/get-started/).

## The basics

- **Image** - a read-only snapshot of a filesystem (OS libraries, python, CUDA libraries, your packages and code) plus metadata like the default command. Images are built in layers, each layer being the result of one instruction in the `Dockerfile`.
- **Container** - a running instance of an image. Its processes run on the host's kernel (it's not a VM), but see only the image's filesystem and whatever has been explicitly mounted or exposed to it. Anything a container writes to its own filesystem is lost when it's removed.
- **Registry** - a server that stores images, e.g., Docker Hub, NVIDIA NGC (`nvcr.io`), GitHub (`ghcr.io`), or your cloud's private registry (ECR, Artifact Registry, ACR). An image is referenced as `registry/repository:tag`, e.g., `nvcr.io/nvidia/pytorch:25.08-py3`.
- **Runtime** - the software that runs the containers: Docker and Podman on workstations, containerd on k8s nodes, and enroot/Apptainer on HPC (SLURM) clusters. All of them can use the same OCI (Docker) images.

## CUDA and the NVIDIA driver

This is the most important thing to understand about GPU containers: **the NVIDIA kernel driver comes from the host, everything else comes from the image.**

- The host has the kernel driver installed. The [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) mounts the matching driver user-space libraries (`libcuda.so`, `nvidia-smi`, etc.) and the `/dev/nvidia*` devices into the container at start time. That's why you should never install the NVIDIA driver inside an image.
- The image provides the CUDA runtime, cuDNN, NCCL and the frameworks built against them.

Therefore the CUDA version inside the image must be supported by the host's driver. Run `nvidia-smi` on the host - the `CUDA Version` in the top right corner is the newest CUDA version this driver supports. If the image's CUDA is newer than that, you will get errors like `CUDA driver version is insufficient for CUDA runtime version`, or `torch.cuda.is_available()` returning `False`. The fix is either an older image or a newer driver on the host. The full compatibility rules (including forward compatibility packages for datacenter GPUs) are in [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/), and the whole driver stack - which parts come from the host and which from the image, and how to check each of them - is explained in [Drivers: Kernel, Host and Container](./drivers.md).

A quick sanity check inside any new container:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

The same idea applies to AMD GPUs: the `amdgpu` kernel driver is on the host, ROCm user-space is in the image, and the devices are passed with `--device=/dev/kfd --device=/dev/dri`.

## Choosing a base image

You rarely start from scratch. The main options:

1. **[NVIDIA NGC PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)** (`nvcr.io/nvidia/pytorch:YY.MM-py3`) - released monthly, with PyTorch, CUDA, cuDNN, NCCL, Transformer Engine, Apex and many other libraries built and tested together. Great when you want something that just works on NVIDIA GPUs and good performance out of the box. The catch is that it's a large image, and it ships its own PyTorch build, so `pip install`ing a package that depends on a different `torch` version may replace it with a PyPI build - check `pip list | grep torch` after installing your requirements. Its [release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/) list the exact versions of each component and the minimal driver version required.
2. **[PyTorch official images](https://hub.docker.com/r/pytorch/pytorch)** (`pytorch/pytorch:<version>-cuda<X.Y>-cudnn<N>-runtime` or `-devel`) - smaller and closer to vanilla `pip install torch`.
3. **[NVIDIA CUDA images](https://hub.docker.com/r/nvidia/cuda)** (`nvidia/cuda:<version>-cudnn-devel-ubuntu24.04` etc.) + your own python env. Most control, most work.
4. **[ROCm images](https://hub.docker.com/u/rocm)** (`rocm/pytorch`, `rocm/vllm`, ...) for AMD GPUs.
5. **Inference server images** - e.g., `vllm/vllm-openai` or `lmsysorg/sglang` - ready to serve a model with a single command.

`runtime` vs `devel` image variants: the `devel` ones include the CUDA compiler `nvcc` and headers, which you need to compile CUDA extensions (e.g., building flash-attention from source). The `runtime` ones are smaller and enough if all you install are pre-built wheels.

Note that `pip install torch` wheels bundle their own CUDA runtime libraries (the `nvidia-*` pip packages), so you don't need a CUDA base image just to run PyTorch - a plain `python` or `ubuntu` image plus the NVIDIA Container Toolkit on the host is enough.

## Running GPU containers with Docker

Once the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on the host and docker is configured to use it (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`), check that the GPUs are visible:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

A typical command line for training:

```bash
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

What each of these does and why you want it:

- `--gpus all` - expose all GPUs; `--gpus '"device=0,1"'` for a subset (the quotes are needed).
- `--ipc=host` - use the host's shared memory. Without it `/dev/shm` is only 64MB, and PyTorch DataLoader workers (which pass tensors via shared memory) crash with `bus error` or `No space left on device`, and NCCL can fail too. Alternatively give the container its own bigger `/dev/shm` with `--shm-size=64g`. On k8s the equivalent is a [memory-backed emptyDir](../kubernetes/users.md#shared-memory).
- `--ulimit memlock=-1` - allow unlimited memory pinning, needed for RDMA (InfiniBand/RoCE) and pinned-memory transfers. `--ulimit stack=67108864` is the 64MB stack size NVIDIA recommends for its images.
- `--network=host` - use the host's network directly, which is the simplest way for multi-node training, as otherwise the container gets an isolated network and other nodes can't reach it. With the default bridge networking you also get a `docker0` interface that NCCL may try to use, see [network debug](../../network/debug/README.md#nccl-with-docker-containers).
- `-v host_path:container_path` - mount host directories. Your data, checkpoints and caches must live on mounts, otherwise they disappear with the container.
- `-e HF_TOKEN` - pass an env var from the host without writing its value on the command line.
- `--rm` - delete the container when it exits, so stopped containers don't accumulate.

For InfiniBand/RoCE you additionally need the RDMA devices and the right permissions: `--device=/dev/infiniband --cap-add=IPC_LOCK` (or `--privileged`, which gives access to everything on the host). The image also needs the RDMA user-space libraries (`libibverbs`, `librdmacm` - already included in the NGC images).

For debugging hanging processes with `py-spy` or `gdb` add `--cap-add=SYS_PTRACE`, see [diagnosing multi-gpu hanging](../../debug/pytorch.md#approaches-to-diagnosing-multi-gpu-hanging--deadlocks).

Other everyday commands:

```bash
docker ps                          # running containers
docker exec -it CONTAINER bash     # another shell in a running container
docker logs -f CONTAINER           # its stdout/stderr
docker images                      # local images
docker system df                   # how much disk images and containers use
docker system prune                # remove stopped containers and dangling images
```

### File ownership

By default processes in a container run as `root`, so files written to mounted host directories end up owned by root and you can't delete them as a normal user. Run as yourself instead:

```bash
docker run --user $(id -u):$(id -g) ...
```

Some programs then fail because `$HOME` isn't writable - set `-e HOME=/tmp` or mount a home directory.

## Building your own image

Here is a typical `Dockerfile` for a training project:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.08-py3

# system packages - rarely change, so they come first to be cached
RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux htop iotop \
    && rm -rf /var/lib/apt/lists/*

# python dependencies - change sometimes
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# heavy CUDA extensions - limit the parallel compilation jobs or the build may get OOM-killed
# RUN MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# your code - changes all the time, so it comes last
WORKDIR /workspace
COPY . /workspace

CMD ["bash"]
```

Build and push it (for anything beyond experiments let CI do this, see [Building Images in CI](./ci.md)):

```bash
docker build -t my-registry.example.com/my-team/train:2026-10-01 .
docker push my-registry.example.com/my-team/train:2026-10-01
```

Tips:

- **Order the layers from least to most frequently changing.** Docker re-builds everything after the first changed layer, so if `COPY . /workspace` came before `pip install` every code edit would re-install all the dependencies.
- **Use a `.dockerignore` file.** `docker build` sends the whole directory (the "build context") to the builder. Exclude `.git`, datasets, checkpoints, `__pycache__`, virtual envs, etc. - otherwise a build may try to upload hundreds of GBs.
- **Never put secrets in the image** (`ENV HF_TOKEN=...` or a copied `.netrc`) - anybody who can pull the image can read them. Pass them at run time.
- **Don't use the `latest` tag** for anything you run. It's a moving target, so you can't tell which version a job ran with, and different nodes may end up with different images. Use a date, a version or the git commit as the tag. For complete reproducibility you can refer to an image by its digest: `image@sha256:...`.
- **Build for the right CPU architecture.** If you build on an Apple Silicon Mac you get an `arm64` image by default, which fails on `x86_64` GPU nodes with `exec format error`. Use `docker buildx build --platform linux/amd64 ...`. Conversely, NVIDIA's Grace-based systems (GH200, GB200, GB300) are `arm64`, and need `linux/arm64` images.
- **Compiling CUDA extensions** requires a `devel` base image, and since there are no GPUs during `docker build` you need to tell the build which GPU architectures to compile for, e.g., `ENV TORCH_CUDA_ARCH_LIST="9.0;10.0"` for H100/H200 and B200.

### Keep the images small

ML images are typically 10-25GB. Every node has to pull the image before your job can start, so a 20GB image on 100 nodes means 2TB going through the registry, and the job may wait many minutes in `ContainerCreating`. Ways to keep things fast:

- Use `--no-cache-dir` with `pip` and remove the `apt` lists in the same `RUN` instruction that created them - a file deleted in a later layer still takes space in the earlier layer.
- Use multi-stage builds: compile in a `devel` stage and copy only the results into a `runtime` stage.
- Never put datasets or model weights into the image - mount them or download them at start time.
- Keep the rarely changing heavy layers at the bottom - nodes that already have them cached will download only the changed top layers.
- Ask the admins whether the cluster pre-pulls images onto nodes, or supports lazy image loading (e.g., GKE image streaming, or the [SOCI](https://github.com/awslabs/soci-snapshotter)/[stargz](https://github.com/containerd/stargz-snapshotter) snapshotters).

To see which layers are big: `docker history IMAGE`, or use [dive](https://github.com/wagoodman/dive) for an interactive view.

## Containers on SLURM

Docker requires a root daemon, so HPC clusters use rootless container runtimes instead. The two most common:

**[enroot](https://github.com/NVIDIA/enroot) + [pyxis](https://github.com/NVIDIA/pyxis)** - pyxis is a SLURM plugin that adds container flags to `srun`:

```bash
srun --container-image=nvcr.io#nvidia/pytorch:25.08-py3 \
     --container-mounts=/data:/data,$HOME:$HOME \
     python train.py
```

Note the `#` between the registry and the image name - that's enroot's syntax. Because importing a big image on every job is slow, you can import it once into a squashfs file and point to it instead:

```bash
enroot import -o /shared/images/pytorch-25.08.sqsh docker://nvcr.io#nvidia/pytorch:25.08-py3
srun --container-image=/shared/images/pytorch-25.08.sqsh ...
```

**[Apptainer](https://apptainer.org/)** (formerly Singularity):

```bash
apptainer pull pytorch.sif docker://nvcr.io/nvidia/pytorch:25.08-py3
srun apptainer exec --nv --bind /data:/data pytorch.sif python train.py
```

`--nv` makes the host's NVIDIA driver and GPUs available inside the container (`--rocm` for AMD).

Unlike Docker, both of these run the container as your own user (no root daemon) and use the host's network, which is why they fit HPC environments well. Apptainer also mounts your home directory by default.

## Where to go next

- [Drivers: Kernel, Host and Container](./drivers.md) - the GPU and network driver stack, its dependencies and how it relates to containers.
- [Building Images in CI](./ci.md) - building, testing, tagging and pushing the images automatically.
- [Kubernetes](../kubernetes/) - running these containers at scale.
- [NCCL with docker containers](../../network/debug/README.md#nccl-with-docker-containers) - network-specific container settings.
- [Dev Containers](https://containers.dev/) - develop inside the same container image you train with, directly from VSCode and other IDEs.
