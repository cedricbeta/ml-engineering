# Kubernetes 上的数据与模型加载：中文要点

> 英文原文：[Data and Model Loading on Kubernetes](./storage.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
在 SLURM 集群上，通常每个节点都挂着同一个共享文件系统，直接从 `/data` 读就行。k8s 上除非 pod spec 要求，否则什么都不会挂载，而且 pod 自己的文件系统会随 pod 一起消失。这一章讲数据集、模型权重和 checkpoint 应该放在哪里，以及如何快速把它们送到 GPU。文件系统本身的背景知识见 [Storage](../../storage/) 章节。

## 核心概念
- **StorageClass**：可以用来创建新卷的存储类型。
- **PVC 的 access mode**：`ReadWriteMany`（`RWX`）可以被所有 node 上的 pod 同时挂载，多节点训练需要它；`ReadWriteOnce`（`RWO`）只能被同一个 node 上的 pod 挂载。
- **`emptyDir`**：随 pod 创建和删除的临时卷，放在 node 的磁盘上（node 有 NVMe 的话可能在 NVMe 上）；加 `medium: Memory` 时放在内存里，用作 `/dev/shm`。
- **`hostPath`**：直接使用 node 上的目录，数据会留在 node 上，适合做 node 本地缓存，但常被集群策略禁止。
- **init container**：在主容器之前运行的容器，可以用来在 pod 启动时把数据拷到本地盘。
- **FUSE CSI driver**：把对象存储挂载成普通目录，如 Mountpoint for S3、Cloud Storage FUSE、BlobFuse。
- **`fsGroup`**：让 k8s 给你的用户组授予已挂载卷的写权限。

## 关键要点
- 存放位置怎么选：容器镜像只放代码和 python 包；容器自己的文件系统不放任何重要的东西；共享文件系统（NFS、Lustre、Weka、VAST、GPFS 等）上的 PVC 放数据集、权重和 checkpoint；块设备（EBS、Persistent Disk 等）上的 PVC 速度快，但读写只能单节点用（`ReadWriteOnce`），部分支持只读共享（`ReadOnlyMany`）；对象存储对大文件顺序读吞吐很高，对小文件和随机读很慢。[详见](./storage.md#the-options)
- 第一件事是查 access mode：`RWO` 卷被另一个 node 上的 pod 使用时，那个 pod 会卡在 `ContainerCreating`，并出现 `Multi-Attach error` 事件。依赖一个卷之前，先在 pod 里实测它的性能，同一个 storage class 的实际表现可能和标称差很多。[详见](./storage.md#finding-out-what-you-have)
- 镜像以非 root 用户运行时，往新建的卷里写可能遇到 `Permission denied`，设置 `fsGroup` 即可；文件数以百万计的卷每次挂载都改属主会非常慢，加 `fsGroupChangePolicy: OnRootMismatch` 只在需要时修改。[详见](./storage.md#permissions)
- 数据集按优先级：① 已预先放好数据的共享高速文件系统（RWX PVC），最接近 SLURM 的体验；② 用 init container 在 pod 启动时拷到 node 的 NVMe，适合数据集放得下本地盘、而共享存储又慢的情况（`emptyDir` 是否真的在 NVMe 上取决于 node 的配置，要问管理员；每次重启都要重新拷贝）；③ 用专门的 loader 从对象存储流式读取（[webdataset](https://github.com/webdataset/webdataset)、[MosaicML streaming](https://github.com/mosaicml/streaming)、HF `datasets` 的 `streaming=True`），前提是数据按大 shard 顺序读取，且重启后能从准确位置继续；④ 用 FUSE 挂载对象存储，看起来像普通目录，但随机访问和 `mmap` 很慢，只用于顺序读。[详见](./storage.md#datasets)
- 预处理后的数据集缓存（如 `HF_DATASETS_CACHE`）也要放在持久存储上，否则每次重启都要重新 tokenize。
- 模型权重：bf16 的 70B 模型约 140GB，64 个 pod 各自从 HF hub 下载就是 9TB 流量，还可能被限流，job 的启动时间取决于最慢的那次下载。做法是用一个单独的一次性 Job 下载到共享存储，训练/推理 pod 从本地路径加载，并设置 `HF_HUB_OFFLINE=1`，这样 hub 宕机或限流都影响不到你的 job。[详见](./storage.md#model-weights)
- 加载还是慢，通常是共享存储不适合 loader 的访问模式：可以先用 init container 把权重拷到 node 本地 NVMe（顺序拷贝往往比 loader 的读取快得多）；vLLM 可以用 `--load-format runai_streamer` 以高并发直接从对象存储读取；k8s 1.36+ 可以把 OCI 镜像挂载为只读卷（image volumes），权重打包成 OCI artifact 后能像容器镜像一样缓存在 node 上。不要把权重打进容器镜像。
- checkpoint：写到所有 rank 能并发写入的 RWX PVC，或直接写对象存储（如 PyTorch DCP 配 [s3torchconnector](https://github.com/awslabs/s3-connector-for-pytorch) 这类 S3 后端）；保存速度决定能多频繁地做 checkpoint，异步保存（如 `torch.distributed.checkpoint.async_save`）能把保存移出关键路径；快存储上保留最近几个，较旧的用单独的 job 或 CronJob 转移到对象存储；保存要原子化。[详见](./storage.md#checkpoints)
- 缓存：k8s 上 pod 频繁重启，每次都从空的文件系统开始，平时理所当然的各种缓存都要从头重建，每次重启可能多花好几分钟。把 `HF_HOME`、`HF_DATASETS_CACHE`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR` 指向持久存储。[详见](./storage.md#caches-that-make-restarts-faster)

## 常用命令 / 配置
```bash
kubectl get storageclass   # 可以用来创建新卷的存储类型
kubectl get pvc            # namespace 里已有的卷
kubectl describe pvc NAME  # 卷的容量、access mode 和背后的 storage class
```

```yaml
# 在支持 RWX 的 storage class 上新建一个共享卷
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-shared-pvc
spec:
  accessModes: ["ReadWriteMany"]
  storageClassName: shared-fs   # from `kubectl get storageclass`
  resources:
    requests:
      storage: 10Ti
```

```yaml
# 以非 root 运行时，让 k8s 给你的组授予挂载卷的写权限
spec:
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
```

```yaml
# 把重启时会被重建的缓存指向持久存储
env:
- name: HF_HOME                  # HF hub downloads
  value: /shared/cache/huggingface
- name: HF_DATASETS_CACHE        # preprocessed datasets
  value: /shared/cache/huggingface/datasets
- name: TRITON_CACHE_DIR         # compiled Triton kernels
  value: /shared/cache/triton
- name: TORCHINDUCTOR_CACHE_DIR  # torch.compile artifacts
  value: /shared/cache/inductor
```

## 常见坑
- 另一个 node 上的 pod 卡在 `ContainerCreating`，事件里有 `Multi-Attach error` -> 用的是 `ReadWriteOnce` 卷 -> 多节点训练改用 `ReadWriteMany` 卷。
- 往新卷写入时 `Permission denied` -> 镜像以非 root 用户运行 -> 设置 `fsGroup`；文件很多导致挂载很慢时再加 `fsGroupChangePolicy: OnRootMismatch`。
- 每次重启都重新下载、重新 tokenize、重新编译 kernel -> 缓存在 pod 的临时文件系统里 -> 把各类缓存目录指向持久存储。
- 多个 pod 并发写同一个缓存时出现奇怪的错误 -> 并发写冲突 -> 每个 node 用自己的子目录，或把编译缓存放在 node 的本地盘上。
- 数据集放在 FUSE 挂载的对象存储上，读得极慢 -> 随机访问和 `mmap` 在 FUSE 上很慢 -> 只用于顺序读，或换成预先放好数据的共享文件系统、本地 NVMe 副本。
- job 启动被 hub 限流或故障卡住 -> 每个 pod 都在启动时从 hub 下载权重 -> 一次性下载到共享存储，并设置 `HF_HUB_OFFLINE=1`。

## 相关章节
- [Storage](../../storage/)：[fio](../../storage/README.md#fio)、[usability perception benchmarks](../../storage/README.md#usability-perception-io-benchmarks)、[Local storage beats cloud storage](../../storage/README.md#local-storage-beats-cloud-storage)、[mmap vs sequential dataset reads](../../storage/README.md#mmap-vs-sequential-dataset-reads)、[Share caches in group environments](../../storage/README.md#share-caches-in-group-environments)
- [Kubernetes for Users](./users.md)（PVC、[Secrets](./users.md#secrets)）
- [Checkpoint and resume](./fault-tolerance.md#checkpoint-and-resume)、[Frequent checkpoint saving](../../training/fault-tolerance/README.md#frequent-checkpoint-saving)
- [Keep the images small](../containers/README.md#keep-the-images-small)
- 外部：[image volumes](https://kubernetes.io/docs/tasks/configure-pod-container/image-volumes/)、[Mountpoint for S3](https://github.com/awslabs/mountpoint-s3-csi-driver)、[Cloud Storage FUSE](https://github.com/GoogleCloudPlatform/gcs-fuse-csi-driver)、[BlobFuse](https://github.com/Azure/azure-storage-fuse)
