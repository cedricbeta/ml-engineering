# Kubernetes 用户实用指南：中文要点

> 英文原文：[Kubernetes for Users](./users.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
写给已经拿到 k8s 集群访问权限、要在上面做训练或推理的 ML 工程师。它不是 k8s 手册，只讲日常需要的那一部分：连接集群、找到 GPU、核心工作流、交互式开发、ML pod 必备的 spec 设置、多节点训练，以及常见故障的诊断和清理。术语不熟的话先读 [Key concepts for newcomers](./README.md#key-concepts-for-newcomers)。

## 核心概念
- **kubeconfig / context**：管理员给的连接配置文件（或生成它的云 CLI 命令）。`kubectl` 默认读 `~/.kube/config`，或 `KUBECONFIG` 指向的文件；一个 context 对应一个可访问的集群。
- **`requests` / `limits`**：`requests` 是调度器为 pod 预留的资源，`limits` 是硬上限。
- **`/dev/shm`**：共享内存。PyTorch DataLoader worker 通过它传递 tensor，NCCL 也会用。
- **Access mode**：`ReadWriteMany` 的 PVC 可以被多个 pod 同时挂载（多节点训练需要），`ReadWriteOnce` 只能在单个 node 上使用。
- **Secret**：存放 token 等凭据，以环境变量的形式注入 pod。
- **Capability**：容器默认没有的权限，例如调试需要的 `SYS_PTRACE`、RDMA pin 内存需要的 `IPC_LOCK`。
- **Kueue**：很多集群使用的排队系统，通过 job 上的 label（如 `kueue.x-k8s.io/queue-name: my-team-queue`）提交到队列，用 `kubectl get workloads` 查看队列。
- **Gang scheduling**：多节点的 pod 必须一起被调度，否则先调度上的 pod 会占着 GPU 干等其余的 pod。由 Kueue 或 Volcano 解决，通常由管理员配置。

## 关键要点
- 设好默认 namespace，就不必每条命令都加 `-n my-team`（下文示例都假设已设置）；`kubectl auth can-i` 检查权限。很多人会加 `alias k=kubectl` 并开启 shell 补全。[详见](./users.md#getting-connected)
- 出问题时先 `kubectl describe` 对应对象，读最底部的 `Events:`，大多数时候它会告诉你哪里错了；想知道某个对象有哪些字段，用 `kubectl explain` 而不是上网搜。[详见](./users.md#the-core-workflow)
- 交互式开发 = 一个运行 `sleep infinity` 的 pod，再 `exec` 进去，相当于 SLURM 的 `salloc` + `srun --pty bash`。闲置的 dev pod 仍然占着 GPU，用完要删除；`exec` 会话断开时其中启动的进程也会一起结束，长时间运行的任务要用 `tmux` 或 `nohup`。[详见](./users.md#interactive-development)
- GPU 必须写在 `limits` 里（`requests` 默认与之相同；如果显式设置 `requests`，两者必须相等），且必须是整数。要整台 node 就申请它的全部 GPU，但 CPU 和内存仍要显式申请，因为默认 requests 可能很小。CPU `limit` 通过 throttling 实现，会严重拖慢 DataLoader worker 和 tokenization，所以很多 ML 集群只设 CPU `requests`。[详见](./users.md#gpus-cpus-and-memory)
- Docker 和 k8s 默认只给容器 64MB 的 `/dev/shm`，要挂载一个 `medium: Memory` 的 emptyDir；注意写进去的内容计入容器的内存 limit。[详见](./users.md#shared-memory)
- 容器文件系统随 pod 消失，数据、checkpoint、日志必须写到挂载的卷或云存储；多节点训练要用 `ReadWriteMany` 的 PVC。[详见](./users.md#persistent-storage)
- 不要把 token（HF、W&B、S3 key）放进镜像或提交到 git 的 YAML，存成 Secret 再用 `envFrom` 注入。[详见](./users.md#secrets)
- 多节点训练有 3 个层次：① **Indexed Job + headless Service**，只用内置对象，每个 pod 的 `$JOB_COMPLETION_INDEX` 作为 `torchrun --node-rank`，rank 0 有稳定 DNS 名作为 `--master-addr`，缺点是一个 pod 失败时没有人重启其他 pod；② **JobSet**，处理 DNS/headless service，任一 pod 失败时可以整组重启，和 Kueue 配合良好；③ **Kubeflow Trainer**（前身是 Kubeflow Training Operator 的 `PyTorchJob`），在此之上设置 `torchrun` 的分布式环境变量，并集成 DeepSpeed、MPI 等。单节点放不下的模型做推理用 LeaderWorkerSet，见 [Inference on Kubernetes](./inference.md)。[详见](./users.md#multi-node-training)
- 退出码：`137` = 128 + 9（`SIGKILL`），通常是 OOM kill；`143` = 128 + 15（`SIGTERM`），表示 pod 被要求终止（被删除、被抢占或 node 被 drain）。[详见](./users.md#diagnosing-typical-problems)
- 结束的 Job 和它的 pod 会一直保留（方便看日志），直到被删除；在 Job 的 `spec` 里加 `ttlSecondsAfterFinished: 86400` 可以让它结束一天后自动删除。[详见](./users.md#cleaning-up)

## 常用命令 / 配置
```bash
kubectl config set-context --current --namespace=my-team  # 设置默认 namespace
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count   # 列出所有 node 及其 GPU 型号
kubectl describe node NODE_NAME | grep -A3 Taints   # 查看 node 的 taint，pod 必须容忍才能调度上去
kubectl get pods -o wide         # 列出 pod，-o wide 还会显示每个 pod 在哪个 node 上
kubectl describe pod POD         # pod 的完整状态，最底部是 Events
kubectl logs -f POD              # 持续查看 pod 的 stdout/stderr
kubectl exec -it POD -- bash     # 在运行中的 pod 里开一个 shell
kubectl get events --sort-by=.lastTimestamp   # 整个 namespace 最近的事件
kubectl port-forward pod/dev-pod 8888:8888    # 在本地 http://localhost:8888 访问 pod 里的 Jupyter/TensorBoard/推理服务
kubectl logs -l job-name=gpu-test --prefix --tail=-1 --max-log-requests=64   # 一次看 job 所有 pod 的日志，每行带 pod 名前缀
kubectl get pod POD -o jsonpath='{.status.containerStatuses[*].lastState}'  # 上一次终止的退出码和原因
kubectl debug node/NODE_NAME -it --image=ubuntu   # 有权限时在 node 本身（不是容器）上开 shell，node 的根文件系统挂在 /host
```

```yaml
# 挂载内存型 volume 作为 /dev/shm，解决默认只有 64MB 的问题
containers:
- name: trainer
  volumeMounts:
  - name: dshm
    mountPath: /dev/shm
volumes:
- name: dshm
  emptyDir:
    medium: Memory
    sizeLimit: 64Gi
```

## 常见坑
- `Pending` -> 调度器找不到合适的 node -> `kubectl describe pod`，看是 `Insufficient nvidia.com/gpu`（GPU 被占满或申请数超过单个 node 的数量）、`untolerated taint`（加 toleration）还是 `unbound PersistentVolumeClaim`（PVC 名字写错）。
- 长时间停在 `ContainerCreating` -> 通常镜像还在拉取（ML 镜像常有 10-25GB）-> `kubectl describe pod` 的 Events 里能看到拉取进度，尽量 [减小镜像](../containers/README.md#keep-the-images-small)。
- `CrashLoopBackOff` -> 容器不断退出、k8s 不断重启它 -> `kubectl logs POD --previous` 查看上一次（崩溃那次）的输出。
- `OOMKilled`、退出码 137 -> 超出内存 limit -> 提高 limit 或减少内存使用，见 [Debugging CPU memory OOM](../../debug/pytorch.md#debugging-cpu-memory-oom)。
- `bus error`、`No space left on device` 或莫名其妙的 NCCL 失败 -> `/dev/shm` 只有默认的 64MB -> 挂载 `medium: Memory` 的 emptyDir。
- `py-spy dump` / `gdb` attach 不上卡住的进程 -> 容器默认没有 `SYS_PTRACE` -> 在 `securityContext.capabilities.add` 里加上（集群安全策略可能不允许）。

## 相关章节
- [Kubernetes 概览与术语](./README.md)、[Overcoming job reset on CPU OOM event](./README.md#overcoming-job-reset-on-cpu-oom-event)
- 示例：[dev-pod.yaml](./dev-pod.yaml)、[multi-node-job.yaml](./multi-node-job.yaml)（运行 [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)，适合作为新集群上的第一个 job）、[jobset-train.yaml](./jobset-train.yaml)
- [Fault Tolerance on Kubernetes](./fault-tolerance.md)、[Fast Inter-node Networking](./network.md)、[Data and Model Loading](./storage.md)
- 调试卡住的 job：[diagnosing multi-gpu hanging](../../debug/pytorch.md#approaches-to-diagnosing-multi-gpu-hanging--deadlocks)、[network debug](../../network/debug/)
- 外部：[Kueue](https://kueue.sigs.k8s.io/)、[JobSet](https://jobset.sigs.k8s.io/)、[Kubeflow Trainer](https://www.kubeflow.org/docs/components/trainer/)、[LeaderWorkerSet](https://lws.sigs.k8s.io/)、[Volcano](https://volcano.sh/)
