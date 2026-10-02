# Kubernetes 上的容错：中文要点

> 英文原文：[Fault Tolerance on Kubernetes](./fault-tolerance.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
在 k8s 上，训练 pod 被杀的频率远高于 SLURM：硬件故障、OOM、抢占、管理员没通知的 node 升级都会导致 pod 被杀。这一章讲如何在 k8s 上扛过这些情况，同时尽量少损失训练时间。策略分 4 部分：频繁保存 checkpoint 并自动恢复；任一 pod 失败时所有 pod 一起重启；优雅终止时保存最后一个 checkpoint；避免可以避免的中断和坏节点。完整示例见 [jobset-train.yaml](./jobset-train.yaml)，通用原则见 [Fault Tolerance](../../training/fault-tolerance/) 章节。

## 核心概念
- **`DisruptionTarget`**：pod 上的一个 condition，表示 pod 是被 k8s 杀掉的，而不是你的代码出错；它的 `reason` 说明原因，如 `PreemptionByScheduler`、`EvictionByEvictionAPI`、`TerminationByKubelet`、`DeletionByTaintManager`。
- **JobSet `failurePolicy`**：任一 Job 失败时重建全部 pod，最多重启 `maxRestarts` 次；`rules` 可以按 Job 的失败原因（`onJobFailureReasons`）和失败消息（`onJobFailureMessagePatterns`）选择不同动作。
- **`podFailurePolicy`**：Job 级别的规则，根据 pod 的 condition 或退出码决定动作（如 `FailJob`）。
- **`terminationGracePeriodSeconds`**：从 pod 被标记删除开始计时的宽限期，到期后容器还在运行就会被 `SIGKILL`，默认只有 30 秒。
- **`preStop` hook**：pod 删除时，在发送 `SIGTERM` 之前于容器内运行的钩子；hook 返回后才会发送 `SIGTERM`。
- **PodDisruptionBudget (PDB)**：告诉 k8s 不能主动驱逐你的 pod，`kubectl drain`、cluster autoscaler 和托管服务的升级都会遵守它。
- **cordon**：把 node 标记为不可调度（`kubectl cordon NODE`，恢复用 `kubectl uncordon NODE`）。

## 关键要点
- **checkpoint + 自动恢复是基础**，因为硬件故障没有任何预警。checkpoint 要存到比 pod 活得久的存储（共享文件系统上的 `ReadWriteMany` PVC 或对象存储）；首次启动和每次重启必须用同一条命令，由训练脚本自己找到最新的 checkpoint 恢复（如 HF Trainer 的 `trainer.train(resume_from_checkpoint=True)`，Megatron-LM 从 `--load` 中的 `latest_checkpointed_iteration.txt` 恢复）。checkpoint 还要原子化、要完整：先写到临时目录，所有 rank 都写完后再 rename，或者最后一步才更新 "latest" 指针文件，至少保留最近 2 个 checkpoint；除了模型和优化器，还要保存 LR scheduler、RNG 状态和 dataloader 位置，否则每次重启都会打乱数据顺序，重复或跳过数据。第一天就测试：训练中途 `kubectl delete pod` 删掉一个 pod，确认 job 能恢复并以预期的 loss 继续。[详见](./fault-tolerance.md#checkpoint-and-resume)
- **一个 pod 失败，就重启整个 job**：其他 pod 不会跟着死，而是卡在下一个 collective 里，直到 NCCL 超时（PyTorch 默认 10 分钟，但各种框架常常设得长得多）才崩溃；k8s 单独重建的那个 pod 也无法重新加入原来的 process group。正确做法是：任一 pod 失败 -> 杀掉所有 pod -> 全部重新启动 -> 从 checkpoint 恢复。普通 Job 做不到，JobSet 可以。[详见](./fault-tolerance.md#restart-the-whole-job-when-any-pod-fails)
- **抢占不要算作失败**：Job 的 `podFailurePolicy` 让带 `DisruptionTarget` condition 的 pod 以原因 `PodFailurePolicy` 使 Job 失败，消息类似 `Pod ns/name has condition DisruptionTarget matching FailJob rule at index 0`；JobSet 规则同时匹配这个原因和 `"has condition DisruptionTarget"` 消息，用 `RestartJobSetAndIgnoreMaxRestarts` 重启且不计数。真正的崩溃以 `BackoffLimitExceeded` 失败，走默认动作 `RestartJobSet`，计入 `maxRestarts`。只要有不止一条 `FailJob` 规则就必须匹配消息，因为它们产生的原因都是 `PodFailurePolicy`：例如让训练在不可重试的错误（错误配置、数据集缺失）时以特殊退出码（如 42）退出，再用匹配 `"failed with exit code 42"` 的 `FailJobSet` 规则停掉整个 JobSet。注意 k8s 看到的是容器 PID 1 的退出码：`torchrun` 作为 PID 1 时，worker 以 42 退出会让 `torchrun` 自己以 1 退出，所以只有当你的 launcher 把 worker 的退出码传递出来时，按退出码的规则才有效。[详见](./fault-tolerance.md#dont-count-preemptions-as-failures)
- **进程内重启**：`torchrun --max-restarts=N` 在原来的 pod 里重启 worker 进程，不需要重新调度和拉镜像，更快，但只对软件故障有效；node 没了的话 pod 还是要重建。不要把它和下文的“保存并退出”机制一起用：worker 保存后退出，`torchrun` 会重启它们，它们又发现 flag 文件再次退出，如此反复；这种情况下保持默认的 `--max-restarts=0`。[详见](./fault-tolerance.md#in-process-restarts)
- **终止流程**：pod 被标记删除 -> 运行 `preStop` hook（如果有） -> 向每个容器的 PID 1 发 `SIGTERM` -> 从第一步算起超过 `terminationGracePeriodSeconds` 仍在运行则 `SIGKILL`。默认 30 秒太短，应设为“跑完当前 step + 保存 checkpoint”所需时间的 2 倍。spot VM 被回收或硬性的 node-pressure 驱逐不会遵守这个值。[详见](./fault-tolerance.md#the-termination-sequence)
- **坑 1：信号必须送到你的程序**。只有 PID 1 收到 `SIGTERM`，如果命令是 `bash -c`，PID 1 是 `bash`，它不会把信号转发给子进程。解决：最后一条命令前加 `exec`；用作 `ENTRYPOINT` 的启动脚本也要这样做，或使用 [tini](https://github.com/krallin/tini) 这类会转发信号的 init 进程。[详见](./fault-tolerance.md#gotcha-1-the-signal-must-reach-your-program)
- **坑 2：`torchrun` 只给 worker 30 秒**。它把 `SIGTERM` 转发给 worker 后，30 秒内没退出就 `SIGKILL`，与 `terminationGracePeriodSeconds` 无关。从 `torch==2.12` 起可以用 `torchrun --shutdown-timeout=SECONDS`（或 `TORCH_ELASTIC_SHUTDOWN_TIMEOUT` 环境变量）修改，更早的版本里这 30 秒是写死的。适用于任何版本的做法是不依赖 `SIGTERM`：用 `preStop` hook 通知训练保存并退出，并等它结束。[详见](./fault-tolerance.md#gotcha-2-torchrun-only-gives-the-workers-30-seconds)
- **训练循环一侧**：每个 rank 检查 flag 文件（以及作为后备的 `SIGTERM` 标志），再通过 `all_reduce` 达成一致，这样只要有一个 pod 被删除，所有 rank 都保存同一个 step 并一起以非零退出码退出，让 job 重启并恢复。每隔几个 step 才检查一次，因为 `.item()` 会强制 CPU-GPU 同步。[详见](./fault-tolerance.md#the-training-loop-side)
- **避免中断**：请管理员为训练 node pool 配置维护窗口/排除期（有些托管服务如 GKE 默认自动升级）；PDB 挡不住硬件故障、spot 回收、OOM 和直接的 `kubectl delete pod`，托管升级也只在有限时间内遵守 PDB，要和管理员协调；坏节点会让重启后的 job 被调度回去再次失败，耗光 `maxRestarts`；坏节点的处理见下方“常见坑”。[详见](./fault-tolerance.md#avoiding-disruptions)
- **检测 hang**：hang 住的 job 在 k8s 看来一切 `Running`，GPU 却一直被占着。把 hang 变成崩溃（从而触发重启）的是 NCCL timeout，所以要确认实际生效的是哪个值：PyTorch 默认 10 分钟，但框架常会覆盖，例如 HF Trainer 的 `ddp_timeout` 默认 30 分钟，有些配方甚至设成几个小时，意味着每次重启前 GPU 要空等几个小时。应显式设置，并且要比最慢的合法操作（如在 collective 中保存/加载 checkpoint）长，但不要长太多。[详见](./fault-tolerance.md#detecting-hangs)

## 常用命令 / 配置
```bash
kubectl get pod POD -o jsonpath='{.status.conditions}' | python -m json.tool   # 查看 pod 的 conditions，找 DisruptionTarget 及其 reason
kubectl get events --field-selector involvedObject.name=POD   # 与这个 pod 相关的事件
kubectl exec POD -- touch /tmp/save-and-exit   # 手动触发“保存并退出”
```

```yaml
# 不把抢占/驱逐计入 maxRestarts
spec:
  failurePolicy:
    maxRestarts: 20
    rules:
    - action: RestartJobSetAndIgnoreMaxRestarts
      onJobFailureReasons:
      - PodFailurePolicy
      onJobFailureMessagePatterns:
      - "has condition DisruptionTarget"
  replicatedJobs:
  - name: workers
    template:
      spec:
        backoffLimit: 0
        podFailurePolicy:
          rules:
          - action: FailJob
            onPodConditions:
            - type: DisruptionTarget
```

```yaml
# pod spec 里：把宽限期设长（默认只有 30 秒）
spec:
  terminationGracePeriodSeconds: 900
```

```yaml
# 容器上的 preStop hook：先通知训练保存并退出，等 train.py 结束后 k8s 才发 SIGTERM
# 正则 [t]rain.py 能匹配 train.py，但不会匹配 hook 自己的 bash 命令行，否则会永远等下去
lifecycle:
  preStop:
    exec:
      command:
      - bash
      - -c
      - touch /tmp/save-and-exit; while pgrep -f "[t]rain.py" > /dev/null; do sleep 5; done
```

## 常见坑
- 删除 pod 时训练根本没保存，宽限期结束直接被 `SIGKILL` -> PID 1 是 `bash`，不转发 `SIGTERM` -> 最后一条命令写成 `exec torchrun ... train.py`，或用 tini。
- 宽限期设了 15 分钟，超过 30 秒的 checkpoint 保存还是被打断 -> `torchrun` 只给 worker 30 秒 -> 改用 `preStop` hook 方案，或在 `torch==2.12` 及以上用 `--shutdown-timeout`。
- 按退出码设的 `FailJobSet` 规则从不生效，不可重试的错误照样重启 20 次 -> PID 1 是 `torchrun`，worker 的退出码 42 变成了 `torchrun` 的 1 -> 让 launcher 把 worker 的退出码传递出来。
- 每次重启都崩溃 -> 恢复时读到了保存到一半的 checkpoint -> 原子化保存，至少保留最近 2 个。
- 重启循环很快耗光 `maxRestarts` -> 一直被调度回同一个坏 node -> 用 `kubectl get pods -o wide` 和最先失败的 rank 的日志（如 `Xid` 错误）找出 node，`kubectl cordon NODE` 或用 `nodeAffinity` 的 `NotIn` 排除它，并报告给管理员。
- job 卡住不动，但 k8s 显示 `Running`，GPU 空等几个小时才重启 -> 框架把 NCCL timeout 设得很长（如 HF Trainer 的 `ddp_timeout` 默认 30 分钟） -> 在 `dist.init_process_group` 中显式设置 `timeout`，比最慢的合法操作长一些即可。

## 相关章节
- 通用原则：[Fault Tolerance](../../training/fault-tolerance/)，包括 [Frequent checkpoint saving](../../training/fault-tolerance/README.md#frequent-checkpoint-saving)、[save switch](../../training/fault-tolerance/README.md#save-switch)、[Is-job-hanging watchdog](../../training/fault-tolerance/README.md#is-job-hanging-watchdog)、[Always plan to have more nodes than needed](../../training/fault-tolerance/README.md#always-plan-to-have-more-nodes-than-needed)
- 示例：[jobset-train.yaml](./jobset-train.yaml)、[multi-node-job.yaml](./multi-node-job.yaml)
- [Storage](./storage.md)、[NVIDIA GPU debug](../../compute/accelerator/nvidia/debug.md)、[Diagnosing NCCL collective hangs](../../debug/pytorch.md#diagnosing-nccl-collective-hangs)
- 外部：[JobSet](https://jobset.sigs.k8s.io/)、[`podFailurePolicy`](https://kubernetes.io/docs/tasks/job/pod-failure-policy/)、[PodDisruptionBudget](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/)
