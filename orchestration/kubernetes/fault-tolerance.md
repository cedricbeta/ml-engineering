# Fault Tolerance on Kubernetes

On k8s your training pods will get killed far more often than on SLURM - by hardware failures, by OOM, by preemption, by node upgrades the admins didn't tell you about. This chapter shows how to survive all of these with minimal loss of training time. The general principles are in the [Fault Tolerance](../../training/fault-tolerance/) chapter - here we cover how they translate to k8s.

The strategy has 4 parts:

1. [Save checkpoints frequently](#checkpoint-and-resume) and resume automatically - this is what saves you when things die without warning.
2. [Restart all pods together](#restart-the-whole-job-when-any-pod-fails) when any of them fails.
3. [Save a last checkpoint on a graceful termination](#graceful-termination) - a bonus that saves the work since the last checkpoint when k8s gives a warning.
4. [Avoid the avoidable disruptions](#avoiding-disruptions) and [the bad nodes](#dealing-with-bad-nodes).

A complete example combining all of the above is [jobset-train.yaml](./jobset-train.yaml).

## Why pods die

| Cause | Warning before the kill? | What you see |
| :---- | :----------------------- | :----------- |
| a bug or an exception in your code | n/a | `Error`, non-zero exit code |
| a GPU/NIC/node hardware failure | none | the pod is `Error`/`Unknown`, or the job hangs in a collective until the NCCL timeout |
| CPU OOM | none | `OOMKilled`, exit code 137 |
| preemption by a higher priority job (Kueue/scheduler) | SIGTERM + grace period | `DisruptionTarget` condition on the pod |
| node drain - a node upgrade, maintenance, the autoscaler removing a node | SIGTERM + grace period | `DisruptionTarget` condition on the pod |
| spot/preemptible VM reclaimed by the cloud provider | a short notice (e.g., 30 seconds on GCP, 2 minutes on AWS) regardless of the grace period you asked for | the node disappears |
| node-pressure eviction (the node ran out of memory or disk) | may be none | `Evicted` |

To find out why a pod died:

```bash
kubectl get pod POD -o jsonpath='{.status.conditions}' | python -m json.tool
kubectl get pod POD -o jsonpath='{.status.containerStatuses[*].state}'
kubectl get events --field-selector involvedObject.name=POD
```

A `DisruptionTarget` condition tells you the pod was killed by k8s and not by your code, and its `reason` says why (e.g., `PreemptionByScheduler`, `EvictionByEvictionAPI`, `TerminationByKubelet`, `DeletionByTaintManager`).

## Checkpoint and resume

No matter how good the graceful termination handling is, a hardware failure gives no warning at all, so the foundation is:

1. **Save checkpoints frequently** to storage that outlives the pod - a `ReadWriteMany` PVC on a shared file system or object storage, see [Storage](./storage.md). How frequently is a trade-off between the cost of saving and the cost of re-computing, see [Frequent checkpoint saving](../../training/fault-tolerance/README.md#frequent-checkpoint-saving).
2. **Resume automatically.** The pod's command must be the same for the first start and for every restart, so the training script itself must find the latest checkpoint and resume from it. Most frameworks support it, e.g., HF Trainer's `trainer.train(resume_from_checkpoint=True)` resumes from the last checkpoint found in `output_dir`, and Megatron-LM resumes from the `latest_checkpointed_iteration.txt` in `--load`.
3. **Make the checkpoints atomic.** A pod can be killed in the middle of saving a checkpoint. If your resume logic then picks up that half-written checkpoint, the job will crash on every restart. Save into a temporary directory and rename it once all ranks have finished writing, or update the "latest" pointer file only as the last step. Always keep at least the 2 last checkpoints.
4. **Save the full training state** - not just the model and optimizer, but also the LR scheduler, the RNG states and the dataloader position. Otherwise every restart changes the data order and repeats or skips data.

Test this on day one: start the training, `kubectl delete pod` one of the pods in the middle of a run and check that the job comes back and continues from the last checkpoint with the expected loss.

## Restart the whole job when any pod fails

In a multi-node training job, when one pod dies the remaining pods don't die with it - they hang in the next collective until the NCCL timeout (10 minutes by default in PyTorch, but frameworks often set it much higher), then crash. And when k8s re-creates just the one failed pod, it can't rejoin the existing process group anyway. So the right behavior is: any pod fails -> kill all the pods -> start them all again -> resume from the checkpoint.

A plain Job (as in [multi-node-job.yaml](./multi-node-job.yaml)) doesn't do this. [JobSet](https://jobset.sigs.k8s.io/) does:

```yaml
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
spec:
  failurePolicy:
    maxRestarts: 20
  replicatedJobs:
  - name: workers
    template:
      spec:
        backoffLimit: 0   # any failed pod fails the Job, which makes JobSet restart everything
        ...
```

JobSet will re-create all the pods up to `maxRestarts` times. `kubectl get jobset` shows how many restarts happened so far.

### Don't count preemptions as failures

If the job gets preempted 20 times over a long training it shouldn't give up as if it crashed 20 times. The combination of a Job's [`podFailurePolicy`](https://kubernetes.io/docs/tasks/job/pod-failure-policy/) and JobSet's [`failurePolicy.rules`](https://jobset.sigs.k8s.io/docs/tasks/failure_policy/) lets you distinguish the two:

```yaml
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

A pod killed by a disruption (preemption, eviction, drain) fails its Job with the reason `PodFailurePolicy` and a message like `Pod ns/name has condition DisruptionTarget matching FailJob rule at index 0`, which the JobSet rule matches and restarts without counting. A crash fails the Job with `BackoffLimitExceeded`, which falls through to the default action `RestartJobSet` and counts towards `maxRestarts`.

Matching on the message matters as soon as there is more than one `FailJob` rule, since all of them produce the same reason `PodFailurePolicy`. For example, to not restart 20 times on a non-retriable error (a bad config, a missing dataset), make the training exit with a special code and stop the whole JobSet on it:

```yaml
spec:
  failurePolicy:
    rules:
    - action: FailJobSet
      onJobFailureReasons: [PodFailurePolicy]
      onJobFailureMessagePatterns: ["failed with exit code 42"]
    - action: RestartJobSetAndIgnoreMaxRestarts
      onJobFailureReasons: [PodFailurePolicy]
      onJobFailureMessagePatterns: ["has condition DisruptionTarget"]
  replicatedJobs:
  - name: workers
    template:
      spec:
        podFailurePolicy:
          rules:
          - action: FailJob
            onPodConditions:
            - type: DisruptionTarget
          - action: FailJob
            onExitCodes:
              containerName: trainer
              operator: In
              values: [42]
```

The catch: the exit code k8s sees is that of the container's PID 1. With `torchrun` as PID 1, a worker exiting with 42 makes `torchrun` itself exit with 1, so this rule only works if your launcher passes the worker's exit code through.

### In-process restarts

`torchrun --max-restarts=N` can restart the worker processes inside the same pods without k8s getting involved - this is faster since there is no rescheduling and no re-pulling of the image, but it only helps with software failures - if the node is gone, the pods need to be re-created anyway. Don't combine it with the [save-and-exit mechanism](#the-training-loop-side) below: the workers would exit after saving, `torchrun` would restart them, and they would keep finding the flag file and exiting again. Keep the default `--max-restarts=0` in that case. See also [Rebuilding the process group in place](../../training/fault-tolerance/README.md#rebuilding-the-process-group-in-place).

## Graceful termination

When k8s deletes a pod on purpose (preemption, drain, `kubectl delete`) it gives the pod a chance to exit cleanly. If you use this chance to save a checkpoint, you lose none of the work done since the last checkpoint.

### The termination sequence

1. The pod is marked for deletion.
2. If the container has a [`preStop` hook](https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/), it's run.
3. `SIGTERM` is sent to PID 1 of each container.
4. If the containers are still running when `terminationGracePeriodSeconds` (counted from step 1) has passed, they get `SIGKILL`ed.

The default `terminationGracePeriodSeconds` is only 30 seconds, which is way too short to finish the current training step and to save a checkpoint of a large model. Set it in the pod spec to how long these 2 take, times 2 to be safe:

```yaml
spec:
  terminationGracePeriodSeconds: 900
```

Note that a spot VM being reclaimed or a hard node-pressure eviction won't honor it.

### Gotcha 1: the signal must reach your program

Only PID 1 of the container receives the `SIGTERM`. If your container's command is:

```yaml
command: ["bash", "-c"]
args:
- |
  pip install something
  torchrun ... train.py
```

then PID 1 is `bash`, and `bash` doesn't forward the signal to its children. The training never hears about it and gets `SIGKILL`ed at the end of the grace period. The fix is to `exec` the last command, which replaces `bash` with `torchrun`:

```yaml
  exec torchrun ... train.py
```

If you use a launcher script as the image's `ENTRYPOINT`, make sure it does the same, or use an init process like [tini](https://github.com/krallin/tini) that forwards the signals.

### Gotcha 2: `torchrun` only gives the workers 30 seconds

`torchrun` handles `SIGTERM` by forwarding it to all the worker processes, but if they haven't exited within 30 seconds it `SIGKILL`s them - regardless of `terminationGracePeriodSeconds`. So even with a 15 minute grace period, a checkpoint that takes more than 30 seconds to save will be cut short if you rely on `SIGTERM`.

Starting from `torch==2.12` this can be changed with `torchrun --shutdown-timeout=SECONDS` (or the `TORCH_ELASTIC_SHUTDOWN_TIMEOUT` env var). In older versions the 30 seconds are hardcoded.

A way around it that works with any version and any launcher is to not rely on `SIGTERM` at all, and use the `preStop` hook to tell the training to save and exit, and to wait for it - the `SIGTERM` is only sent after the hook returns:

```yaml
lifecycle:
  preStop:
    exec:
      command:
      - bash
      - -c
      - touch /tmp/save-and-exit; while pgrep -f "[t]rain.py" > /dev/null; do sleep 5; done
```

(`[t]rain.py` is a regex that matches `train.py` but not this command line itself - otherwise `pgrep` would find the hook's own `bash` process and wait forever.)

This is the k8s version of the [save switch](../../training/fault-tolerance/README.md#save-switch), and the same file can be used to trigger a save manually: `kubectl exec POD -- touch /tmp/save-and-exit`.

### The training loop side

Each rank checks the flag file (and a `SIGTERM` flag, as a fallback for cases when the `preStop` hook wasn't run), and the ranks agree via an `all_reduce` - so if any one pod is being deleted, all the ranks save the same step and exit together:

```python
import os, signal, sys
import torch
import torch.distributed as dist

SAVE_AND_EXIT_FILE = "/tmp/save-and-exit"
got_sigterm = False

def sigterm_handler(signum, frame):
    global got_sigterm
    got_sigterm = True

signal.signal(signal.SIGTERM, sigterm_handler)

def should_save_and_exit():
    # if any rank wants to stop - all ranks stop
    flag = torch.tensor(int(got_sigterm or os.path.exists(SAVE_AND_EXIT_FILE)), device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return flag.item() == 1

for step in range(start_step, max_steps):
    train_step()

    if step % 10 == 0 and should_save_and_exit():
        save_checkpoint(step)
        dist.barrier()
        # non-zero exit code, so that the job gets restarted and resumes from this checkpoint
        sys.exit(1)
```

The check is done every few steps since `.item()` forces a CPU-GPU synchronization. The same logic would work with SLURM's `--signal`, see [Dealing with forced job preemption](../../training/fault-tolerance/README.md#dealing-with-forced-job-preemption).

## Avoiding disruptions

### Node upgrades

Managed k8s services upgrade the node pools to new k8s versions, and some (e.g., GKE) do it automatically by default. An upgrade drains the nodes one by one, which for a multi-node training means one restart per node. Ask the admins to configure maintenance windows/exclusions for the training node pools, so that the upgrades happen when you choose.

### PodDisruptionBudget

A [PodDisruptionBudget](https://kubernetes.io/docs/concepts/workloads/pods/disruptions/) (PDB) tells k8s that it must not voluntarily evict your pods - `kubectl drain`, the cluster autoscaler and the managed upgrades respect it:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: train-pdb
spec:
  maxUnavailable: 0
  selector:
    matchLabels:
      jobset.sigs.k8s.io/jobset-name: train
```

A PDB doesn't protect against hardware failures, spot reclaims, OOM or a direct `kubectl delete pod`. Also note that managed upgrades respect PDBs only for a limited time before forcing the eviction anyway, and a PDB that blocks drains forever will get you into an argument with your admins - coordinate with them.

Additionally, the annotation `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"` on the pods tells the cluster autoscaler not to scale down the nodes they run on.

## Dealing with bad nodes

When a node has a failing GPU or NIC, the restarted job often gets scheduled right back onto the same node and fails again - a restart loop that burns through `maxRestarts`.

- Find the culprit: `kubectl get pods -o wide` shows which node each pod ran on, and the logs of the first rank that failed usually point to it (e.g., an `Xid` error, see [NVIDIA GPU debug](../../compute/accelerator/nvidia/debug.md)).
- If you have the permissions, take the node out of scheduling with `kubectl cordon NODE` (`kubectl uncordon NODE` to bring it back) and report it to the admins.
- If not, exclude the node in your pod spec until it's fixed:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: NotIn
          values: ["bad-node-1", "bad-node-2"]
```

Many clusters run automatic GPU health checks (e.g., NVIDIA's DCGM-based health checks, or the cloud provider's node auto-repair) that taint or replace bad nodes - ask your admin what's in place.

Since a failed node may take hours to be replaced, it pays to have spare nodes in the node pool, see [Always plan to have more nodes than needed](../../training/fault-tolerance/README.md#always-plan-to-have-more-nodes-than-needed).

## Detecting hangs

A dead pod gets noticed, but a hung job looks perfectly `Running` to k8s while burning the GPUs. The NCCL timeout is what turns a hang into a crash, which then triggers the restart logic above - so check which timeout is actually in effect. PyTorch's default for NCCL is 10 minutes, but frameworks often override it, e.g., HF Trainer's `ddp_timeout` defaults to 30 minutes, and some recipes set it to hours, which means hours of idle GPUs before every restart. Set it explicitly:

```python
import datetime
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=20))
```

Pick a timeout longer than the slowest legitimate operation, e.g., saving or loading a checkpoint, if it's done in a collective - but not much longer. For more on what to watch, see [Is-job-hanging watchdog](../../training/fault-tolerance/README.md#is-job-hanging-watchdog), and to debug the hang itself see [Diagnosing NCCL collective hangs](../../debug/pytorch.md#diagnosing-nccl-collective-hangs).
