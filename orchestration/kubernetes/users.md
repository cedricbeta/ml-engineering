# Kubernetes for Users

This is a practical guide for ML engineers who were given access to a k8s cluster and need to get training or inference work done. It's not a k8s manual - just the subset you need day-to-day. If the vocabulary is new to you, first read [Key concepts for newcomers](./README.md#key-concepts-for-newcomers).

## Getting connected

The admins will give you a kubeconfig file (or a cloud CLI command that creates one, e.g., `aws eks update-kubeconfig ...` or `gcloud container clusters get-credentials ...`). `kubectl` reads `~/.kube/config` by default, or whatever `KUBECONFIG` points to.

```bash
kubectl config get-contexts          # list the clusters you can talk to (* marks the current one)
kubectl config use-context my-ctx    # switch clusters
kubectl config set-context --current --namespace=my-team  # set the default namespace
kubectl auth can-i create pods       # check your permissions
```

Setting the default namespace saves you from adding `-n my-team` to every command. All the examples below assume it has been set.

Since you will type `kubectl` a lot, most people add `alias k=kubectl` and enable [shell completion](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_completion/).

## Finding the GPUs

```bash
# all nodes with their GPU type (label added by NVIDIA GPU Feature Discovery)
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count

# how many GPUs each node can give to pods
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\.com/gpu

# how many GPUs are already taken on a given node
kubectl describe node NODE_NAME | grep -A10 "Allocated resources"

# what taints a node has - your pods must tolerate them to land there
kubectl describe node NODE_NAME | grep -A3 Taints
```

Many clusters also run a queueing system like [Kueue](https://kueue.sigs.k8s.io/) - in which case you submit to a queue via a label on your job (e.g., `kueue.x-k8s.io/queue-name: my-team-queue`) and see the queue with `kubectl get workloads`. Ask your admin.

## The core workflow

```bash
kubectl apply -f job.yaml        # create (or update) whatever is described in the file
kubectl get pods -o wide         # list pods - `-o wide` also shows which node each runs on
kubectl get pods -w              # ... and keep watching for changes
kubectl describe pod POD         # the full state of a pod, including the Events at the bottom
kubectl logs -f POD              # follow the stdout/stderr of a pod
kubectl exec -it POD -- bash     # get a shell inside a running pod
kubectl delete -f job.yaml       # delete everything described in the file
```

Rule of thumb: when something doesn't work, `kubectl describe` the object and read the `Events:` section at the bottom - it tells you most of the time what went wrong. To see the recent events of the whole namespace:

```bash
kubectl get events --sort-by=.lastTimestamp
```

To understand what fields an object takes, instead of searching the web:
```bash
kubectl explain pod.spec.containers.resources
kubectl explain job.spec --recursive | less
```

## Interactive development

The k8s equivalent of SLURM's `salloc` + `srun --pty bash` is a pod running `sleep infinity` which you `exec` into. Copy [dev-pod.yaml](./dev-pod.yaml) and adapt it, then:

```bash
kubectl apply -f dev-pod.yaml
kubectl wait --for=condition=Ready pod/dev-pod --timeout=30m
kubectl exec -it dev-pod -- bash
```

Inside the pod you can now run `nvidia-smi`, `python train.py`, etc. just like on a normal node.

Important: an idle dev pod keeps its GPUs reserved, so delete it when you're done: `kubectl delete pod dev-pod`.

Useful helpers while developing:

```bash
# copy files in and out (requires `tar` in the image)
kubectl cp ./my_script.py dev-pod:/workspace/
kubectl cp dev-pod:/workspace/results.json ./results.json

# access a Jupyter/TensorBoard/inference server running inside the pod at http://localhost:8888
kubectl port-forward pod/dev-pod 8888:8888

# CPU/memory usage of the pods (requires metrics-server to be installed)
kubectl top pod
```

If you lose the `exec` session, the processes you launched in it die with it, so use `tmux` or `nohup` inside the pod for anything long running - same as you would over ssh.

Many IDEs (e.g., VSCode's "Kubernetes" and "Dev Containers" extensions) can attach directly to a running pod.

## Pod spec essentials for ML workloads

These are the settings that you're likely to need in almost every ML pod. [dev-pod.yaml](./dev-pod.yaml) and [multi-node-job.yaml](./multi-node-job.yaml) already have them.

### GPUs, CPUs and memory

```yaml
resources:
  limits:
    nvidia.com/gpu: 8
    memory: 512Gi
  requests:
    cpu: "64"
    memory: 512Gi
```

- GPUs must be set in `limits` (the `requests` then default to the same value - if you set them explicitly they must be equal) and must be whole numbers.
- `requests` is what the scheduler reserves for your pod, `limits` is the hard ceiling. Exceeding the memory limit gets your process OOM-killed - see [Overcoming job reset on CPU OOM event](./README.md#overcoming-job-reset-on-cpu-oom-event).
- If you want a whole node, request all of its GPUs - the node's CPU and memory are then usually yours too, but still request them explicitly, as the default requests can be tiny.
- A CPU `limit` is enforced by throttling, which can severely slow down DataLoader workers and tokenization. Many ML clusters set only CPU `requests` for this reason.

### Shared memory

Docker and k8s give a container only 64MB of `/dev/shm`. PyTorch DataLoader workers pass tensors via `/dev/shm`, and NCCL uses it too, so with the default you get errors like `bus error` or `No space left on device` or a mysterious NCCL failure. The fix is to mount a memory-backed volume:

```yaml
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

Note that whatever is written to this volume counts against the container's memory limit.

### Persistent storage

The container's filesystem disappears with the pod, so data, checkpoints and logs must go to a mounted volume or cloud storage:

```bash
kubectl get pvc            # the persistent volume claims available in your namespace
kubectl get storageclass   # the types of storage you can create new PVCs from
```

Then mount the PVC in the pod spec (see `data` in [dev-pod.yaml](./dev-pod.yaml)). Whether a PVC can be mounted by many pods at once (needed for multi-node training) depends on its access mode - `ReadWriteMany` can, `ReadWriteOnce` can only be used from a single node. For where to put datasets, model weights and checkpoints see [Data and Model Loading](./storage.md).

### Secrets

Never bake tokens (HF, W&B, S3 keys) into images or YAML files committed to git. Store them as a Secret and inject them as env vars:

```bash
kubectl create secret generic my-tokens --from-literal=HF_TOKEN=hf_xxx --from-literal=WANDB_API_KEY=xxx
```

```yaml
containers:
- name: trainer
  envFrom:
  - secretRef:
      name: my-tokens
```

### Debugging capabilities

Containers run without the `SYS_PTRACE` capability by default, so `py-spy dump` and `gdb` can't attach to your hanging process. Add it to the pod spec when you need to debug (the cluster's security policy may disallow it):

```yaml
securityContext:
  capabilities:
    add: ["SYS_PTRACE"]
```

For RDMA networking you typically need `IPC_LOCK` too, so that the memory can be pinned.

## Multi-node training

Core k8s doesn't know how to launch a group of pods together and give each one its rank. There are 3 levels of solutions:

1. **Indexed Job + headless Service** - only built-in k8s objects. Each pod gets `$JOB_COMPLETION_INDEX` which becomes the `torchrun --node-rank`, and rank 0 gets a stable DNS name for `--master-addr`. See the fully working [multi-node-job.yaml](./multi-node-job.yaml) - it's a great first job on a new cluster since it runs [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py) across all nodes. The downside is that if one pod fails, nothing restarts the others.
2. **[JobSet](https://jobset.sigs.k8s.io/)** - a k8s-native API for groups of jobs that handles the DNS/headless service, can restart the whole group when any pod fails, and plays well with Kueue. See [jobset-train.yaml](./jobset-train.yaml) and [Fault Tolerance on Kubernetes](./fault-tolerance.md).
3. **[Kubeflow Trainer](https://www.kubeflow.org/docs/components/trainer/)** - (formerly Kubeflow Training Operator with its `PyTorchJob`) on top of that sets the distributed env vars for `torchrun`, and has integrations with DeepSpeed, MPI, etc.

For inference of models that don't fit onto a single node there is [LeaderWorkerSet](https://lws.sigs.k8s.io/), see [Inference on Kubernetes](./inference.md).

Multi-node jobs also need the fast inter-node network inside the pods, see [Fast Inter-node Networking](./network.md).

Whichever you use, multi-node pods also need gang scheduling: if only some of the pods can be scheduled, the scheduled ones sit there holding GPUs while waiting for the rest. Kueue or [Volcano](https://volcano.sh/) solve this, and that's usually set up by the admins.

To see the logs of all pods of a job at once, each line prefixed by the pod's name:

```bash
kubectl logs -l job-name=gpu-test --prefix --tail=-1 --max-log-requests=64
```

## Diagnosing typical problems

The `STATUS` column of `kubectl get pods` tells you where to look:

| Status | What it means | What to do |
| :----- | :------------ | :--------- |
| `Pending` | the scheduler can't find a node for the pod | `kubectl describe pod` - look for `Insufficient nvidia.com/gpu` (all GPUs are taken or you asked for more than a node has), `untolerated taint` (add a toleration), or `unbound PersistentVolumeClaim` (wrong PVC name) |
| `ContainerCreating` for a long time | usually the image is still being pulled - ML images are often 10-25GB | `kubectl describe pod` shows the pull progress in Events, see [image size](../containers/README.md#keep-the-images-small) |
| `ImagePullBackOff` / `ErrImagePull` | wrong image name/tag, or no credentials for a private registry | check the name, ask for `imagePullSecrets` |
| `CrashLoopBackOff` | the container keeps exiting and k8s keeps restarting it | `kubectl logs POD --previous` shows the output of the previous (crashed) run |
| `OOMKilled` | exceeded the memory limit, exit code 137 | raise the memory limit or reduce memory use, see [Debugging CPU memory OOM](../../debug/pytorch.md#debugging-cpu-memory-oom) |
| `Error` | the main process exited with a non-zero exit code | `kubectl logs POD` |
| `Evicted` | the node ran low on memory or disk (often too much written to the container's local filesystem) | write to volumes instead |
| `Terminating` stuck | the node is unreachable or the processes don't exit | `kubectl delete pod POD --grace-period=0 --force` as the last resort |

To get the exit code and reason of the last termination:

```bash
kubectl get pod POD -o jsonpath='{.status.containerStatuses[*].lastState}'
```

Exit code `137` = 128 + 9 (`SIGKILL`) - usually an OOM kill, and `143` = 128 + 15 (`SIGTERM`) - the pod was asked to terminate (deleted, preempted or the node was drained).

### Debugging a hanging job

When a training job hangs (typically in a NCCL collective), `exec` into the pods and use the same tools as on a normal node - see [diagnosing multi-gpu hanging](../../debug/pytorch.md#approaches-to-diagnosing-multi-gpu-hanging--deadlocks) and [network debug](../../network/debug/):

```bash
kubectl exec -it POD -- bash
pip install py-spy
py-spy dump -n -p $(pgrep -f train.py | head -1)
```

For this to work the pod needs the `SYS_PTRACE` [capability](#debugging-capabilities).

### Debugging the node itself

If you have the permissions, you can get a shell on the node (not the container) - useful for checking `dmesg`, `nvidia-smi` of the whole node, or the network:

```bash
kubectl debug node/NODE_NAME -it --image=ubuntu
# the node's root filesystem is mounted at /host
chroot /host
```

## Cleaning up

Finished Jobs and their pods stay around (so that you can read their logs) until deleted. Clean up after yourself:

```bash
kubectl delete job JOB_NAME          # deletes the job and its pods
kubectl get pods --field-selector=status.phase==Succeeded
kubectl delete pods --field-selector=status.phase==Failed
```

Or add `ttlSecondsAfterFinished: 86400` to the Job's `spec` to have it removed automatically a day after it finishes.
