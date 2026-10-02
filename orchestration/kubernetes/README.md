# Kubernetes

In general IMHO kubernetes is the wrong environment for doing training work, with [SLURM](../slurm/) being a much more user-friendly orchestration. SLURM builds on top of Unix and so it just makes multi-node coordination easier. But k8s reinvents the wheel completely, leading to lots of complexities and many normal Unix features are missing from its feature set, making it into a very challenging environment to work in, requiring a large support team to just deal with so many k8s problems and a lack of rudimentary Unix features (e.g., you can't even have Unix usernames and all users run under the same Unix username - think accountability and security!).

When forced into k8s one will find a way to get their work done, but at a huge human and $$ cost to the training shop.

This is going to be a small chapter since I can't imagine how I'd even approach covering k8s complexities here, so I'm just going to cover important gotchas that should ease your work a bit.

- [Day One on a New Cluster](./new-cluster-checklist.md) - a step by step checklist to verify a new cluster before running anything expensive on it.
- [Kubernetes for Users](./users.md) - the `kubectl` commands and pod-spec settings needed to do ML work, and how to diagnose the typical failures.
- [Fault Tolerance](./fault-tolerance.md) - surviving hardware failures, preemptions and node upgrades: checkpoint/resume, restarting the whole job, graceful termination.
- [Fast Inter-node Networking](./network.md) - getting InfiniBand/RoCE/EFA into the pods and verifying that NCCL actually uses it.
- [Data and Model Loading](./storage.md) - where to put datasets, model weights, checkpoints and caches.
- [Inference on Kubernetes](./inference.md) - serving LLMs: health probes, updates, multi-node replicas, cold start, autoscaling.
- [Snapshots](./snapshots.md) - restoring initialized replicas instead of cold-starting them: from compile caches and sleep mode to process checkpoint/restore with Dynamo Snapshot.
- [Containers](../containers/) - everything in k8s runs inside a container, so you will need to know how to build and debug container images, and [how to build them in CI](../containers/ci.md).
- [dev-pod.yaml](./dev-pod.yaml) - an interactive GPU pod (the k8s equivalent of SLURM's `salloc`).
- [multi-node-job.yaml](./multi-node-job.yaml) - a minimal multi-node `torchrun` job.
- [jobset-train.yaml](./jobset-train.yaml) - a fault-tolerant multi-node training job.
- [vllm-deployment.yaml](./vllm-deployment.yaml) - a vLLM inference server.

## Key concepts for newcomers

If you come from SLURM or from working on a plain Unix box, here is a minimal mapping of the k8s vocabulary you will see everywhere:

| k8s term | What it is | Closest SLURM/Unix analogy |
| :------- | :--------- | :------------------------- |
| Cluster | a set of machines managed by a single control plane | a SLURM cluster |
| Node | a single machine (VM or bare metal) | a node |
| Pod | one or more containers scheduled together on the same node, sharing network and volumes - the unit of scheduling | a job step running on one node |
| Container | a process running from a container image in an isolated filesystem | a process (in a chroot) |
| Job | runs pods to completion, possibly re-trying them | `sbatch` |
| Deployment | keeps N identical pods always running - used for inference servers, not training, see [Inference on Kubernetes](./inference.md) | a daemon managed by systemd |
| Namespace | a named partition of the cluster's objects and quotas | an account/partition |
| Service | a stable DNS name (and optionally a load balancer) in front of a set of pods | `/etc/hosts` entry |
| PersistentVolumeClaim (PVC) | a request for persistent storage that gets mounted into pods | a shared file system mount |
| ConfigMap / Secret | small config files or credentials injected into pods | files in `~/.config`, env vars |
| Taint / Toleration | a node "repels" pods unless they explicitly tolerate the taint | partition access restrictions |
| `kubectl` | the CLI client that talks to the cluster's API server | `squeue`, `sbatch`, `scancel`, ... rolled into one |

Things that are fundamentally different from SLURM and that trip up most newcomers:

1. **Everything is declarative.** You don't run a command on a node - you submit a YAML description of what should exist (`kubectl apply -f job.yaml`) and k8s works to make it so. To change something you edit the YAML and re-apply it, or delete and re-create the object.
2. **Everything is a container.** There is no "login into the compute node and run `python train.py`". Your code, its python environment and CUDA libraries must be in a container image (or mounted into it), see [Containers](../containers/).
3. **The pod's filesystem is ephemeral.** Anything written outside of a mounted persistent volume (PVC, NFS, etc.) disappears when the pod ends - and this includes checkpoints and logs. Always save checkpoints to persistent storage or to cloud object storage.
4. **Pods are cattle, not pets.** k8s will happily kill and re-create your pod (OOM, node drain, preemption, a failed health check). Design your training to resume from the last checkpoint, see [Fault Tolerance on Kubernetes](./fault-tolerance.md).
5. **Multi-node isn't built in.** Core k8s has no notion of "give me 16 nodes at once and tell each process its rank". You either wire it yourself (see [multi-node-job.yaml](./multi-node-job.yaml)) or use an operator built for this, see [Multi-node training](./users.md#multi-node-training).
6. **GPUs are an "extended resource".** You request them with `nvidia.com/gpu: 8` in the pod's resource limits. GPUs can't be shared or overcommitted between pods (unless your admin has configured MIG or time-slicing), and a pod not requesting GPUs shouldn't see any (but depending on the cluster's configuration it might, see [this gotcha](../containers/drivers.md#how-gpus-get-into-a-container)).

### How GPUs get into a pod

On a GPU cluster the admins typically install the [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator), which deploys on every GPU node:

- the NVIDIA driver (unless it's pre-installed on the node image)
- the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) - mounts the driver libraries and `/dev/nvidia*` devices into containers
- the [device plugin](https://github.com/NVIDIA/k8s-device-plugin) - advertises `nvidia.com/gpu` to the scheduler
- GPU Feature Discovery - adds node labels like `nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3`
- DCGM exporter - GPU metrics for Prometheus/Grafana

The NVIDIA driver lives on the node, while the CUDA user-space libraries (CUDA runtime, cuDNN, NCCL) live inside your container image. Therefore the image's CUDA version must be supported by the node's driver, see [CUDA and the driver](../containers/README.md#cuda-and-the-nvidia-driver), and for the full picture of all the drivers involved see [Drivers: Kernel, Host and Container](../containers/drivers.md).

The equivalent for AMD GPUs is the [AMD GPU Operator](https://github.com/ROCm/gpu-operator) with the `amd.com/gpu` resource.

For the fast inter-node network (InfiniBand/RoCE) there is the [NVIDIA Network Operator](https://github.com/Mellanox/network-operator), and the cloud providers have their own variants (e.g., EFA on AWS, GPUDirect-TCPXO/RDMA on GCP). How RDMA devices are exposed to pods is very cluster-specific, so ask your admin what resources and annotations a pod needs to get them - otherwise NCCL will silently fall back to the slow TCP sockets, see [Fast Inter-node Networking](./network.md).

### Kubernetes managed services

You're unlikely to build a k8s cluster yourself. Typically you get one from a cloud provider - EKS (AWS), GKE (Google), AKS (Azure), or from a neocloud (CoreWeave, Nebius, Lambda, etc.) - and the cluster's admin team gives you a kubeconfig file and a namespace to work in.

### Recommended learning materials

- [Kubernetes Basics tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/) - the official interactive intro.
- [kind](https://kind.sigs.k8s.io/) or [minikube](https://minikube.sigs.k8s.io/) - run a toy k8s cluster on your laptop to practice `kubectl` without needing GPUs or risking a shared cluster.
- [kubectl cheatsheet](https://kubernetes.io/docs/reference/kubectl/quick-reference/)
- [k9s](https://k9scli.io/) - a terminal UI for k8s, which makes exploring pods, logs and events much faster than typing `kubectl` commands.


## Setup

### Overcoming job reset on CPU OOM event

This default feature enabled in k8s v1.28 makes absolutely no sense in the context of interactive training jobs. If any of your processes get CPU OOM'ed you get kicked out and your job gets reset. This feature is good for serving inference in production, but it's a huge problem for interactive training work, for example when one is trying to find out an optimal performance training configuration, as instead of being able to recover from a crashed process, adjust things and try again, you have to start from scratch and not even have logs, leading to the problem unless you manually sync those in time.

So to overcome this you need to get k8s to set `memory.oom.group = 0` either on the node pool level or cluster level configs - the default is `memory.oom.group = 1` so if your job gets killed on a cpu-oom event ask your k8s admin to make this change. The value of `0` will just kill the process that caused cpu-oom.

Here is how this is done. Kubernetes 1.32 introduced the kubelet flag [`singleProcessOOMKill`](https://github.com/kubernetes/kubernetes/pull/126096), which allows you to set `memory.oom.group = 0`.

```
compute:
  additionalNodePools:
    - name: foo
      kubeletConfig:
        singleProcessOOMKill: true
```


To check the actual setting from within the running node, do:
```bash
$ cat /sys/fs/cgroup/memory.oom.group
```
A value of `0` means the fix is in effect and only the offending process will be killed on a CPU OOM event; a value of `1` is the problematic default that kills the whole job's process group.