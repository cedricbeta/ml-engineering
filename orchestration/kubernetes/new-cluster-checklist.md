# Day One on a New Cluster

You've been given access to a new GPU cluster (or a new node pool, or a new image). Before you run anything expensive on it, spend a day verifying that it delivers what you're paying for. Every problem found now - a bad GPU, a NIC that fell back to TCP, slow storage, a job that doesn't resume - would otherwise be found in the middle of a large run, at a much higher cost.

This checklist ties together the tools from the other chapters, in the order that makes sense: each step depends on the previous one working. For each step there is what to run, what a good result looks like, and where to look if it isn't.

It's written for k8s, but the same steps apply on SLURM - just launch the same scripts with `srun` instead of a pod.

The commands that use this repo's YAML files and scripts (`dev-pod.yaml`, `multi-node-job.yaml`, `../containers/driver-report.sh`) assume you run them from `orchestration/kubernetes/` in a checkout of [the repo](https://github.com/stas00/ml-engineering) - adapt them to your cluster first. The commands that run inside the pods download what they need with `curl`.

For a more thorough acceptance test of the hardware itself - DCGM diagnostics, MAMF/MSMF on every GPU, intra- and inter-node all-reduce and fio, written up as a dated report - the book also has an [evaluate-cluster](../../skills/evaluate-cluster/SKILL.md) skill that an AI agent can run for you. This checklist complements it with the k8s-specific parts: the driver stack inside the pods, RDMA in the pods, PVCs, and the fault tolerance drill.

Keep the numbers you measure - they become the baseline that you compare against whenever something seems slow later on. A template is at the [end](#record-the-baseline).

## 0. Access and inventory

```bash
kubectl config get-contexts
kubectl auth can-i create pods
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count,nvidia.com/cuda.driver-version.full
kubectl get storageclass
kubectl get pvc
```

Good: you can create pods, you see the expected number of nodes, all of them with the same GPU type and the same driver version.

Watch out for: nodes with fewer GPUs than the rest (a GPU fell off the bus), and nodes with a different driver version (they were set up at a different time and may behave differently). See [Getting connected](./users.md#getting-connected) and [Finding the GPUs](./users.md#finding-the-gpus).

## 1. The driver stack

Start a [dev pod](./dev-pod.yaml) and run [driver-report.sh](../containers/driver-report.sh) inside it:

```bash
kubectl apply -f dev-pod.yaml
kubectl wait --for=condition=Ready pod/dev-pod --timeout=30m
kubectl exec -i dev-pod -- bash -s < ../containers/driver-report.sh
```

Good:
- the kernel driver version and the `libcuda.so` version are identical
- the CUDA version supported by the driver is >= the one PyTorch was built with, and `torch.cuda.is_available()` is `True` with all GPUs visible
- your GPU's architecture (e.g., `sm_90`) is in the list of archs compiled into torch
- on NVSwitch systems the Fabric state is `Completed` / `Success`
- the RDMA devices are present, their ports are `ACTIVE` at the expected rate, and the memlock limit is `unlimited`

Otherwise see [Symptom to cause](../containers/drivers.md#symptom-to-cause).

Note how long the pod took to become ready - most of it is the image pull, which you'll pay on every new node (`kubectl describe pod dev-pod` shows `Successfully pulled image ... in ...`).

## 2. A single node

Inside the dev pod, check that all GPUs of the node can talk to each other:

```bash
curl -sO https://raw.githubusercontent.com/stas00/ml-engineering/master/debug/torch-distributed-gpu-test.py
NCCL_DEBUG=INFO torchrun --nproc-per-node=8 torch-distributed-gpu-test.py
```

(`--nproc-per-node` is the number of GPUs per node - 4 on GB200/GB300 NVL72.)

Good: every rank reports OK. If it hangs, see [network debug](../../network/debug/).

Then measure the GPUs' actual compute performance with [mamf-finder-all-gpus.py](../../compute/accelerator/benchmarks/mamf-finder-all-gpus.py): it runs [mamf-finder.py](../../compute/accelerator/benchmarks/mamf-finder.py)'s automatic search on GPU0, and then measures every other GPU while all the others compute too - the way a training job loads the node:

```bash
B=https://raw.githubusercontent.com/stas00/ml-engineering/master/compute/accelerator/benchmarks
curl -sO $B/mamf-finder.py
curl -sO $B/mamf-finder-all-gpus.py   # runs ./mamf-finder.py, so keep both in one directory
python mamf-finder-all-gpus.py
```

Good: the MAMF and MSMF are close to the values for this GPU in the [Maximum Achievable and Sustainable Matmul FLOPS comparison table](../../compute/accelerator/README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table), and the per-GPU spread is small - a single slow GPU slows down a whole synchronous training job, see [Not all accelerators are created equal](../../compute/accelerator/README.md#not-all-accelerators-are-created-equal). Repeat on several nodes.

## 3. Multi-node connectivity

Run [multi-node-job.yaml](./multi-node-job.yaml) on 2 nodes and check the NCCL logs:

```bash
kubectl apply -f multi-node-job.yaml
kubectl logs -l job-name=gpu-test --prefix --tail=-1 | grep -E "NET/|via NET"
```

Good: `NET/IB` (or `NET/OFI` on AWS), one NIC per GPU listed, and `GDRDMA` in the channel lines. Bad: `NET/Socket` - the fast network isn't used. See [Fast Inter-node Networking](./network.md#2-does-nccl-use-them). On GB200/GB300 NVL72 also look for the `MNNVL 1 ...` line, which shows that the NVLink between the nodes is used - without it NCCL has silently fallen back to the NICs, see [How NCCL uses it](../containers/drivers.md#how-nccl-uses-it).

## 4. Network bandwidth

Run [all_reduce_bench.py](../../network/benchmarks/all_reduce_bench.py) with the same job (see [Is the bandwidth what it should be?](./network.md#3-is-the-bandwidth-what-it-should-be)) on 2 nodes, then on all the nodes you plan to use.

Good: the `busbw` is in line with the [expectations for this hardware](../../network/benchmarks/README.md#all_reduce-benchmark), and doesn't drop much going from 2 nodes to all nodes. If it does, bisect: run on pairs of nodes to find the slow node, NIC or link.

## 5. Storage

From a pod with the shared PVC mounted, measure the storage with the access patterns of your workload - see [fio](../../storage/README.md#fio) and [fio-scan](../../storage/fio-scan):

```bash
apt-get update && apt-get install -y fio
# fio-scan uses fio-json-extract.py to summarize the results, so get both
curl -sO https://raw.githubusercontent.com/stas00/ml-engineering/master/storage/fio-scan
curl -sO https://raw.githubusercontent.com/stas00/ml-engineering/master/storage/fio-json-extract.py
bash fio-scan /shared
```

Also check:
- the PVC can be mounted by pods on 2 different nodes at the same time (`ReadWriteMany`), and a file written on one is seen on the other
- how long writing a file of your checkpoint's size takes, and reading it back - this determines your checkpoint frequency, see [Frequent checkpoint saving](../../training/fault-tolerance/README.md#frequent-checkpoint-saving)
- how long it takes to load your model's weights from it

See [Data and Model Loading](./storage.md).

## 6. Fault tolerance drill

With a small model, run your actual training through [jobset-train.yaml](./jobset-train.yaml), and while it's running:

1. `kubectl exec POD -- touch /tmp/save-and-exit` - the job should save a checkpoint, exit, restart and resume from that checkpoint.
2. `kubectl delete pod POD` on one of the pods - the job should save (via the `preStop` hook), restart all pods and resume. Note that a direct deletion isn't a disruption in k8s's eyes (the pod gets no `DisruptionTarget` condition), so this restart counts towards `maxRestarts`, unlike a real preemption.
3. Check the loss curve across the restarts - it should continue smoothly, without jumps that indicate lost optimizer state or a changed data order.

See [Fault Tolerance on Kubernetes](./fault-tolerance.md). If you serve models, do the equivalent for inference: delete a replica under load and check that no requests fail, see [Updates and shutdowns](./inference.md#updates-and-shutdowns).

## 7. Observability

Before the first real run, make sure you'll be able to see what's going on:

- Can you see the GPU utilization, memory and temperature per GPU (typically DCGM exporter + Grafana - ask your admins for the dashboard)?
- Do the logs of a pod survive its deletion (a log collection system), or do you need to write them to the PVC yourself?
- Where do you see the events of a failed job a day later? (`kubectl get events` only keeps about an hour by default.)

## Record the baseline

Keep a table like this one per cluster, and re-run the checks after any driver, image or cluster change:

| Check | Date | Result | Notes |
| :---- | :--- | :----- | :---- |
| driver / CUDA / NCCL versions | | | |
| image pull time on a fresh node | | | |
| MAMF / MSMF bf16 TFLOPS (per-GPU min / median) | | | |
| all-reduce busbw, 2 nodes | | | |
| all-reduce busbw, N nodes | | | |
| storage: sequential read / write GB/s | | | |
| checkpoint save / load time | | | |
| model weights load time | | | |
| restart-to-resumed-training time | | | |
