# Fast Inter-node Networking on Kubernetes

Multi-node training and inference need the backend network - InfiniBand, RoCE, AWS EFA or GCP's GPUDirect - to be available inside the pods. The trap is that when it isn't, nothing fails: NCCL quietly falls back to TCP sockets over the pod network, and the job runs, just several times to an order of magnitude slower. So on a new cluster, or with a new image, always verify that the fast network is actually used.

For the networking background see the [Network](../../network/) chapter, and for the general NCCL debug techniques see [Network debug](../../network/debug/).

## Why it's harder on k8s

By default a pod gets a single virtual Ethernet interface (`eth0`) connected to the cluster's pod network. Everything else on the node - the RDMA devices under `/dev/infiniband`, the extra NICs, the kernel modules - is invisible to it unless the cluster explicitly exposes it, which requires several components that the admins install:

- a device plugin (or a DRA driver) that exposes the RDMA devices as a schedulable resource, e.g., the [NVIDIA Network Operator](https://github.com/Mellanox/network-operator) for IB/RoCE, or the EFA device plugin on AWS
- for RoCE/SR-IOV setups, a secondary network attached to the pod via [Multus](https://github.com/k8snetworkplumbingwg/multus-cni), so the pod gets additional interfaces (`net1`, `net2`, ...)
- for GPUDirect RDMA (the NIC reading and writing GPU memory directly), the `nvidia-peermem` kernel module or DMA-BUF support on the node
- the cloud provider's NCCL plugin when it's needed (AWS's `aws-ofi-nccl`, GCP's plugins for GPUDirect-TCPXO and RDMA)

Your pod then has to ask for all of this, which is very cluster-specific - so this chapter gives you a checklist and the means to verify, but you'll need the exact resource names and annotations from your admins or your cloud provider's docs.

## What a pod needs

A pod spec that is likely to work, with the cluster-specific parts marked:

```yaml
metadata:
  annotations:
    # cluster-specific: only on clusters that use Multus secondary networks
    k8s.v1.cni.cncf.io/networks: rdma-net-1,rdma-net-2,...
spec:
  containers:
  - name: trainer
    resources:
      limits:
        nvidia.com/gpu: 8
        # cluster-specific, examples:
        # rdma/rdma_shared_device_a: 1        # NVIDIA Network Operator's shared RDMA device plugin
        # vpc.amazonaws.com/efa: 32           # AWS EFA on p5.48xlarge
        # hugepages-2Mi: 5120Mi               # AWS EFA requires huge pages too
    securityContext:
      capabilities:
        add: ["IPC_LOCK"]  # RDMA needs to pin memory
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
```

Plus the image must contain the user-space libraries for the fabric:

- IB/RoCE: `rdma-core` (`libibverbs`, `librdmacm` and the `libmlx5` provider) - included in the NGC images
- AWS EFA: `libfabric` and the `aws-ofi-nccl` NCCL plugin - installed by the [EFA installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html), or use AWS's Deep Learning Containers
- GCP: the NCCL plugin and the NCCL env settings provided by Google for the machine type - see [GKE GPU networking](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx)

Some clusters instead run the training pods with `hostNetwork: true`, which gives the pod all of the node's interfaces. It's simpler, but it gives up the network isolation between pods and causes port conflicts if 2 pods run on the same node, so it's usually only allowed for whole-node jobs.

Always request all GPUs and all NICs of a node for multi-node training. A pod with 4 GPUs may get GPUs and NICs on different PCIe switches or NUMA nodes, which hurts performance. `nvidia-smi topo -m` inside the pod shows which NICs are close to which GPUs.

Cloud provider guides:
- AWS: [EFA on EKS](https://docs.aws.amazon.com/eks/latest/userguide/node-efa.html)
- GCP: [GPUDirect on GKE](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpu-bandwidth-gpudirect-tcpx) and the related RDMA pages
- NVIDIA: [Network Operator](https://github.com/Mellanox/network-operator) for self-managed IB/RoCE clusters

## Verifying step by step

Start a [dev pod](./dev-pod.yaml) with the networking settings from above, and check from the bottom up.

### 1. Does the pod see the devices?

```bash
ls /dev/infiniband            # IB/RoCE/EFA: expect uverbs0, uverbs1, ...
ibv_devices                   # if `ibverbs-utils` is installed: lists mlx5_0 ... or efa devices
ibv_devinfo | grep -E "hca_id|state|link_layer"
ulimit -l                     # should be `unlimited` - otherwise memory registration fails
```

If `/dev/infiniband` doesn't exist, the pod didn't get the RDMA resource - check the pod's `resources` and that the node advertises it: `kubectl describe node NODE | grep -A15 Allocatable`.

On AWS EFA: `fi_info -p efa` (in `/opt/amazon/efa/bin/`) should list the EFA devices.

### 2. Does NCCL use them?

Run a 2-node job, e.g., [multi-node-job.yaml](./multi-node-job.yaml) (it already sets `NCCL_DEBUG=INFO`), and look at the NCCL init logs:

```bash
kubectl logs gpu-test-0-xxxxx | grep -E "NET/|via NET"
```

What you want to see (also see the [explanations in the Network debug chapter](../../network/debug/README.md#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues)):

```
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB ... [7]mlx5_7:1/IB [RO]; OOB eth0:10.4.2.17<0>
NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA
```

- `NET/IB` - the RDMA transport is used (for RoCE the devices are listed with `/RoCE`). On AWS you should see `NET/OFI` and the `efa` provider instead, and on GCP the name of Google's plugin.
- one device per GPU - if only some of the NICs are listed, only part of the bandwidth will be used.
- `GDRDMA` - GPUDirect RDMA is used, i.e., the data goes from GPU memory straight to the NIC. Without it the data is staged through CPU memory, which is slower.
- `OOB eth0` - the bootstrap (out-of-band) connection goes through the pod network, which is fine.

What indicates the fast network isn't used:

```
NCCL INFO NET/Socket : Using [0]eth0:10.4.2.17<0>
```

This means NCCL found no RDMA devices and fell back to TCP over the pod network.

If the pod has extra interfaces (from Multus), NCCL may pick the wrong one for the bootstrap, which can result in a hang at init. Then pin it explicitly with `NCCL_SOCKET_IFNAME=eth0`, see [`NCCL_SOCKET_IFNAME`](../../network/debug/README.md#nccl_socket_ifname).

### 3. Is the bandwidth what it should be?

The logs can look right while the performance is still bad (e.g., misconfigured switches or adaptive routing), so measure. The easiest way is to run [all_reduce_bench.py](../../network/benchmarks/all_reduce_bench.py) using the same [multi-node-job.yaml](./multi-node-job.yaml) - just replace the script it downloads and runs:

```bash
curl -sO https://raw.githubusercontent.com/stas00/ml-engineering/master/network/benchmarks/all_reduce_bench.py
exec torchrun ... all_reduce_bench.py
```

and compare the `busbw` you get with the expectations in [Network benchmarks](../../network/benchmarks/README.md#all_reduce-benchmark). A TCP fallback typically shows up as an inter-node `busbw` an order of magnitude below the hardware's capability.

If you prefer [nccl-tests](https://github.com/NVIDIA/nccl-tests), on k8s it's usually run via the [MPI Operator](https://github.com/kubeflow/mpi-operator) - the cloud providers' docs linked above have ready to use examples.

Run the benchmark on 2 nodes first, then on all the nodes you plan to use - a single node with a bad NIC or cable will slow down the whole job, see [network debug](../../network/debug/).

## Pod placement

How close the nodes are to each other in the network topology matters too: nodes under the same leaf switch communicate with fewer hops than nodes in different parts of the fabric. Plain k8s scheduling knows nothing about the topology, so it may spread your pods over the whole data center. Ways to deal with it:

- [Kueue's Topology Aware Scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/) places the pods of a job as close as possible, given node labels that describe the topology.
- cloud providers have their own mechanisms, e.g., compact placement policies on GCP, and placement groups and capacity blocks on AWS.

Ask your admins what's available.
