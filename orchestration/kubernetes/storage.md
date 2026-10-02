# Data and Model Loading on Kubernetes

On a SLURM cluster you typically have a shared file system mounted on every node, and you just read from `/data`. On k8s nothing is mounted unless your pod spec asks for it, and the pod's own filesystem disappears with the pod. This chapter covers where to put the datasets, the model weights and the checkpoints, and how to get them to the GPUs fast.

For the file system background (which file systems to choose, IO concepts, benchmarking) see the [Storage](../../storage/) chapter.

## The options

| Where | Survives the pod? | Shared between nodes? | Speed | Good for |
| :---- | :---------------- | :-------------------- | :---- | :------- |
| the container image | n/a (read-only) | yes (each node pulls it) | fast once pulled | code and python packages only |
| the container's own filesystem | no | no | node's disk | nothing important |
| `emptyDir` on the node's disk | no | no | node's disk, NVMe if the node has it | staging a local copy of data or weights, scratch |
| `emptyDir` with `medium: Memory` | no | no | RAM | `/dev/shm` |
| `hostPath` | yes, stays on the node | no | node's disk | node-local caches (often forbidden by the cluster's policy) |
| PVC on a shared file system (NFS, Lustre, Weka, VAST, GPFS, ...) | yes | yes, if `ReadWriteMany` | depends on the file system | datasets, weights, checkpoints |
| PVC on a block device (EBS, Persistent Disk, ...) | yes | read-write: no (`ReadWriteOnce`), some support read-only sharing (`ReadOnlyMany`) | fast, but one node | single-node work |
| object storage (S3, GCS, Azure Blob) via a FUSE CSI driver or directly from code | yes | yes | high throughput for large sequential reads, slow for small/random ones | datasets in large shards, weights, checkpoint offload |

## Finding out what you have

```bash
kubectl get storageclass   # the kinds of storage you can create new volumes from
kubectl get pvc            # the volumes that already exist in your namespace
kubectl describe pvc NAME  # capacity, access modes and the storage class behind it
```

The access mode is the first thing to check: multi-node training needs all pods to see the same files, which requires `ReadWriteMany` (`RWX`). A `ReadWriteOnce` (`RWO`) volume can only be mounted by pods on a single node - a pod on another node will be stuck in `ContainerCreating` with a `Multi-Attach error` event.

To create a new shared volume, if a storage class with RWX support exists:

```yaml
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

Then mount it in the pod spec:

```yaml
spec:
  containers:
  - name: trainer
    volumeMounts:
    - name: shared
      mountPath: /shared
  volumes:
  - name: shared
    persistentVolumeClaim:
      claimName: my-shared-pvc
```

Before relying on a volume, measure it from inside a pod - the same storage class can perform very differently from what's on the label. Use [fio](../../storage/README.md#fio) and the [usability perception benchmarks](../../storage/README.md#usability-perception-io-benchmarks) from the Storage chapter.

### Permissions

If your image runs as a non-root user, you may get `Permission denied` when writing to a freshly created volume. Setting `fsGroup` makes k8s give your group write access to the mounted volumes:

```yaml
spec:
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
```

On a volume with millions of files changing the ownership on every mount can take a very long time - add `fsGroupChangePolicy: OnRootMismatch` to only do it when needed.

## Datasets

In order of preference:

1. **A shared fast file system** (RWX PVC) with the dataset pre-staged on it. This is the closest to the SLURM experience, and the data is read directly by all nodes.
2. **A local copy on the node's NVMe**, made at pod start by an init container. Great when the dataset fits on the local disk and the shared storage is slow:

   ```yaml
   spec:
     initContainers:
     - name: stage-data
       image: amazon/aws-cli
       command: ["aws", "s3", "sync", "s3://my-bucket/dataset/", "/local-data/"]
       volumeMounts:
       - name: local-data
         mountPath: /local-data
     containers:
     - name: trainer
       resources:
         requests:
           ephemeral-storage: 2Ti  # so the pod lands on a node with enough local disk and isn't evicted
       volumeMounts:
       - name: local-data
         mountPath: /local-data
     volumes:
     - name: local-data
       emptyDir: {}
   ```

   Whether `emptyDir` ends up on a fast local NVMe depends on how the nodes are configured (e.g., GKE needs local SSDs to be set up as ephemeral storage, AWS instance store NVMe needs to be formatted and mounted by the node's bootstrap) - ask your admins. Note that the copy is lost when the pod is deleted, so every restart pays the copying time again.

3. **Streaming from object storage** with a data loader built for it, e.g., [webdataset](https://github.com/webdataset/webdataset), [MosaicML streaming](https://github.com/mosaicml/streaming), or HF `datasets` with `streaming=True`. This works well only if the dataset is stored in large shards that are read sequentially, and the loader can resume at the exact position after a restart - see [Local storage beats cloud storage](../../storage/README.md#local-storage-beats-cloud-storage) for the pitfalls.
4. **Object storage mounted via FUSE** ([Mountpoint for S3](https://github.com/awslabs/mountpoint-s3-csi-driver), [Cloud Storage FUSE](https://github.com/GoogleCloudPlatform/gcs-fuse-csi-driver), [BlobFuse](https://github.com/Azure/azure-storage-fuse)). It looks like a normal directory, but random access and `mmap` - which many dataset formats rely on - are very slow on it, see [mmap vs sequential dataset reads](../../storage/README.md#mmap-vs-sequential-dataset-reads). Use it for sequential reads only.

Whatever you pick, put the preprocessed dataset cache on persistent storage too (e.g., `HF_DATASETS_CACHE`), otherwise every restart re-tokenizes the dataset.

## Model weights

A 70B model in bf16 is ~140GB. If each of 64 pods downloads it from the HF hub at start, that's 9TB of traffic, you may get rate-limited, and the start time of the job depends on the slowest download. So:

1. **Download once, to shared storage**, with a separate one-time Job:

   ```yaml
   apiVersion: batch/v1
   kind: Job
   metadata:
     name: download-model
   spec:
     backoffLimit: 3
     template:
       spec:
         restartPolicy: Never
         containers:
         - name: download
           image: python:3.12-slim
           command: ["bash", "-c"]
           args:
           - |
             pip install -U huggingface_hub
             hf download meta-llama/Llama-3.1-70B-Instruct --local-dir /shared/models/Llama-3.1-70B-Instruct
           envFrom:
           - secretRef:
               name: my-tokens   # HF_TOKEN, see users.md#secrets
           volumeMounts:
           - name: shared
             mountPath: /shared
         volumes:
         - name: shared
           persistentVolumeClaim:
             claimName: my-shared-pvc
   ```

2. **Load from the local path in the training/inference pods** (`from_pretrained("/shared/models/Llama-3.1-70B-Instruct")`) and set `HF_HUB_OFFLINE=1`, so that no pod ever reaches out to the hub - a hub outage or rate limit then can't break your job.

3. **If loading is still slow**, it's usually because the shared storage is slow at the access pattern of the loader. Options:
   - copy the weights to the node's local NVMe first (an init container, as with the datasets above) - a sequential copy is often much faster than the loader's reads.
   - for vLLM, stream directly from object storage with `--load-format runai_streamer`, which reads with high concurrency.
   - k8s 1.36+ can mount an OCI image as a read-only volume ([image volumes](https://kubernetes.io/docs/tasks/configure-pod-container/image-volumes/)), so the weights can be packaged as an OCI artifact and get cached on the nodes just like container images - without bloating the image with the code.

Do not bake the weights into the container image, see [Keep the images small](../containers/README.md#keep-the-images-small).

## Checkpoints

- Write them to a `ReadWriteMany` PVC that all ranks can write to concurrently, or directly to object storage (e.g., PyTorch DCP with an S3 backend such as [s3torchconnector](https://github.com/awslabs/s3-connector-for-pytorch)).
- Saving speed determines how often you can checkpoint and how much a failure costs you, see [Frequent checkpoint saving](../../training/fault-tolerance/README.md#frequent-checkpoint-saving). Async checkpointing (e.g., `torch.distributed.checkpoint.async_save`) takes the saving off the critical path.
- Keep the last few checkpoints on the fast storage for a quick resume, and offload older ones to object storage with a separate job or CronJob.
- Make them atomic, see [Checkpoint and resume](./fault-tolerance.md#checkpoint-and-resume).

## Caches that make restarts faster

(For going further and saving the whole initialized process, see [Snapshots](./snapshots.md).)

On k8s pods get restarted a lot, and each restart starts with an empty filesystem. Several caches that are taken for granted on a normal machine are then rebuilt from scratch every time - which can add many minutes to every restart. Point them to persistent storage:

```yaml
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

If many pods write to the same cache concurrently and you see weird errors, give each node its own sub-directory, or keep the compile caches on the node's local disk. See also [Share caches in group environments](../../storage/README.md#share-caches-in-group-environments).
