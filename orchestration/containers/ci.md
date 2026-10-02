# Building Images in CI

Once more than one person uses an image, or it's used for anything that matters, it should be built by CI from what's committed to git and not on somebody's laptop. A laptop-built image has an unknown state (uncommitted changes, a stale base image, the wrong CPU architecture), takes forever to upload over a home connection, and can't be rebuilt when its author is on vacation.

This chapter covers how to set up such a pipeline for ML images, which are much bigger and slower to build than the typical web service images that CI tools are designed for. The examples use GitHub Actions, but the same ideas apply to GitLab CI, Buildkite, etc. To learn how to write the `Dockerfile` itself see [Building your own image](./README.md#building-your-own-image).

## The pipeline

1. **Build** on every push and pull request.
2. **Smoke test** the image - can all the packages be imported?
3. **Push** to the registry with an immutable tag (the git commit SHA), but only from `main` and release tags.
4. **Test on GPUs** - run a short real workload on the cluster.
5. **Promote** - move a tag like `stable` to the image that passed the GPU test.
6. Your training/inference jobs refer to the image by its immutable tag or digest.

A complete workflow implementing steps 1-3 is [build-image.yml](./build-image.yml) - copy it to `.github/workflows/` in your repo and adapt it. The rest of this chapter explains the parts that are specific to ML images, and adds steps 4-6.

## Problems specific to ML images

### Disk space

The standard GitHub-hosted Linux runners come with only 14GB of free disk space. A typical ML image is 10-25GB, and while building it the space is needed for the image layers, the build cache and the exported image, so you get `no space left on device`. Options:

- delete the preinstalled software you don't need at the start of the job, as the "Free disk space" step in [build-image.yml](./build-image.yml) does - this typically frees a few dozen GBs.
- use GitHub's larger runners, which come with more disk, CPUs and RAM.
- use self-hosted runners (see [Building on your own machines](#building-on-your-own-machines)).

### Build time

Compiling CUDA extensions (flash-attention, apex, DeepSpeed ops, custom kernels) on a 4-core CI runner can take hours, or get OOM-killed. Before compiling anything, check whether a pre-built wheel exists for your exact combination of python, torch and CUDA versions - e.g., flash-attention publishes many of them in its GitHub releases. If you have to compile:

- set `MAX_JOBS` to avoid the OOM, see the [Dockerfile example](./README.md#building-your-own-image).
- set `TORCH_CUDA_ARCH_LIST` to only the GPU architectures you actually use - every additional architecture means compiling all the kernels again.
- build the extensions in a separate base image that is rebuilt rarely (e.g., when the torch version changes), and build the frequently changing application image on top of it.

### Layer caching

CI runners are ephemeral, so every job starts with an empty Docker cache and rebuilds everything from scratch - unless you tell BuildKit to store the cache elsewhere. The best option for big images is the registry cache:

```yaml
cache-from: type=registry,ref=ghcr.io/my-org/train:buildcache
cache-to: type=registry,ref=ghcr.io/my-org/train:buildcache,mode=max
```

`mode=max` caches all layers, including those of intermediate stages in multi-stage builds. With it, a commit that only changes your code rebuilds only the last layer. GitHub's own cache backend (`type=gha`) also works, but its size limit is easily exceeded by a single ML image.

Note that `RUN --mount=type=cache,...` cache mounts (e.g., for the pip cache) are not stored in the registry cache, so on ephemeral runners they don't help - they are only useful on persistent builders.

Just as important is the [order of the layers](./README.md#building-your-own-image) - put whatever changes most often last. A common mistake is passing the git SHA as a build argument early in the `Dockerfile`: since its value changes on every commit, all the layers after it get rebuilt every time. Declare it at the very end:

```dockerfile
# ... all the heavy layers above ...
COPY . /workspace

# last, since it changes with every commit
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
```

Now your training script can log `os.environ["GIT_SHA"]`, so every run records exactly which code it ran with.

### CPU architecture

GitHub's default runners are `x86_64` (`linux/amd64`), and so are most GPU nodes. But NVIDIA's Grace-based systems (GH200, GB200, GB300) are `arm64`. Building `arm64` images on `x86_64` runners via QEMU emulation works, but is extremely slow for anything compiled. Instead, build each architecture on a native runner (GitHub has `ubuntu-24.04-arm` runners), push them under per-architecture tags, and combine them into a single multi-architecture tag:

```bash
docker buildx imagetools create -t $IMAGE:$SHA $IMAGE:$SHA-amd64 $IMAGE:$SHA-arm64
```

Nodes pulling `$IMAGE:$SHA` then automatically get the right one.

## Tagging and reproducibility

- **Tag every image with the full git SHA**, and never overwrite such a tag. Then for every image you can find the code it was built from, and for every commit its image. [build-image.yml](./build-image.yml) does this with `type=sha,format=long,prefix=`.
- **Moving tags** like `main` or `stable` are a convenience for humans, but don't use them in jobs - when the tag moves, a restarted pod will pull a different image than the one the job started with, and the pods of the same job may end up running different images. The same is true for `latest`.
- **For complete reproducibility use the digest.** A tag can be overwritten, a digest can't. `docker buildx imagetools inspect $IMAGE:$SHA` shows the digest, which you can use as `$IMAGE@sha256:...`. To see which image a running pod actually uses:

```bash
kubectl get pod POD -o jsonpath='{.status.containerStatuses[*].imageID}'
```

- **Pin the dependencies.** The image is only as reproducible as its inputs. Pin the base image (`FROM nvcr.io/nvidia/pytorch:25.08-py3`, or even by digest), and use a lock file for the python packages (e.g., generated with `uv pip compile` or `pip-compile`) - a `requirements.txt` with `transformers>=4.0` will produce a different image every week. Tools like [Dependabot](https://docs.github.com/en/code-security/dependabot) or [Renovate](https://github.com/renovatebot/renovate) can then open PRs that bump the pinned versions, and the CI tests the result.

With immutable tags, k8s's default `imagePullPolicy: IfNotPresent` is what you want - a node pulls each image once and then uses its local copy.

## Testing on GPUs

The CI runners have no GPUs, so the smoke test can only catch packaging problems - a missing package, a version conflict (`pip check`), an import error. Problems like "the CUDA version in the image is too new for the driver on our nodes" or "NCCL doesn't find the InfiniBand devices" only show up on the real hardware.

So, for images that are about to be used for real work, also run a short test on the cluster. What to run:

- [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py) on 1 node, and if the image is meant for multi-node work on 2 nodes, see [multi-node-job.yaml](../kubernetes/multi-node-job.yaml).
- a few training steps of a tiny model with your actual training code, checking that the loss goes down.
- for inference images, start the server with a small model and send a request.

How to run it from CI:

1. **Submit a k8s Job from the CI** - the CI gets credentials to the cluster, applies a Job with the new image, and waits for it. Here `ci/gpu-test-job.yaml` is your Job template (e.g., a copy of [multi-node-job.yaml](../kubernetes/multi-node-job.yaml)) with `IMAGE_PLACEHOLDER` in place of the image and `JOB_NAME` in place of the job name everywhere it appears - in [multi-node-job.yaml](../kubernetes/multi-node-job.yaml) that's the Service's name and selector, the Job's name, the `subdomain` and the `--master-addr`. Prefer short-lived credentials via OIDC (e.g., [aws-actions/configure-aws-credentials](https://github.com/aws-actions/configure-aws-credentials) or [google-github-actions/auth](https://github.com/google-github-actions/auth)) over storing a long-lived kubeconfig as a CI secret.

```bash
JOB=gpu-test-${GITHUB_SHA::8}
sed -e "s|IMAGE_PLACEHOLDER|$IMAGE:$GITHUB_SHA|g" -e "s|JOB_NAME|$JOB|g" ci/gpu-test-job.yaml > job.yaml
kubectl apply -f job.yaml
# `kubectl wait` can wait for only one condition, so poll for either success or failure
for i in $(seq 180); do
    state=$(kubectl get job $JOB -o jsonpath='{.status.conditions[?(@.status=="True")].type}')
    if echo "$state" | grep -qw Complete; then break; fi
    if echo "$state" | grep -qw Failed; then kubectl logs job/$JOB --tail=200; exit 1; fi
    sleep 10
done
echo "$state" | grep -qw Complete || { echo "timed out"; exit 1; }
kubectl logs job/$JOB --tail=50
kubectl delete -f job.yaml  # the Job and the Service
```

2. **Run the CI runners on the cluster itself** with [Actions Runner Controller](https://github.com/actions/actions-runner-controller) (ARC), which runs GitHub runners as pods - including pods that request GPUs. Then a workflow job with `runs-on: <your GPU runner set>` runs directly on a GPU node.

GPU time is expensive and the cluster may be busy with training, so run these tests only on merges to `main` and on releases, not on every PR commit - and ask your admins for a low-priority queue for them.

Once the test passes, promote the image by pointing a moving tag at it - this copies the manifest within the registry without pulling the image:

```bash
docker buildx imagetools create --tag $IMAGE:stable $IMAGE:$SHA
```

## Secrets and security

- **Never `COPY` credentials into the image or pass them via `ARG`/`ENV`** - they remain in the image's layers or metadata, and anybody who can pull the image can read them. If the build itself needs a secret, e.g., a token for a private package index, use a BuildKit secret mount, which is available only during that one `RUN` instruction:

```dockerfile
RUN --mount=type=secret,id=pip_index_url \
    PIP_INDEX_URL=$(cat /run/secrets/pip_index_url) pip install --no-cache-dir -r requirements.txt
```

```yaml
- uses: docker/build-push-action@v7
  with:
    secrets: |
      pip_index_url=${{ secrets.PIP_INDEX_URL }}
```

- **Scan the images for known vulnerabilities**, e.g., with [Trivy](https://github.com/aquasecurity/trivy-action). ML base images tend to have many findings, most of them irrelevant to a training job, so start by reporting rather than failing the build, and see what your security team requires.
- **Sign the images** (e.g., with [cosign](https://github.com/sigstore/cosign)) if your cluster enforces that only trusted images may run.

## Registry and pulling

- **Put the registry close to the cluster** (same cloud and region). A 20GB image pulled by 100 nodes is 2TB of traffic, and cross-region or Internet traffic is both slower and billed.
- **Set up retention policies** (ECR lifecycle policies, Artifact Registry cleanup policies, etc.). An image per commit at 20GB each adds up quickly. Since all images built on the same base share its layers, the actual storage is less than the sum of the image sizes - but it still grows.
- **Docker Hub rate-limits pulls**, and a big cluster behind a single NAT IP address hits the limits quickly. Use your cloud's pull-through cache (e.g., ECR pull through cache rules, Artifact Registry remote repositories) or mirror the images you need into your own registry.
- **Private registries need credentials on the cluster.** On the managed k8s services the nodes usually get access to the same cloud's registry via IAM. For any other registry, create a pull secret:

```bash
kubectl create secret docker-registry regcred \
    --docker-server=ghcr.io --docker-username=USER --docker-password=TOKEN
```

and reference it in the pod spec:

```yaml
spec:
  imagePullSecrets:
  - name: regcred
```

or attach it to the namespace's default ServiceAccount so that all pods use it:

```bash
kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "regcred"}]}'
```

Use a token with read-only access for this, not your personal token.

## Building on your own machines

When the hosted runners are too small or too slow, you can:

- run self-hosted CI runners on big CPU machines with large disks - their local Docker cache persists between builds, so the cache mounts work as well.
- build on the k8s cluster: `docker buildx create --driver kubernetes` runs BuildKit as pods on the cluster's CPU nodes, close to the registry.
- use a hosted remote BuildKit service with a persistent cache.

Whichever you choose, keep the build defined by the files in git (`Dockerfile`, lock files, the CI workflow), so that anybody can reproduce the image.
