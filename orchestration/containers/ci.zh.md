# 在 CI 中构建镜像：中文要点

> 英文原文：[Building Images in CI](./ci.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
一旦镜像不止一个人用，或者用于任何重要的工作，就应该由 CI 根据 git 里已提交的内容来构建，而不是在某人的笔记本上构建：笔记本上构建的镜像状态未知（未提交的修改、过期的基础镜像、错误的 CPU 架构），通过家庭网络上传要很久，作者休假时也没法重建。ML 镜像比 CI 工具通常面对的 web 服务镜像大得多、构建慢得多，这一章讲如何为它们搭建流水线。示例用 GitHub Actions，同样的思路也适用于 GitLab CI、Buildkite 等。

## 核心概念
- **流水线的 6 步**：① 每次 push 和 PR 都构建；② smoke test（所有包能否 import）；③ 用不可变 tag（git commit SHA）推送到 registry，但只从 `main` 和 release tag 推送；④ 在集群上用 GPU 跑一个简短的真实负载；⑤ promote：把 `stable` 这类 tag 指向通过 GPU 测试的镜像；⑥ 训练/推理 job 通过不可变 tag 或 digest 引用镜像。[build-image.yml](./build-image.yml) 实现了 1-3 步。
- **Registry cache**：BuildKit 把层缓存存到 registry 里，弥补 CI runner 每次都从空缓存开始的问题；`mode=max` 会缓存所有层，包括 multi-stage build 中间阶段的层。
- **不可变 tag vs 移动 tag**：完整 git SHA 做的 tag 永不覆盖；`main`、`stable`、`latest` 这类会移动的 tag 只方便人看，不要在 job 里用。
- **Digest**：tag 可以被覆盖，digest 不能，引用形式是 `$IMAGE@sha256:...`。
- **多架构 tag**：把各架构分别构建的镜像合并成一个 tag，node 拉取时自动得到适合自己架构的那个。
- **BuildKit secret mount**：构建时需要的 secret 只在那一条 `RUN` 指令中可用，不会留在镜像里。
- **ARC (Actions Runner Controller)**：把 GitHub runner 作为 pod 跑在集群上，包括申请 GPU 的 pod。

## 关键要点
- **磁盘空间**：GitHub 托管的标准 Linux runner 只有 14GB 可用磁盘，而典型 ML 镜像有 10-25GB，构建时还要放层、构建缓存和导出的镜像，于是报 `no space left on device`。办法：在 job 开头删除不需要的预装软件（[build-image.yml](./build-image.yml) 里的 "Free disk space" 步骤，通常能腾出几十 GB）、用 GitHub 的大规格 runner，或用 self-hosted runner。[详见](./ci.md#disk-space)
- **构建时间**：在 4 核 CI runner 上编译 CUDA 扩展（flash-attention、apex、DeepSpeed ops、自定义 kernel）可能要几个小时，或被 OOM-kill。先看有没有对应 python、torch、CUDA 版本组合的预编译 wheel；必须编译时，设置 `MAX_JOBS` 避免 OOM，`TORCH_CUDA_ARCH_LIST` 只包含实际用到的 GPU 架构（每多一个架构就要把所有 kernel 再编译一遍），并把扩展放进一个很少重建的单独基础镜像，频繁变化的应用镜像构建在它之上。[详见](./ci.md#build-time)
- **层缓存**：大镜像最好用 registry cache。GitHub 自带的缓存后端（`type=gha`）也能用，但一个 ML 镜像就很容易超出它的大小限制。`RUN --mount=type=cache,...` 的缓存不会存进 registry cache，所以在临时 runner 上没有用，只对持久的 builder 有效。[详见](./ci.md#layer-caching)
- **git SHA 要放在 `Dockerfile` 最后**：常见错误是在前面就把 git SHA 作为 build argument 传入，它每次提交都变，导致之后所有层每次都重建。放到最后后，训练脚本可以记录 `os.environ["GIT_SHA"]`，每次运行都知道用的是哪份代码。
- **CPU 架构**：GitHub 默认 runner 和大多数 GPU node 是 `x86_64`（`linux/amd64`），而 NVIDIA 基于 Grace 的系统（GH200、GB200、GB300）是 `arm64`。在 `x86_64` runner 上用 QEMU 模拟构建 `arm64` 可行，但凡要编译的东西都极慢。应在原生 runner（GitHub 有 `ubuntu-24.04-arm`）上分别构建、推送为各架构的 tag，再合并成一个多架构 tag。[详见](./ci.md#cpu-architecture)
- **tag 与可复现性**：每个镜像都打完整 git SHA tag 且永不覆盖（[build-image.yml](./build-image.yml) 用 `type=sha,format=long,prefix=`）；job 里不用移动 tag，否则 tag 移动后，重启的 pod 会拉到和 job 开始时不同的镜像，同一个 job 的 pod 也可能跑着不同的镜像；完全可复现用 digest；pin 住依赖，包括基础镜像（甚至按 digest），python 包用 lock 文件（如 `uv pip compile` 或 `pip-compile` 生成），再用 Dependabot 或 Renovate 提 PR 升级版本、由 CI 测试。用不可变 tag 时，k8s 默认的 `imagePullPolicy: IfNotPresent` 正好合适。[详见](./ci.md#tagging-and-reproducibility)
- **GPU 测试**：CI runner 没有 GPU，smoke test 只能发现打包问题（缺包、`pip check` 能查出的版本冲突、import 错误）；“镜像的 CUDA 对 node 的 driver 来说太新”“NCCL 找不到 InfiniBand 设备”这类问题只有在真实硬件上才会暴露。可以跑 [torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)（1 个节点，多节点镜像再跑 2 个节点）、用真实训练代码跑几步小模型看 loss 是否下降、推理镜像则用小模型起服务并发一个请求。GPU 时间贵，所以只在合并到 `main` 和发版时跑，并向管理员申请低优先级队列。[详见](./ci.md#testing-on-gpus)
- **从 CI 跑 GPU 测试的两种方式**：① CI 拿到集群凭据后提交一个使用新镜像的 k8s Job 并等待结果（优先用 OIDC 拿短期凭据，而不是把长期 kubeconfig 存成 CI secret）。Job 模板 `ci/gpu-test-job.yaml`（例如 [multi-node-job.yaml](../kubernetes/multi-node-job.yaml) 的副本）里用 `IMAGE_PLACEHOLDER` 代替镜像，并在 job 名字出现的每一处都用 `JOB_NAME` 代替：在 multi-node-job.yaml 里就是 Service 的名字和 selector、Job 的名字、`subdomain` 和 `--master-addr`；渲染成 `job.yaml` 后 apply，结束后用 `kubectl delete -f job.yaml` 同时删掉 Job 和 Service。② 用 ARC 把 runner 直接跑在集群的 GPU node 上。
- **secret 与安全**：不要 `COPY` 凭据进镜像，也不要通过 `ARG`/`ENV` 传入，它们会留在镜像的层或元数据里，任何能拉取镜像的人都能读到；用 Trivy 扫描漏洞（ML 基础镜像的告警多数与训练无关，先报告而不是让构建失败）；集群要求只运行可信镜像时用 cosign 签名。[详见](./ci.md#secrets-and-security)
- **registry 与拉取**：registry 放在离集群近的地方（同一云、同一区域）；设置保留策略；Docker Hub 有拉取限流，大集群共用一个 NAT IP 很快就会触发，要用云上的 pull-through cache 或把镜像同步到自己的 registry；私有 registry 需要在集群上配凭据，用只读 token 而不是个人 token。[详见](./ci.md#registry-and-pulling)

## 常用命令 / 配置
```yaml
# 把 BuildKit 层缓存存到 registry；mode=max 连中间阶段的层也缓存
cache-from: type=registry,ref=ghcr.io/my-org/train:buildcache
cache-to: type=registry,ref=ghcr.io/my-org/train:buildcache,mode=max
```

```dockerfile
# Dockerfile 最后：每次提交都会变的 git SHA 放在最末尾
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
```

```bash
docker buildx imagetools create -t $IMAGE:$SHA $IMAGE:$SHA-amd64 $IMAGE:$SHA-arm64   # 把两个架构合并成一个多架构 tag
docker buildx imagetools create --tag $IMAGE:stable $IMAGE:$SHA   # promote：在 registry 内复制 manifest，无需拉取镜像
kubectl get pod POD -o jsonpath='{.status.containerStatuses[*].imageID}'   # 查看运行中的 pod 实际用的镜像
# GPU 测试：渲染 Job 模板（/g 替换每一处）、提交、最后删除 Job 和 Service
JOB=gpu-test-${GITHUB_SHA::8}
sed -e "s|IMAGE_PLACEHOLDER|$IMAGE:$GITHUB_SHA|g" -e "s|JOB_NAME|$JOB|g" ci/gpu-test-job.yaml > job.yaml
kubectl apply -f job.yaml
kubectl delete -f job.yaml  # 同时删除 Job 和 Service
# 为私有 registry 创建 pull secret，并挂到 namespace 的 default ServiceAccount 上
kubectl create secret docker-registry regcred \
    --docker-server=ghcr.io --docker-username=USER --docker-password=TOKEN
kubectl patch serviceaccount default -p '{"imagePullSecrets": [{"name": "regcred"}]}'
```

## 常见坑
- 构建时报 `no space left on device` -> GitHub 托管 runner 只有 14GB 可用磁盘 -> 先删除预装软件，或换大规格 runner / self-hosted runner。
- 只改了代码，却每次都重建所有层 -> runner 是临时的、缓存为空，或 git SHA build argument 写在了 `Dockerfile` 前面 -> 用 registry cache（`mode=max`），把 `ARG GIT_SHA` 移到最后。
- 编译 CUDA 扩展花几个小时或被 OOM-kill -> 4 核 CI runner 太小 -> 优先用预编译 wheel，设置 `MAX_JOBS` 和 `TORCH_CUDA_ARCH_LIST`，把扩展放进单独的基础镜像。
- 同一个 job 的 pod 跑的镜像不一样 -> job 用了 `main`、`stable` 或 `latest` 这类移动 tag -> 改用完整 git SHA tag 或 digest。
- 大集群拉取 Docker Hub 镜像被限流 -> 整个集群共用一个 NAT IP -> 用云上的 pull-through cache，或把镜像同步到自己的 registry。
- `arm64` 镜像构建慢得离谱 -> 在 `x86_64` runner 上用 QEMU 模拟 -> 在原生 `arm64` runner 上构建，再合并成多架构 tag。

## 相关章节
- [Containers](./README.md)，尤其是 [Building your own image](./README.md#building-your-own-image)（`Dockerfile` 写法、层的顺序、`MAX_JOBS` 示例）
- [build-image.yml](./build-image.yml)：实现 1-3 步的完整 GitHub Actions workflow，复制到 `.github/workflows/` 后修改即可
- GPU 测试：[torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)、[multi-node-job.yaml](../kubernetes/multi-node-job.yaml)
- [Building on your own machines](./ci.md#building-on-your-own-machines)：self-hosted runner、`docker buildx create --driver kubernetes`、托管的远程 BuildKit 服务
- 外部：[Actions Runner Controller](https://github.com/actions/actions-runner-controller)、[Trivy](https://github.com/aquasecurity/trivy-action)、[cosign](https://github.com/sigstore/cosign)、[Dependabot](https://docs.github.com/en/code-security/dependabot)、[Renovate](https://github.com/renovatebot/renovate)
