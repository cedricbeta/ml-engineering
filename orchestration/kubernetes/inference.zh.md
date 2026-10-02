# Kubernetes 上的推理服务：中文要点

> 英文原文：[Inference on Kubernetes](./inference.md) - 这一页是要点总结，细节、完整示例和解释以原文为准。

## 这一章解决什么问题
推理服务是 k8s 最擅长的场景：长期运行的服务、滚动更新、负载均衡和自动扩缩容正是它设计要做的事。但 LLM 推理服务是很不寻常的 k8s 负载：启动要很多分钟，单个副本就需要好几块 GPU，空闲着也很贵。k8s 的大多数默认值是为小而启动快的 web 服务调的，这一章讲需要改哪些。示例用 vLLM，同样适用于 SGLang、TensorRT-LLM 等；推理本身的概念（prefill/decode、KV cache、batching）、性能指标和框架选择见 [Inference](../../inference/) 章节。完整示例见 [vllm-deployment.yaml](./vllm-deployment.yaml)。

## 核心概念
- **Deployment**：保持 N 个相同的服务副本运行，替换失败的副本，执行滚动更新。和训练 Job 不同，它的 pod 永远不会“完成”，服务因任何原因退出都会被 k8s 重启。
- **Service**：一个稳定的名字和端口，把请求负载均衡到处于 ready 状态的副本上。
- **PodDisruptionBudget**：在 node drain 和升级期间保证至少有部分副本在提供服务。
- **HPA / [KEDA](https://keda.sh/)**：根据负载调整副本数量。
- **[LeaderWorkerSet](https://lws.sigs.k8s.io/) (LWS)**：单个副本跨多个 node 时用来代替 Deployment，可以理解为“pod 组的 Deployment”。
- **Ingress / Gateway**：把 Service 暴露到集群外部。
- **三种 probe**：liveness probe 连续失败 `failureThreshold` 次就重启容器；readiness probe 失败期间 pod 被从 Service 中摘除，不接收流量；startup probe 成功之前，另外两种 probe 都不会运行。

## 关键要点
- **单节点副本**：GPU 数量必须等于 tensor parallelism 的度数；权重从预先准备好的共享存储加载并设置 `HF_HUB_OFFLINE=1`，而不是每次启动都从 HF hub 下载；挂载足够大的 `/dev/shm`，因为 tensor parallelism 的各 GPU worker 进程之间用共享内存通信；vLLM 会预先分配大部分显存给 KV cache（`--gpu-memory-utilization`，默认约 90%），所以不要在同一组 GPU 上放两个服务，模型小到不值一整块 GPU 时用 MIG 切分（问管理员有没有配置）。[详见](./inference.md#a-single-node-replica)
- **Health probe 是最常见的坑**：web 服务几秒钟就能启动，所以到处被复制的 liveness probe 常带着 `initialDelaySeconds: 30`。而 LLM 服务要加载几十到几百 GB 的权重，再编译并 capture CUDA graph，5-30 分钟很常见。用那样的 liveness probe，服务会在加载中途被杀、重启、再被杀，表现为 `CrashLoopBackOff`，而服务日志里没有任何错误。解决办法是给 startup probe 足够的预算，覆盖你预期最慢的启动。vLLM 和 SGLang 都提供 `/health`，模型加载完、引擎运行起来后才开始响应。[详见](./inference.md#health-probes---the-most-common-trap)
- **设置 probe 的 `timeoutSeconds`**：默认 1 秒对重负载下的服务太短，忙碌但健康的服务因 liveness probe 超时被重启，恰恰发生在最需要它的时候。
- **滚动更新**：Deployment 默认（`maxSurge: 25%` 向上取整，`maxUnavailable: 25%` 向下取整）在副本不多时会先启动新 pod，再终止旧 pod。如果 GPU 全被占着，新 pod 会一直 `Pending`，更新卡住。没有空闲 GPU 时用 `maxSurge: 0` 加 `maxUnavailable: 1`，一次替换一个副本，但更新期间容量会下降，要在低流量时段更新。[详见](./inference.md#updates-and-shutdowns)
- **优雅关闭**：pod 被删除时，“从 Service 中摘除”和“发送 `SIGTERM`”是并行发生的，所以几秒钟内请求仍可能到达一个已经在关闭的服务。标准做法是加一个只 sleep 的 `preStop` hook，让 endpoint 先更新，服务再收到信号。`terminationGracePeriodSeconds` 要覆盖 sleep 时间加上完成最长在途请求的时间，还要确认你用的服务版本收到 `SIGTERM` 时是完成正在运行的请求还是直接丢弃。PDB 设 `maxUnavailable: 1`，node drain 和升级每次最多下线一个副本，而不是一次全部下线；`minAvailable: 1` 只保证有一个副本在线，8 个副本时一次 drain 就可能下线 7 个。
- **多节点副本**（例如 bf16 的 405B 模型放不进单个 node）：每个副本是跨多个 node 的一组 pod，leader 提供 HTTP API，worker 运行模型的其余部分。LWS 给每组注入 `LWS_LEADER_ADDRESS`、`LWS_GROUP_SIZE` 和 `LWS_WORKER_INDEX`，分别作为 leader 地址、节点数和节点 rank 传给服务；`restartPolicy: RecreateGroupOnPodRestart` 时组内任一 pod 失败就重建整组，与多节点训练的“全有或全无”逻辑相同；Service 只选中 leader。这些 pod 和训练一样需要高速跨节点网络。[详见](./inference.md#multi-node-replicas)
- **冷启动**：从“需要多一个副本”到“它开始处理请求”，每个阶段都可能花几分钟：拿到 node（cluster autoscaler，没有容量时可能永远拿不到，对策是保留 warm node、预留容量）；拉镜像（10GB+ 的镜像要几分钟，对策是 registry 靠近集群、镜像缓存/streaming）；加载权重（取决于存储，从几秒到几十分钟，对策是快速共享存储、本地 NVMe 副本、`--load-format runai_streamer`）；预热（`torch.compile`、CUDA graph capture，大模型可能要几分钟，对策是持久化编译缓存，把 `VLLM_CACHE_ROOT` 放在 PVC 上，开发时用 `--enforce-eager`，启动快但服务慢）。用 `kubectl describe pod` 和服务日志测量每个阶段。[详见](./inference.md#cold-start)
- **自动扩缩容**：标准 HPA 按 CPU 利用率扩缩，这完全反映不了推理服务的负载，因为活都在 GPU 上。应该按反映用户体验的指标扩缩，例如排队请求数或 KV cache 使用率，vLLM 在 `/metrics` 中导出（如 `vllm:num_requests_waiting`、`vllm:kv_cache_usage_perc`，名字随版本变化，要查你所用版本的 `/metrics`），由 Prometheus 抓取后用 KEDA 驱动扩缩。扩容和冷启动一样慢，所以自动扩缩容只能应对缓慢的日常变化，应对不了突发峰值，`minReplicaCount` 要设得足够高；缩容要慢，给缩容设较长的 stabilization window（HPA 的 `behavior.scaleDown` 设置，用 KEDA 时写在 ScaledObject 的 `advanced.horizontalPodAutoscalerConfig.behavior` 下）；scale to zero（`minReplicaCount: 0`，KEDA 可以，原生 HPA 不行）能为很少用的模型省钱，但第一个请求要等完整的冷启动。[详见](./inference.md#autoscaling)
- **路由与压测**：Service 不了解副本的状态，而 LLM 请求的成本差异巨大，已经缓存了某个 prompt 前缀 KV cache 的副本也能回答得快得多；[Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) 及基于它的 [llm-d](https://github.com/llm-d/llm-d) 会按副本负载和前缀缓存路由。上线前要在集群内部用真实负载压测（排除你的机器到集群之间的网络；同一 namespace 内 `http://llm` 就能解析到 Service，其他 namespace 用 `http://llm.NAMESPACE.svc`），不加 `--request-rate` 时所有 prompt 会一次性发出，测到的是最大吞吐而不是用户在真实负载下的延迟，所以要在多个速率下跑，再把 TTFT、TPOT 和吞吐与需求对比。[详见](./inference.md#benchmarking-the-deployment)

## 常用命令 / 配置
```bash
kubectl port-forward svc/llm 8000:80   # 把 Service 转发到本机 8000 端口
curl localhost:8000/v1/models          # 列出服务提供的模型
# 在同一 namespace 里装有 vLLM 的 pod 中压测；--request-rate 设为真实负载的速率，并多试几个速率
vllm bench serve --base-url http://llm --model llama-70b \
    --tokenizer /shared/models/Llama-3.1-70B-Instruct \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 500 --request-rate 10
```

```yaml
# startup probe 给足启动时间；之后由 liveness probe 接手（注意 timeoutSeconds）
startupProbe:
  httpGet:
    path: /health
    port: http
  periodSeconds: 10
  failureThreshold: 180  # 180 x 10s = 最多 30 分钟启动时间
livenessProbe:
  httpGet:
    path: /health
    port: http
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 6
```

```yaml
# 优雅关闭：先 sleep，让 endpoint 更新后服务再收到 SIGTERM
terminationGracePeriodSeconds: 120
lifecycle:
  preStop:
    exec:
      command: ["sleep", "15"]
```

## 常见坑
- `CrashLoopBackOff`，服务日志里却没有任何错误 -> 照搬了 `initialDelaySeconds: 30` 的 liveness probe，加载权重时就被杀了 -> 加 startup probe，给足预算（如 `failureThreshold: 180` × `periodSeconds: 10` = 30 分钟）。
- 负载高时健康的服务反而被重启 -> probe 的 `timeoutSeconds` 默认只有 1 秒 -> 调大（示例中为 5）。
- 滚动更新卡住，新 pod 一直 `Pending` -> 默认先启动新 pod，但 GPU 已经全被占用 -> `maxSurge: 0` 加 `maxUnavailable: 1`。
- node drain 时一次下线了大部分副本 -> PDB 用的是 `minAvailable: 1`，只保证一个副本在线 -> 改用 `maxUnavailable: 1`。
- 删除或替换 pod 时有请求失败 -> 摘除 endpoint 和发送 `SIGTERM` 是并行的 -> `preStop` hook 先 sleep，并让 `terminationGracePeriodSeconds` 覆盖在途请求。
- HPA 不随推理负载扩容 -> 它按 CPU 利用率扩缩 -> 用 KEDA 按 vLLM 的排队请求数或 KV cache 使用率扩缩。

## 相关章节
- [Inference](../../inference/)：[Speeding up model loading time](../../inference/README.md#speeding-up-model-loading-time)、[Key inference performance metrics](../../inference/README.md#key-inference-performance-metrics)、[Benchmarks](../../inference/README.md#benchmarks)
- 示例：[vllm-deployment.yaml](./vllm-deployment.yaml)；多节点：[vLLM LWS example](https://github.com/kubernetes-sigs/lws/blob/main/docs/examples/leaderworkerset/basic/vllm.yaml)
- [Model weights](./storage.md#model-weights)、[Shared memory](./users.md#shared-memory)、[Keep the images small](../containers/README.md#keep-the-images-small)
- [Restart the whole job when any pod fails](./fault-tolerance.md#restart-the-whole-job-when-any-pod-fails)、[Fast Inter-node Networking](./network.md)
- 外部：[vLLM](https://github.com/vllm-project/vllm)、[KEDA](https://keda.sh/)、[LeaderWorkerSet](https://lws.sigs.k8s.io/)
