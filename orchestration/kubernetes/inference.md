# Inference on Kubernetes

Serving is where k8s is at its strongest - long-running servers, rolling updates, load balancing and autoscaling are exactly what it was built for. But LLM inference servers are unusual k8s workloads: they take many minutes to start, a single replica needs several GPUs, and they are expensive to keep idle. Most of the k8s defaults are tuned for small, fast-starting web services, and this chapter covers what to change.

For the inference concepts (prefill/decode, KV cache, batching), the performance metrics and how to choose a framework, see the [Inference](../../inference/) chapter. The examples here use [vLLM](https://github.com/vllm-project/vllm), but the same applies to SGLang, TensorRT-LLM, etc.

A complete example is [vllm-deployment.yaml](./vllm-deployment.yaml).

## The building blocks

| Object | Role |
| :----- | :--- |
| Deployment | keeps N identical replicas of the server running, replaces failed ones, performs rolling updates |
| Service | a stable name and port that load-balances the requests over the ready replicas |
| PodDisruptionBudget | keeps at least some replicas serving during node drains and upgrades |
| HPA / [KEDA](https://keda.sh/) | changes the number of replicas based on the load |
| [LeaderWorkerSet](https://lws.sigs.k8s.io/) | instead of a Deployment, when a single replica spans multiple nodes |
| Ingress / Gateway | exposes the Service outside the cluster |

Unlike training Jobs, a Deployment's pods are never "done" - if the server exits for any reason, k8s restarts it.

## A single-node replica

The essential parts of [vllm-deployment.yaml](./vllm-deployment.yaml):

```yaml
containers:
- name: vllm
  image: vllm/vllm-openai:v0.28.0
  command: ["vllm", "serve"]
  args:
  - /shared/models/Llama-3.1-70B-Instruct
  - --served-model-name=llama-70b
  - --tensor-parallel-size=4
  - --port=8000
  resources:
    limits:
      nvidia.com/gpu: 4
```

- The number of GPUs must match the tensor parallelism degree.
- Load the weights from shared storage that was populated once in advance, and set `HF_HUB_OFFLINE=1` - not from the HF hub on every start, see [Model weights](./storage.md#model-weights).
- Mount a large `/dev/shm` - tensor parallelism uses shared memory between the GPU worker processes, see [Shared memory](./users.md#shared-memory).
- vLLM pre-allocates most of the GPU memory for the KV cache (`--gpu-memory-utilization`, about 90% by default), so don't try to fit 2 servers onto the same GPUs. If a model is too small to justify a whole GPU, use MIG to split the GPU (ask your admins whether it's set up).

Test it from your machine:

```bash
kubectl port-forward svc/llm 8000:80
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "llama-70b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Health probes - the most common trap

k8s has 3 kinds of probes, and the combination matters a lot for servers that take long to start:

- **liveness probe** - if it fails `failureThreshold` times in a row, the container gets restarted
- **readiness probe** - while it fails, the pod is removed from the Service and gets no traffic
- **startup probe** - until it succeeds, the other 2 probes aren't run

A typical web service starts in seconds, so it's common to see a liveness probe with `initialDelaySeconds: 30` copied around. An LLM server needs to load tens or hundreds of GBs of weights, then compile and capture CUDA graphs - easily 5-30 minutes. With such a liveness probe the server gets killed in the middle of loading, restarts, gets killed again - a `CrashLoopBackOff` without any error in the server's logs.

The fix is a startup probe with enough budget for the slowest start you expect:

```yaml
startupProbe:
  httpGet:
    path: /health
    port: http
  periodSeconds: 10
  failureThreshold: 180  # 180 x 10s = up to 30 minutes to start
livenessProbe:
  httpGet:
    path: /health
    port: http
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 6
```

Also set the probes' `timeoutSeconds` - the default of 1 second is too short for a server under heavy load, and a liveness probe timing out on a busy (but healthy) server restarts it exactly when it's needed most.

vLLM and SGLang both serve a `/health` endpoint, which starts responding once the model is loaded and the engine is running.

## Updates and shutdowns

**Rolling updates.** By default (`maxSurge: 25%` rounded up, `maxUnavailable: 25%` rounded down) a Deployment with a few replicas first starts a new pod and only then terminates an old one. If all GPUs are taken, the new pod stays `Pending` forever and the update gets stuck. Without spare GPUs, use `maxSurge: 0` and `maxUnavailable: 1` - replace one replica at a time - but then the capacity is reduced during the update, so do updates during low traffic.

**Graceful shutdown.** When a pod is deleted, removing it from the Service and sending it `SIGTERM` happen in parallel, so for a few seconds requests may still arrive at a server that is already shutting down. The standard remedy is a `preStop` hook that just sleeps, so the endpoints get updated before the server gets the signal:

```yaml
terminationGracePeriodSeconds: 120
lifecycle:
  preStop:
    exec:
      command: ["sleep", "15"]
```

Then make sure `terminationGracePeriodSeconds` covers the sleep plus the time to finish the longest in-flight requests, and check how your server version behaves on `SIGTERM` - whether it finishes the running requests or drops them.

**PodDisruptionBudget.** With `maxUnavailable: 1`, node drains and upgrades take down at most one replica at a time instead of all at once. (`minAvailable: 1` would only guarantee that one replica stays up - with 8 replicas, a drain could take down 7 at once.)

## Multi-node replicas

When a model doesn't fit onto a single node (e.g., a 405B model in bf16), each replica becomes a group of pods on several nodes: the leader serves the HTTP API, and the workers run the rest of the model. [LeaderWorkerSet](https://lws.sigs.k8s.io/) (LWS) is the k8s API for this - think of it as a Deployment of pod groups:

- each group gets `LWS_LEADER_ADDRESS`, `LWS_GROUP_SIZE` and `LWS_WORKER_INDEX` env vars, which are passed to the server as the address of the leader, the number of nodes and the node's rank.
- with `restartPolicy: RecreateGroupOnPodRestart`, if any pod of the group fails, the whole group is re-created - the same "all or nothing" logic as in [multi-node training](./fault-tolerance.md#restart-the-whole-job-when-any-pod-fails).
- the Service selects only the leaders.

See the [vLLM LWS example](https://github.com/kubernetes-sigs/lws/blob/main/docs/examples/leaderworkerset/basic/vllm.yaml) in the LWS repo. These pods need the fast inter-node network just like training does - see [Fast Inter-node Networking](./network.md).

## Cold start

How long does it take from "we need another replica" to "it serves requests"? Each stage can take minutes:

| Stage | Typical cost | How to speed it up |
| :---- | :----------- | :----------------- |
| getting a node (cluster autoscaler) | minutes, or never if there is no capacity | keep warm nodes, reservations |
| pulling the image | 10GB+ images take minutes | registry close to the cluster, image caching/streaming, see [Keep the images small](../containers/README.md#keep-the-images-small) |
| loading the weights | depends on the storage: seconds to tens of minutes | fast shared storage, local NVMe copy, `--load-format runai_streamer`, see [Model weights](./storage.md#model-weights) and [Speeding up model loading time](../../inference/README.md#speeding-up-model-loading-time) |
| warm-up: `torch.compile`, CUDA graph capture | can be minutes for big models | persistent compile caches (`VLLM_CACHE_ROOT` on a PVC), `--enforce-eager` for dev work (faster start, slower serving) |

To skip most of these stages altogether, a replica can be restored from a snapshot of an already initialized one, see [Snapshots](./snapshots.md).

Measure each stage for your setup: `kubectl describe pod` shows when the pod got scheduled and how long the image pull took (`Successfully pulled image ... in 2m3s`), and the server's log shows the loading and warm-up times.

## Autoscaling

The standard HPA scales on CPU utilization, which says nothing about an inference server's load - the work is on the GPUs. Scale on what reflects the users' experience, e.g., the number of queued requests or the KV cache usage, which vLLM exports at `/metrics` (e.g., `vllm:num_requests_waiting`, `vllm:kv_cache_usage_perc` - the names change between versions, so check your version's `/metrics`). With Prometheus scraping these, [KEDA](https://keda.sh/) can drive the scaling:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm
spec:
  scaleTargetRef:
    name: llm  # the Deployment
  minReplicaCount: 2
  maxReplicaCount: 8
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: sum(vllm:num_requests_waiting{model_name="llama-70b"})
      threshold: "10"
```

Things to keep in mind:

- **Scaling up takes as long as the [cold start](#cold-start).** Autoscaling handles slow daily patterns, not sudden spikes. Keep `minReplicaCount` high enough for the spikes you need to absorb.
- **Scale down slowly.** Use a long stabilization window for scale-down (the HPA `behavior.scaleDown` settings - with KEDA they go under `advanced.horizontalPodAutoscalerConfig.behavior` of the ScaledObject), otherwise the replicas you paid 15 minutes to start get removed after a short lull.
- **Scale to zero** (`minReplicaCount: 0` - KEDA can do it, the plain HPA can't) saves money for rarely used models, but the first request then waits for the full cold start.

## Routing

A Service spreads requests over the replicas without knowing anything about them. For LLMs that's suboptimal in 2 ways: requests differ hugely in cost (a long prompt vs a short one), and a replica that already has a prompt's prefix in its KV cache could answer much faster. Smarter routing is available through the [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) and projects built on it like [llm-d](https://github.com/llm-d/llm-d), which route based on the replicas' load and their prefix caches.

## Benchmarking the deployment

Before putting a deployment in front of users, measure it under a realistic load from inside the cluster (so that the network between your machine and the cluster isn't part of the measurement), e.g., with vLLM's benchmark from a pod that has vLLM installed (`http://llm` resolves to the Service from within the same namespace, from other namespaces use `http://llm.NAMESPACE.svc`):

```bash
vllm bench serve --base-url http://llm --model llama-70b \
    --tokenizer /shared/models/Llama-3.1-70B-Instruct \
    --dataset-name random --random-input-len 1024 --random-output-len 256 \
    --num-prompts 500 --request-rate 10
```

Without `--request-rate` all the prompts are sent at once, which measures the maximum throughput but not the latency your users would see at a realistic load - run it at several rates.

Compare the TTFT, TPOT and throughput against your requirements, see [Key inference performance metrics](../../inference/README.md#key-inference-performance-metrics) and [Benchmarks](../../inference/README.md#benchmarks).
