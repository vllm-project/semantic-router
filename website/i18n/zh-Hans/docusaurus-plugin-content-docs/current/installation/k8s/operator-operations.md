---
title: 运维 Operator 部署
sidebar_label: Kubernetes 操作符
description: 监控、更新、扩缩容并排查由 Kubernetes Operator 管理的 SemanticRouter。
translation:
  source_commit: "f8c1197a9ed47f7a265cbab83bf2d84eb5fa505e"
  source_file: "docs/installation/k8s/operator-operations.md"
  outdated: false
---

# 运维 Operator 部署

本指南覆盖 `SemanticRouter` 资源的日常运维。安装和首次部署见[使用 Kubernetes Operator 部署](operator)。

## 读取协调状态

```bash
kubectl get semanticrouter <name> -o wide
kubectl describe semanticrouter <name>
kubectl get semanticrouter <name> -o jsonpath='{.status.conditions}'
```

将 `metadata.generation`、`status.observedGeneration`、状态条件和就绪副本一起查看。控制器在运行，并不表示最新的自定义资源 generation 已成功应用。

协调停滞时，检查其拥有的工作负载：

```bash
kubectl get deployment,pod,service,configmap,pvc \
  -l app.kubernetes.io/instance=<name>
kubectl logs -n semantic-router-operator-system \
  deployment/semantic-router-operator-controller-manager
```

## 安全更新

1. 导出当前自定义资源，并记录已部署的镜像引用。
2. 审阅 CRD 和发行说明中的 schema 变更。
3. 在非生产环境应用新的自定义资源或 Operator 版本。
4. 等待 observed generation 和就绪副本收敛。
5. 通过每个重要入口发送真实请求。
6. 若就绪状态或路由回退，回滚自定义资源或镜像引用。

固定镜像标签或 digest。除非这些变更有意耦合并已一起测试，不要在一次上线中同时更改 Operator、Router 镜像、路由策略、模型池和存储后端。

## 扩缩容与可用性

通过 `spec.replicas` 设置固定副本，或启用 HPA adapter：

```yaml
spec:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
```

使用节点亲和性、容忍、拓扑打散或反亲和，以及单独管理的 PodDisruptionBudget，以匹配可用性目标。学习或其他副本本地可变状态可能需要额外设计；不要假定增加副本就能让每个有状态路由功能保持一致。

## 指标与追踪

Router 在已配置的 metrics Service 端口（默认 9190）暴露 Prometheus 指标：

```bash
kubectl port-forward service/<name> 9190:9190
curl -sS http://localhost:9190/metrics | head
```

用部署所用的标签配置 `ServiceMonitor` 或等效采集器。通过 `spec.config.observability` 启用 OpenTelemetry，并将追踪发送到 Router 命名空间可达的 collector。保持追踪采样和采集属性与请求数据的敏感度匹配。

## 常见故障

### 后端发现失败

对于 `service` 后端，验证 Service 名称、命名空间、端口、端点和网络策略。对于 KServe，验证 `InferenceService` 已就绪且其 predictor Service 存在。对于 Llama Stack，检查候选 Service 上的标签。

```bash
kubectl get service,endpoints -n <backend-namespace>
kubectl describe inferenceservice <name> -n <backend-namespace>
```

### Gateway 模式没有路由

Operator 不会创建 `HTTPRoute`。对于普通 Gateway HTTP 转发，省略 `spec.gateway.existingRef`，保留 Envoy sidecar，并将 `/v1` 路由到 Router Service 的 `envoy-http` 端口 **8801**。检查 Gateway 允许来自 Router 命名空间的路由，且路由报告 `Accepted=True` 和 `ResolvedRefs=True`。

若设置了 `spec.gateway.existingRef`，则省略 sidecar。验证网关特定的 ExtProc 策略指向 Router Service 的 gRPC 端口（默认 **50051**），且其模型路由指向真实的后端 Service。Router `api` 端口（默认 **8080**）是管理端点，不能服务推理 completions。见两条[现有 Gateway 部署路径](operator#existing-gateway)。

```bash
kubectl get gateway -A
kubectl get httproute -A
```

### Pod 处于 `ImagePullBackOff`

检查 pod 事件，确认镜像存在，并在仓库需要认证时提供 `imagePullSecret`。

```bash
kubectl describe pod <pod-name>
```

仓库镜像和集群出口指引见[受限网络环境](../../troubleshooting/network-tips)。

### PVC 一直处于 pending

检查集群中是否存在所请求的 StorageClass 和访问模式，以及 provisioner 能否满足所请求的容量：

```bash
kubectl get storageclass
kubectl describe pvc <pvc-name>
```

更改存储设置可能影响现有数据。替换 claim 前，先备份持久状态，并遵循存储提供方的迁移流程。

### 模型产物下载失败

从 Secret 引用下载 token，检查出口和证书配置，并查看 Router 日志。不要把 token 直接放在 `spec.env` 或 ConfigMap 中。

## 删除与数据

删除 `SemanticRouter` 之前，先识别哪些状态是临时的、哪些存在 PVC 中，以及哪些由外部 Redis、Valkey、Milvus、Qdrant 或 Postgres 服务持有。删除自定义资源会按其所有权和保留策略移除受管 Kubernetes 对象；它不一定会移除外部数据存储。

先备份持久数据，并在删除 claim 或命名空间前检查 PVC reclaim 策略。

## 参考

- [使用 Kubernetes Operator 部署](operator)
- [SemanticRouter CRD 参考](../../api/semantic-router-crd)
- [API 与可观测性](../../tutorials/global/api-and-observability)
- [升级与回滚](../upgrade-rollback)
