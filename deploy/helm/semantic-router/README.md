# semantic-router

![Version: 0.2.0](https://img.shields.io/badge/Version-0.2.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: latest](https://img.shields.io/badge/AppVersion-latest-informational?style=flat-square)

A Helm chart for deploying Semantic Router - an intelligent routing system for LLM applications

**Homepage:** <https://github.com/vllm-project/semantic-router>

## Maintainers

| Name | Email | Url |
| ---- | ------ | --- |
| Semantic Router Team |  | <https://github.com/vllm-project/semantic-router> |

## Source Code

* <https://github.com/vllm-project/semantic-router>

## Requirements

| Repository | Name | Version |
|------------|------|---------|
| https://charts.bitnami.com/bitnami | semantic-cache-redis(redis) | >=0.0.0 |
| https://charts.bitnami.com/bitnami | response-api-redis(redis) | >=0.0.0 |
| https://grafana.github.io/helm-charts | grafana | >=0.0.0 |
| https://jaegertracing.github.io/helm-charts | jaeger | >=0.0.0 |
| https://milvus-io.github.io/milvus-helm/ | semantic-cache-milvus(milvus) | >=0.0.0 |
| https://prometheus-community.github.io/helm-charts | prometheus | >=0.0.0 |

## Values Schema

The chart ships a narrow `values.schema.json` for public router and dashboard
deployment controls. Helm validates key replica, autoscaling, persistence, and
safety-guard value types before template rendering. Cross-field production
safety rules remain in templates so the chart can emit targeted errors for
invalid HPA replica bounds and unsupported multi-replica local-state
deployments.

Run `make helm-safety-validate HELM_REPO_UPDATE=false` from the repository root
to validate the schema plus the multi-replica local-state safety guards against
the locked chart dependencies.

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| affinity | object | `{}` |  |
| args[0] | string | `"--secure=false"` |  |
| autoscaling.enabled | bool | `false` | Enable horizontal pod autoscaling |
| autoscaling.maxReplicas | int | `10` | Maximum number of replicas |
| autoscaling.minReplicas | int | `1` | Minimum number of replicas |
| autoscaling.targetCPUUtilizationPercentage | int | `80` | Target CPU utilization percentage |
| config.global.integrations.tools.enabled | bool | `true` |  |
| config.global.integrations.tools.fallback_to_empty | bool | `true` |  |
| config.global.integrations.tools.similarity_threshold | float | `0.2` |  |
| config.global.integrations.tools.tools_db_path | string | `"config/tools_db.json"` |  |
| config.global.integrations.tools.top_k | int | `3` |  |
| config.global.services.api.batch_classification.max_batch_size | int | `100` |  |
| config.global.services.api.batch_classification.max_concurrency | int | `8` |  |
| config.global.services.api.batch_classification.metrics.detailed_goroutine_tracking | bool | `true` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[0] | float | `0.001` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[10] | int | `5` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[11] | int | `10` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[12] | int | `30` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[1] | float | `0.005` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[2] | float | `0.01` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[3] | float | `0.025` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[4] | float | `0.05` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[5] | float | `0.1` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[6] | float | `0.25` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[7] | float | `0.5` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[8] | int | `1` |  |
| config.global.services.api.batch_classification.metrics.duration_buckets[9] | float | `2.5` |  |
| config.global.services.api.batch_classification.metrics.enabled | bool | `true` |  |
| config.global.services.api.batch_classification.metrics.high_resolution_timing | bool | `false` |  |
| config.global.services.api.batch_classification.metrics.sample_rate | float | `1` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[0] | int | `1` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[1] | int | `2` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[2] | int | `5` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[3] | int | `10` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[4] | int | `20` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[5] | int | `50` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[6] | int | `100` |  |
| config.global.services.api.batch_classification.metrics.size_buckets[7] | int | `200` |  |
| config.global.services.observability.tracing.enabled | bool | `false` |  |
| config.global.services.observability.tracing.exporter.endpoint | string | `"jaeger:4317"` |  |
| config.global.services.observability.tracing.exporter.insecure | bool | `true` |  |
| config.global.services.observability.tracing.exporter.type | string | `"otlp"` |  |
| config.global.services.observability.tracing.provider | string | `"opentelemetry"` |  |
| config.global.services.observability.tracing.resource.deployment_environment | string | `"development"` |  |
| config.global.services.observability.tracing.resource.service_name | string | `"vllm-semantic-router"` |  |
| config.global.services.observability.tracing.resource.service_version | string | `""` |  |
| config.global.services.observability.tracing.sampling.rate | float | `0.1` |  |
| config.global.services.observability.tracing.sampling.type | string | `"probabilistic"` |  |
| config.global.services.response_api.enabled | bool | `false` |  |
| config.global.services.response_api.max_responses | int | `1000` |  |
| config.global.services.response_api.store_backend | string | `"memory"` |  |
| config.global.services.response_api.ttl_seconds | int | `86400` |  |
| configOverride | object | `null` | Complete canonical Router config supplied by deployment tooling. Unlike `config`, this map atomically replaces chart defaults before Kubernetes integration rewrites. |
| decisionModel | string | `""` | The Router's decision model, written to `global.model_catalog.system.decision_model`: Vela-2.0-0.3B (the default), Vela-2.0-0.8B, Vela-2.0-4B, Vela-2.0-9B or Vela-1.0, case-insensitive. It answers the built-in signals and every `routing.signals.decision` question that names no deployment. The 4B and 9B need a GPU in the Router pod. A changed value is written into the live config on upgrade too; empty leaves the config's own. |
| configMap.applyValuesRevision | string | `""` | Chart values seed the ConfigMap at install. Change this revision on an upgrade to explicitly replace the live config with `config` or `configOverride`; repeating the same revision preserves later API edits. |
| dashboard.allowOpenBootstrap | bool | `false` | Allow first-admin creation via the public, unauthenticated web-form bootstrap endpoint. Off by default: a fresh, internet-reachable deployment should not be claimable by the first stranger who finds it. Production provisions the admin via the DASHBOARD_ADMIN_* env vars (which create it at startup and close the bootstrap path automatically). Set this to true only for demos where signing up the first admin through the UI is acceptable. |
| dashboard.enabled | bool | `false` | Enable the vLLM-SR dashboard |
| dashboard.envFrom | list | `[]` | Extra envFrom sources for the dashboard container (configMapRef / secretRef). Standard core/v1 EnvFromSource list. |
| dashboard.extraEnv | list | `[]` | Extra environment variables for the dashboard container, appended after the chart-managed TARGET_* vars. Use this to set optional integration env the chart does not expose explicitly (for example PROXY_OVERRIDE_ORIGIN, or in extproc mode TARGET_ENVOY_URL for your gateway) without forking the chart. Standard core/v1 EnvVar list. Avoid redefining a chart-managed var (TARGET_*, DASHBOARD_JWT_SECRET): it produces a duplicate env key and Kubernetes applies last-wins. |
| dashboard.image.pullPolicy | string | `"IfNotPresent"` | Dashboard image pull policy |
| dashboard.image.repository | string | `"ghcr.io/vllm-project/semantic-router/dashboard"` | Dashboard image repository |
| dashboard.image.tag | string | `""` | Dashboard image tag (defaults to the chart appVersion) |
| dashboard.jwtSecret | object | `{"existingSecret":"","existingSecretKey":"jwt-secret"}` | JWT signing secret for dashboard auth sessions. Point this at a Secret you manage (ideally an ExternalSecret) so the signing key is stable. If you leave it unset, the dashboard binary falls back to generating a random secret on every pod start, which invalidates all login sessions on each restart (rolling update, chart bump, or node move forces a re-login). A zero-config install still works (the random fallback is a valid signing key, you can log in and use the dashboard); you just lose existing sessions whenever the pod restarts, so leaving it unset is fine for demos but set this for any deployment where sessions need to survive restarts. |
| dashboard.jwtSecret.existingSecret | string | `""` | Name of an existing Secret holding the JWT signing key. When set, the dashboard reads DASHBOARD_JWT_SECRET from it via secretKeyRef. When empty, no env is injected and the binary uses its per-start random fallback. |
| dashboard.jwtSecret.existingSecretKey | string | `"jwt-secret"` | Key within existingSecret holding the JWT signing secret. |
| dashboard.persistence.accessMode | string | `"ReadWriteOnce"` | Access mode for the dashboard-local state PVC |
| dashboard.persistence.annotations | object | `{}` | Annotations for the dashboard-local state PVC |
| dashboard.persistence.enabled | bool | `true` | Persist dashboard-local auth/session/workflow state and config backups. ConfigMap edits require a rollout, so backups must survive pod replacement for the rollback API to remain usable. Set false only for disposable demos. |
| dashboard.persistence.existingClaim | string | `""` | Existing PVC to mount for dashboard-local state |
| dashboard.persistence.mountPath | string | `"/app/data"` | Container mount path for dashboard-local state |
| dashboard.persistence.size | string | `"1Gi"` | Requested dashboard-local state size |
| dashboard.persistence.storageClassName | string | `""` | Storage class name. Leave empty for the cluster default; use "-" to render storageClassName: "". |
| dashboard.podSecurityContext | object | `{"fsGroup":65532}` | Pod-level security context. The default fsGroup matches the non-root user (UID/GID 65532) baked into the upstream dashboard image, ensuring the persistence PVC mount at /app/data is writable by the binary. Without this, the dashboard crashloops with "unable to open database file" when persistence is enabled on storage classes that mount as root:root 0755 (which is the default behavior for most cloud-provider CSI drivers). Override if you build a custom dashboard image with a different non-root UID. |
| dashboard.readonly | bool | `false` | Run dashboard in read-only mode |
| dashboard.replicaCount | int | `1` | Dashboard replica count. Must stay 1 until the dashboard auth/session store supports a shared multi-replica backend. |
| dashboard.resources.limits | object | `{"cpu":"500m","memory":"512Mi"}` |  |
| dashboard.resources.requests | object | `{"cpu":"100m","memory":"128Mi"}` |  |
| dashboard.service.port | int | `8700` | Dashboard service port |
| dashboard.service.targetPort | int | `8700` | Dashboard target port |
| dashboard.service.type | string | `"ClusterIP"` | Dashboard service type |
| config.listeners[0].address | string | `"0.0.0.0"` |  |
| config.listeners[0].name | string | `"http-8899"` |  |
| config.listeners[0].port | int | `8899` |  |
| config.listeners[0].timeout | string | `"300s"` |  |
| config.providers.defaults.model | string | `"replace-with-your-model"` |  |
| config.providers.defaults.reasoning_effort | string | `"high"` |  |
| config.providers.models[0].backend_refs[0].endpoint | string | `"replace-with-your-vllm-service:8000"` |  |
| config.providers.models[0].backend_refs[0].name | string | `"primary"` |  |
| config.providers.models[0].backend_refs[0].protocol | string | `"http"` |  |
| config.providers.models[0].backend_refs[0].provider | string | `"vllm"` |  |
| config.providers.models[0].backend_refs[0].weight | int | `100` |  |
| config.providers.models[0].name | string | `"replace-with-your-model"` |  |
| config.providers.models[0].provider_model_id | string | `"replace-with-your-model"` |  |
| config.routing.decisions[0].description | string | `"Default route for every request while you wire real backends."` |  |
| config.routing.decisions[0].modelRefs[0].model | string | `"replace-with-your-model"` |  |
| config.routing.decisions[0].modelRefs[0].use_reasoning | bool | `false` |  |
| config.routing.decisions[0].name | string | `"default-route"` |  |
| config.routing.decisions[0].priority | int | `100` |  |
| config.routing.decisions[0].rules.conditions | list | `[]` |  |
| config.routing.decisions[0].rules.operator | string | `"AND"` |  |
| config.routing.modelCards[0].name | string | `"replace-with-your-model"` |  |
| config.routing.signals.domains[0].description | string | `"Catch-all domain"` |  |
| config.routing.signals.domains[0].mmlu_categories[0] | string | `"other"` |  |
| config.routing.signals.domains[0].name | string | `"general"` |  |
| config.version | string | `"v0.3"` |  |
| dependencies.observability.grafana.adminPassword | string | `"admin"` |  |
| dependencies.observability.grafana.adminUser | string | `"admin"` |  |
| dependencies.observability.grafana.enabled | bool | `false` |  |
| dependencies.observability.jaeger.enabled | bool | `false` |  |
| dependencies.observability.jaeger.otlpGrpcPort | int | `4317` |  |
| dependencies.observability.jaeger.serviceName | string | `""` |  |
| dependencies.observability.prometheus.enabled | bool | `false` |  |
| dependencies.responseApi.milvus.conversationCollection | string | `"semantic_router_conversations"` |  |
| dependencies.responseApi.milvus.database | string | `"semantic_router_cache"` |  |
| dependencies.responseApi.milvus.enabled | bool | `false` |  |
| dependencies.responseApi.milvus.host | string | `""` |  |
| dependencies.responseApi.milvus.port | int | `19530` |  |
| dependencies.responseApi.milvus.responseCollection | string | `"semantic_router_responses"` |  |
| dependencies.responseApi.redis.database | int | `0` |  |
| dependencies.responseApi.redis.enabled | bool | `false` |  |
| dependencies.responseApi.redis.host | string | `""` |  |
| dependencies.responseApi.redis.password | string | `""` |  |
| dependencies.responseApi.redis.port | int | `6379` |  |
| dependencies.responseApi.redis.timeout | int | `30` |  |
| dependencies.responseApi.redis.tls.enabled | bool | `false` |  |
| dependencies.semanticCache.milvus.auth.enabled | bool | `false` |  |
| dependencies.semanticCache.milvus.auth.password | string | `""` |  |
| dependencies.semanticCache.milvus.auth.username | string | `""` |  |
| dependencies.semanticCache.milvus.collection.description | string | `"Semantic cache for LLM request-response pairs"` |  |
| dependencies.semanticCache.milvus.collection.index.params.efConstruction | int | `64` |  |
| dependencies.semanticCache.milvus.collection.index.params.m | int | `16` |  |
| dependencies.semanticCache.milvus.collection.index.type | string | `"HNSW"` |  |
| dependencies.semanticCache.milvus.collection.metricType | string | `"IP"` |  |
| dependencies.semanticCache.milvus.collection.name | string | `"semantic_cache"` |  |
| dependencies.semanticCache.milvus.collection.vectorFieldName | string | `"embedding"` |  |
| dependencies.semanticCache.milvus.database | string | `"semantic_router_cache"` |  |
| dependencies.semanticCache.milvus.development.autoCreateCollection | bool | `true` |  |
| dependencies.semanticCache.milvus.development.dropCollectionOnStartup | bool | `false` |  |
| dependencies.semanticCache.milvus.enabled | bool | `false` |  |
| dependencies.semanticCache.milvus.host | string | `""` |  |
| dependencies.semanticCache.milvus.port | int | `19530` |  |
| dependencies.semanticCache.milvus.search.params.ef | int | `64` |  |
| dependencies.semanticCache.milvus.search.topk | int | `10` |  |
| dependencies.semanticCache.milvus.timeout | int | `30` |  |
| dependencies.semanticCache.milvus.tls.enabled | bool | `false` |  |
| dependencies.semanticCache.redis.database | int | `0` |  |
| dependencies.semanticCache.redis.development.autoCreateIndex | bool | `true` |  |
| dependencies.semanticCache.redis.development.dropIndexOnStartup | bool | `false` |  |
| dependencies.semanticCache.redis.enabled | bool | `false` |  |
| dependencies.semanticCache.redis.host | string | `""` |  |
| dependencies.semanticCache.redis.index.indexType | string | `"HNSW"` |  |
| dependencies.semanticCache.redis.index.metricType | string | `"COSINE"` |  |
| dependencies.semanticCache.redis.index.name | string | `"semantic_cache_idx"` |  |
| dependencies.semanticCache.redis.index.params.efConstruction | int | `64` |  |
| dependencies.semanticCache.redis.index.params.m | int | `16` |  |
| dependencies.semanticCache.redis.index.prefix | string | `"doc:"` |  |
| dependencies.semanticCache.redis.index.vectorFieldName | string | `"embedding"` |  |
| dependencies.semanticCache.redis.password | string | `""` |  |
| dependencies.semanticCache.redis.port | int | `6379` |  |
| dependencies.semanticCache.redis.search.topk | int | `1` |  |
| dependencies.semanticCache.redis.timeout | int | `30` |  |
| dependencies.semanticCache.redis.tls.enabled | bool | `false` |  |
| env[0].name | string | `"HOME"` |  |
| env[0].value | string | `"/tmp"` |  |
| env[1].name | string | `"TMPDIR"` |  |
| env[1].value | string | `"/tmp"` |  |
| env[2].name | string | `"HF_HOME"` |  |
| env[2].value | string | `"/app/models/.cache/huggingface"` |  |
| env[3].name | string | `"HF_TOKEN"` |  |
| env[3].valueFrom.secretKeyRef.key | string | `"token"` |  |
| env[3].valueFrom.secretKeyRef.name | string | `"hf-token-secret"` |  |
| env[3].valueFrom.secretKeyRef.optional | bool | `true` |  |
| env[4].name | string | `"HUGGINGFACE_HUB_TOKEN"` |  |
| env[4].valueFrom.secretKeyRef.key | string | `"token"` |  |
| env[4].valueFrom.secretKeyRef.name | string | `"hf-token-secret"` |  |
| env[4].valueFrom.secretKeyRef.optional | bool | `true` |  |
| extraVolumeMounts | list | `[]` | Extra Router mounts. A mount at `/app/models` replaces the default model volume mount. |
| extraVolumes | list | `[]` | Volumes for custom mounts; provide a matching volume when replacing `/app/models`. |
| fullnameOverride | string | `""` | Override the full name of the chart |
| gateway.mode | string | `"standalone"` | `standalone`: the Router serves the OpenAI-compatible API on `config.listeners` itself, answers `/health` and `/ready` there, and the Service exposes each listener's port; no Envoy runs. `extproc`: the Router serves Envoy's ext_proc gRPC on `service.grpc.port` for a gateway you run (Envoy Gateway, Envoy AI Gateway, Istio, KServe, llm-d). Releases before standalone mode deployed `extproc`; set it to keep such an integration. |
| gateway.tls.secretName | string | `""` | Standalone only. A `kubernetes.io/tls` Secret mounted at `/app/config/certs`, for listeners whose `tls` names `cert_file: certs/tls.crt` and `key_file: certs/tls.key`. The Secret is mounted as a volume, so a rotated certificate reaches new connections without a restart. |
| global.imageRegistry | string | `""` | Optional registry prefix applied to all images (e.g., mirror in China such as registry.cn-hangzhou.aliyuncs.com) |
| global.namespace | string | `""` | Namespace for all resources (if not specified, uses Release.Namespace) |
| grafana.image.tag | string | `"11.5.1"` |  |
| image.pullPolicy | string | `"IfNotPresent"` | Image pull policy |
| image.repository | string | `"ghcr.io/vllm-project/semantic-router/vllm-sr"` | Image repository: `vllm-sr` (CPU), `vllm-sr-rocm` (AMD GPUs) or `vllm-sr-cuda` (NVIDIA GPUs). |
| image.tag | string | `""` | Image tag (overrides the image tag whose default is the chart appVersion) |
| imagePullSecrets | list | `[]` | Image pull secrets for private registries |
| ingress.annotations | object | `{}` | Ingress annotations |
| ingress.className | string | `""` | Ingress class name |
| ingress.enabled | bool | `false` | Enable ingress |
| ingress.hosts | list | `[{"host":"semantic-router.local","paths":[{"path":"/","pathType":"Prefix"}]}]` | Ingress hosts configuration. A path without `servicePort` goes to the first listener without TLS in standalone mode, and to `service.api.port` in extproc mode. |
| ingress.tls | list | `[]` | Ingress TLS configuration |
| jaeger.allInOne.image.tag | string | `"latest"` |  |
| livenessProbe.enabled | bool | `true` | Enable liveness probe |
| livenessProbe.failureThreshold | int | `5` | Failure threshold |
| livenessProbe.initialDelaySeconds | int | `30` | Initial delay seconds |
| livenessProbe.periodSeconds | int | `30` | Period seconds |
| livenessProbe.timeoutSeconds | int | `10` | Timeout seconds |
| nameOverride | string | `""` | Override the name of the chart |
| nodeSelector | object | `{}` |  |
| observability.alerts.enabled | bool | `false` | Render a PrometheusRule for Semantic Router alerts. Requires Prometheus Operator or another controller that watches PrometheusRule. |
| observability.alerts.labels | object | `{}` | Additional labels added to the PrometheusRule. |
| observability.alerts.thresholds.cacheHitRate | float | `0.2` |  |
| observability.alerts.thresholds.completionLatencyP95Seconds | int | `30` |  |
| observability.alerts.thresholds.firstResponseObservationP95Seconds | int | `5` |  |
| observability.alerts.thresholds.inflightRequests | int | `50` |  |
| observability.alerts.thresholds.requestErrorRate | float | `0.05` |  |
| observability.alerts.thresholds.responseDurationPerOutputTokenP95Seconds | float | `0.25` |  |
| observability.alerts.thresholds.routingLatencyP95Seconds | float | `0.1` |  |
| persistence.accessMode | string | `"ReadWriteOnce"` | Access mode |
| persistence.annotations | object | `{}` | Annotations for PVC |
| persistence.enabled | bool | `true` | Enable persistent volume |
| persistence.existingClaim | string | `""` | Existing claim name (if provided, will use existing PVC instead of creating new one) |
| persistence.size | string | `"10Gi"` | Storage size of the model volume. The runtime downloads the router's models here on first start; Vela Omni Mini alone takes 4.3 GB. |
| persistence.storageClassName | string | `"standard"` | Storage class name. Leave empty for the cluster default; use "-" to render storageClassName: "". |
| podAnnotations | object | `{}` |  |
| podSecurityContext | object | `{}` |  |
| prometheus.server.image.tag | string | `"v2.53.0"` |  |
| rbac.create | bool | `true` | Create RBAC resources (ClusterRole and ClusterRoleBinding) |
| readinessProbe.enabled | bool | `true` | Enable readiness probe |
| readinessProbe.failureThreshold | int | `5` | Failure threshold |
| readinessProbe.initialDelaySeconds | int | `30` | Initial delay seconds |
| readinessProbe.periodSeconds | int | `30` | Period seconds |
| readinessProbe.timeoutSeconds | int | `10` | Timeout seconds |
| replicaCount | int | `1` | Number of replicas for the deployment |
| resources.limits | object | `{"cpu":"2","memory":"7Gi"}` | Resource limits |
| resources.requests | object | `{"cpu":"1","memory":"3Gi"}` | Resource requests |
| response-api-redis.architecture | string | `"standalone"` |  |
| response-api-redis.auth.enabled | bool | `false` |  |
| safetyGuards.rejectMultiReplicaLocalLearningState | bool | `true` | Reject multi-replica router deployments when config enables Router Learning request-time local state. Disable only when accepting replica-local learning divergence or using sticky routing. |
| securityContext.allowPrivilegeEscalation | bool | `false` | Allow privilege escalation |
| securityContext.runAsNonRoot | bool | `false` | Run as non-root user |
| semantic-cache-milvus.cluster.enabled | bool | `false` |  |
| semantic-cache-redis.architecture | string | `"standalone"` |  |
| semantic-cache-redis.auth.enabled | bool | `false` |  |
| service.api.port | int | `8080` | HTTP API port number |
| service.api.protocol | string | `"TCP"` | HTTP API protocol |
| service.api.targetPort | int | `8080` | HTTP API target port |
| service.grpc.port | int | `50051` | gRPC port number (extproc mode only) |
| service.grpc.protocol | string | `"TCP"` | gRPC protocol |
| service.grpc.targetPort | int | `50051` | gRPC target port |
| service.metrics.enabled | bool | `true` | Enable metrics service |
| service.metrics.port | int | `9190` | Metrics port number |
| service.metrics.protocol | string | `"TCP"` | Metrics protocol |
| service.metrics.targetPort | int | `9190` | Metrics target port |
| service.type | string | `"ClusterIP"` | Service type. In standalone mode the Service also exposes every port in `config.listeners`. |
| serviceAccount.annotations | object | `{}` | Annotations to add to the service account |
| serviceAccount.create | bool | `true` | Specifies whether a service account should be created |
| serviceAccount.name | string | `""` | The name of the service account to use |
| startupProbe.enabled | bool | `true` | Enable startup probe |
| startupProbe.failureThreshold | int | `360` | Failure threshold (360 * 10s = 60 minutes total timeout for model downloads) |
| startupProbe.periodSeconds | int | `10` | Period seconds |
| startupProbe.timeoutSeconds | int | `5` | Timeout seconds |
| tolerations | list | `[]` |  |
| toolsDb[0].category | string | `"weather"` |  |
| toolsDb[0].description | string | `"Get current weather information, temperature, conditions, forecast for any location, city, or place. Check weather today, now, current conditions, temperature, rain, sun, cloudy, hot, cold, storm, snow"` |  |
| toolsDb[0].tags[0] | string | `"weather"` |  |
| toolsDb[0].tags[1] | string | `"temperature"` |  |
| toolsDb[0].tags[2] | string | `"forecast"` |  |
| toolsDb[0].tags[3] | string | `"climate"` |  |
| toolsDb[0].tool.function.description | string | `"Get current weather information for a location"` |  |
| toolsDb[0].tool.function.name | string | `"get_weather"` |  |
| toolsDb[0].tool.function.parameters.properties.location.description | string | `"The city and state, e.g. San Francisco, CA"` |  |
| toolsDb[0].tool.function.parameters.properties.location.type | string | `"string"` |  |
| toolsDb[0].tool.function.parameters.properties.unit.description | string | `"Temperature unit"` |  |
| toolsDb[0].tool.function.parameters.properties.unit.enum[0] | string | `"celsius"` |  |
| toolsDb[0].tool.function.parameters.properties.unit.enum[1] | string | `"fahrenheit"` |  |
| toolsDb[0].tool.function.parameters.properties.unit.type | string | `"string"` |  |
| toolsDb[0].tool.function.parameters.required[0] | string | `"location"` |  |
| toolsDb[0].tool.function.parameters.type | string | `"object"` |  |
| toolsDb[0].tool.type | string | `"function"` |  |
| toolsDb[1].category | string | `"search"` |  |
| toolsDb[1].description | string | `"Search the internet, web search, find information online, browse web content, lookup, research, google, find answers, discover, investigate"` |  |
| toolsDb[1].tags[0] | string | `"search"` |  |
| toolsDb[1].tags[1] | string | `"web"` |  |
| toolsDb[1].tags[2] | string | `"internet"` |  |
| toolsDb[1].tags[3] | string | `"information"` |  |
| toolsDb[1].tags[4] | string | `"browse"` |  |
| toolsDb[1].tool.function.description | string | `"Search the web for information"` |  |
| toolsDb[1].tool.function.name | string | `"search_web"` |  |
| toolsDb[1].tool.function.parameters.properties.num_results.default | int | `5` |  |
| toolsDb[1].tool.function.parameters.properties.num_results.description | string | `"Number of results to return"` |  |
| toolsDb[1].tool.function.parameters.properties.num_results.type | string | `"integer"` |  |
| toolsDb[1].tool.function.parameters.properties.query.description | string | `"The search query"` |  |
| toolsDb[1].tool.function.parameters.properties.query.type | string | `"string"` |  |
| toolsDb[1].tool.function.parameters.required[0] | string | `"query"` |  |
| toolsDb[1].tool.function.parameters.type | string | `"object"` |  |
| toolsDb[1].tool.type | string | `"function"` |  |
| toolsDb[2].category | string | `"math"` |  |
| toolsDb[2].description | string | `"Calculate mathematical expressions, solve math problems, arithmetic operations, compute numbers, addition, subtraction, multiplication, division, equations, formula"` |  |
| toolsDb[2].tags[0] | string | `"math"` |  |
| toolsDb[2].tags[1] | string | `"calculation"` |  |
| toolsDb[2].tags[2] | string | `"arithmetic"` |  |
| toolsDb[2].tags[3] | string | `"compute"` |  |
| toolsDb[2].tags[4] | string | `"numbers"` |  |
| toolsDb[2].tool.function.description | string | `"Perform mathematical calculations"` |  |
| toolsDb[2].tool.function.name | string | `"calculate"` |  |
| toolsDb[2].tool.function.parameters.properties.expression.description | string | `"Mathematical expression to evaluate"` |  |
| toolsDb[2].tool.function.parameters.properties.expression.type | string | `"string"` |  |
| toolsDb[2].tool.function.parameters.required[0] | string | `"expression"` |  |
| toolsDb[2].tool.function.parameters.type | string | `"object"` |  |
| toolsDb[2].tool.type | string | `"function"` |  |
| toolsDb[3].category | string | `"communication"` |  |
| toolsDb[3].description | string | `"Send email messages, email communication, contact people via email, mail, message, correspondence, notify, inform"` |  |
| toolsDb[3].tags[0] | string | `"email"` |  |
| toolsDb[3].tags[1] | string | `"send"` |  |
| toolsDb[3].tags[2] | string | `"communication"` |  |
| toolsDb[3].tags[3] | string | `"message"` |  |
| toolsDb[3].tags[4] | string | `"contact"` |  |
| toolsDb[3].tool.function.description | string | `"Send an email message"` |  |
| toolsDb[3].tool.function.name | string | `"send_email"` |  |
| toolsDb[3].tool.function.parameters.properties.body.description | string | `"Email body content"` |  |
| toolsDb[3].tool.function.parameters.properties.body.type | string | `"string"` |  |
| toolsDb[3].tool.function.parameters.properties.subject.description | string | `"Email subject"` |  |
| toolsDb[3].tool.function.parameters.properties.subject.type | string | `"string"` |  |
| toolsDb[3].tool.function.parameters.properties.to.description | string | `"Recipient email address"` |  |
| toolsDb[3].tool.function.parameters.properties.to.type | string | `"string"` |  |
| toolsDb[3].tool.function.parameters.required[0] | string | `"to"` |  |
| toolsDb[3].tool.function.parameters.required[1] | string | `"subject"` |  |
| toolsDb[3].tool.function.parameters.required[2] | string | `"body"` |  |
| toolsDb[3].tool.function.parameters.type | string | `"object"` |  |
| toolsDb[3].tool.type | string | `"function"` |  |
| toolsDb[4].category | string | `"productivity"` |  |
| toolsDb[4].description | string | `"Schedule meetings, create calendar events, set appointments, manage calendar, book time, plan meeting, organize schedule, reminder, agenda"` |  |
| toolsDb[4].tags[0] | string | `"calendar"` |  |
| toolsDb[4].tags[1] | string | `"event"` |  |
| toolsDb[4].tags[2] | string | `"meeting"` |  |
| toolsDb[4].tags[3] | string | `"appointment"` |  |
| toolsDb[4].tags[4] | string | `"schedule"` |  |
| toolsDb[4].tool.function.description | string | `"Create a new calendar event or appointment"` |  |
| toolsDb[4].tool.function.name | string | `"create_calendar_event"` |  |
| toolsDb[4].tool.function.parameters.properties.date.description | string | `"Event date in YYYY-MM-DD format"` |  |
| toolsDb[4].tool.function.parameters.properties.date.type | string | `"string"` |  |
| toolsDb[4].tool.function.parameters.properties.duration.description | string | `"Duration in minutes"` |  |
| toolsDb[4].tool.function.parameters.properties.duration.type | string | `"integer"` |  |
| toolsDb[4].tool.function.parameters.properties.time.description | string | `"Event time in HH:MM format"` |  |
| toolsDb[4].tool.function.parameters.properties.time.type | string | `"string"` |  |
| toolsDb[4].tool.function.parameters.properties.title.description | string | `"Event title"` |  |
| toolsDb[4].tool.function.parameters.properties.title.type | string | `"string"` |  |
| toolsDb[4].tool.function.parameters.required[0] | string | `"title"` |  |
| toolsDb[4].tool.function.parameters.required[1] | string | `"date"` |  |
| toolsDb[4].tool.function.parameters.required[2] | string | `"time"` |  |
| toolsDb[4].tool.function.parameters.type | string | `"object"` |  |
| toolsDb[4].tool.type | string | `"function"` |  |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.14.2](https://github.com/norwoodj/helm-docs/releases/v1.14.2)

### Gateway modes

`gateway.mode` selects what serves client traffic.

* **`standalone` (the default).** The Router serves the OpenAI-compatible API
  on every listener in `config.listeners`, with no Envoy. The container, the
  Service and the probes take their ports from that one list, and each listener
  binds all of the Pod's addresses. Startup and readiness probe `/ready`, and
  liveness probes `/health`, on the first listener without TLS. The Dashboard
  sends Playground requests to that listener. A listener port that the Router
  API, metrics or ext_proc port (50051) already use fails the render.
* **`extproc`.** The Router serves Envoy's ext_proc gRPC on
  `service.grpc.port` (50051) for a gateway you run: Envoy Gateway, Envoy AI
  Gateway, Istio, KServe or llm-d. The Service and the headless Service expose
  that port, and `config.listeners` is unused. This is what every release
  before standalone mode deployed.

The chart passes the mode to the Router (`-gateway=standalone` or
`-gateway=extproc`); `vllm-sr serve --target kubernetes --gateway ...` writes
the same value.

**Listener TLS.** A standalone listener serves TLS when its `tls` names a
certificate and key. Put them in a `kubernetes.io/tls` Secret and set
`gateway.tls.secretName`; the chart mounts it at `/app/config/certs`, so the
listener names `cert_file: certs/tls.crt` and `key_file: certs/tls.key`. The
Secret is mounted as a volume rather than with `subPath`, so when cert-manager
or `kubectl` updates it, Kubernetes replaces the files in place and the Router
serves the new pair on new connections without a restart. Changing a
listener's paths still needs a restart.

**Identity behind an authenticating ingress.** A standalone listener drops the
client identity headers (`x-authz-*`) by default. When an authenticating proxy
in front of the Service sets them, add `identity: {trust_headers: true}` to that
listener in `config.listeners`, with `trusted_peers` set to the proxy's Pod or
node CIDRs; memory, authz signals and per-user rate limits then see the user.

**GPU platforms.** `vllm-sr serve --target kubernetes --platform amd|nvidia`
sets `image.repository` to `vllm-sr-rocm` or `vllm-sr-cuda` and adds
`amd.com/gpu: 1` or `nvidia.com/gpu: 1` to `resources.limits`. Set the same
values yourself to run the built-in models on a GPU node. The Router runs as
uid 65532; on AMD nodes the device plugin's `/dev/kfd` and `/dev/dri` usually
belong to the host's `render` and `video` groups, so add those group IDs to
`podSecurityContext.supplementalGroups` unless the devices are world-accessible.

**Upgrading from a release before standalone mode.** Those releases served
ext_proc only, so a gateway in front of the Router (Envoy Gateway, Envoy AI
Gateway, Istio, KServe, llm-d) keeps working only with
`--set gateway.mode=extproc`; add it to your values before you upgrade. Without
it the upgraded Router serves its listeners instead, and a live config that
still carries the old default listeners (`grpc-50051` or `http-8080`) fails
the render, because those ports belong to ext_proc and the Router API. `helm rollback` restores the
previous release, mode included.

### Runtime readiness and configuration updates

In extproc mode, startup and readiness probes use the standard gRPC health
service on the Router port. The chart defaults to plaintext gRPC
(`--secure=false`), as required by Kubernetes gRPC probes. Deployments that
enable TLS must disable these built-in probes and supply TLS-aware probes.
Liveness continues to check the TCP listener. In standalone mode the probes
use the listener's `/ready` and `/health` instead, over HTTPS when every
listener serves TLS.

In Kubernetes config mode, the first CR configuration must finish model
preparation, warmup and runtime activation before health becomes serving. The
controller watches the Pod namespace by default; `--namespace` explicitly
overrides it. For the first Kubernetes-config install, install without Helm
`--wait`, apply one pool and route in that namespace, then wait for the Router
deployment. On upgrades with existing CRs outside the Pod namespace, retain
their namespace with an explicit `--namespace` argument or move the CRs first.
`Ready=True` on an IntelligentPool/IntelligentRoute acknowledges
activation on the reconciling replica. A failed subsequent candidate reports
`ActivationFailed` without taking the previous serving generation out of service.
Status persistence failures are retried without rebuilding a successful generation.
The startup probe retains its configurable 60-minute default model-download budget.

For chart-managed file configuration, Router management writes update the named
ConfigMap through the Kubernetes API. A successful write returns HTTP 202 with
`activation_status: persisted`; the existing `subPath` mount and active Router
generation stay on the prior document until the Router deployment is rolled
out. `/api/v1/config/hash` reads the saved ConfigMap and reports when activation
becomes active. A second mutation on a stale Pod returns HTTP 409
`CONFIG_ROLLOUT_REQUIRED`, preventing it from overwriting the saved change.
Kubernetes CR-managed configuration remains read-only through this API. Helm
upgrades preserve the live `config.yaml` by default, including Dashboard and
Router API edits. Chart values seed the first install. To intentionally replace
the live document with reviewed chart values, set a new
`configMap.applyValuesRevision` on that upgrade. Reusing that revision on later
upgrades preserves subsequent API edits. Use a server-side Helm dry run when
previewing an upgrade: a client-side render cannot look up the live ConfigMap.
With one Router replica, the Router stores its config versions under
`/app/models/.vllm-sr/config-backups`, so the default models PVC keeps rollback
versions available after the required rollout. A custom `/app/models` mount
must be writable and persistent for the same. With several replicas or
autoscaling, or with `persistence.enabled=false`, each Pod keeps its own
versions in `/tmp/vllm-sr/config-history` for as long as it runs: the ConfigMap
is the document replicas share, so restore an earlier document by saving it
there again. When the Dashboard is enabled, its separate PVC retains Dashboard
config backups.
Managed knowledge base assets need a writable, persistent directory in
addition to the YAML document; their mutation API returns
`KB_ASSET_STORAGE_READ_ONLY` on ConfigMap-backed deployments.
