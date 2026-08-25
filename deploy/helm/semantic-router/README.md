# semantic-router

![Version: 0.2.0](https://img.shields.io/badge/Version-0.2.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: latest](https://img.shields.io/badge/AppVersion-latest-informational?style=flat-square)

A Helm chart for deploying Semantic Router - an intelligent routing system for LLM applications

`config` is the single additive v0.3 Router manifest. The chart stores it in a
content-addressed immutable ConfigMap and mounts `config.yaml` read-only; a
manifest change creates a new ConfigMap name and rolls Router Pods. Models,
Recipes, and Entrypoints remain readable in the same schema. Adding
`global.stores.management.postgres` makes routing state durable; enabling
Router-native access also requires `global.stores.runtime.redis`.

Dashboard and observability dependencies are optional and disabled in the base
values. The chart does not install PostgreSQL or the access Runtime store.

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

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| affinity | object | `{}` |  |
| args[0] | string | `"--secure=false"` |  |
| autoscaling.enabled | bool | `false` |  |
| autoscaling.maxReplicas | int | `10` |  |
| autoscaling.minReplicas | int | `1` |  |
| autoscaling.targetCPUUtilizationPercentage | int | `80` |  |
| config.entrypoints[0].assignments.default-route.models[0].model | string | `"replace-with-your-model"` |  |
| config.entrypoints[0].model_names[0] | string | `"vllm-sr/auto"` |  |
| config.entrypoints[0].model_names[1] | string | `"auto"` |  |
| config.entrypoints[0].recipe | string | `"default"` |  |
| config.global.integrations.tools.enabled | bool | `true` |  |
| config.global.integrations.tools.fallback_to_empty | bool | `true` |  |
| config.global.integrations.tools.similarity_threshold | float | `0.2` |  |
| config.global.integrations.tools.tools_db_path | string | `"config/tools_db.json"` |  |
| config.global.integrations.tools.top_k | int | `3` |  |
| config.global.services.api.batch_classification.concurrency_threshold | int | `5` |  |
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
| config.global.services.backend_dispatch.audience | string | `"vllm-sr.backend-dispatch"` |  |
| config.global.services.backend_dispatch.bind_address | string | `"0.0.0.0"` |  |
| config.global.services.backend_dispatch.capability_ttl | string | `"30s"` |  |
| config.global.services.backend_dispatch.max_request_body_bytes | int | `67108864` |  |
| config.global.services.backend_dispatch.port | int | `8180` |  |
| config.global.services.backend_egress.policy_file | string | `"/app/config/backend-egress-policy.yaml"` |  |
| config.global.services.management_api.bind_address | string | `"0.0.0.0"` |  |
| config.global.services.management_api.enabled | bool | `false` |  |
| config.global.services.management_api.port | int | `8080` |  |
| config.global.services.management_api.remote_exposure | bool | `false` |  |
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
| config.global.stores.memory.embedding_model | string | `"mmbert"` |  |
| config.global.stores.response_cache.backend_type | string | `"memory"` |  |
| config.global.stores.response_cache.embedding_model | string | `"mmbert"` |  |
| config.global.stores.response_cache.enabled | bool | `true` |  |
| config.global.stores.response_cache.eviction_policy | string | `"fifo"` |  |
| config.global.stores.response_cache.max_entries | int | `1000` |  |
| config.global.stores.response_cache.similarity_threshold | float | `0.8` |  |
| config.global.stores.response_cache.ttl_seconds | int | `3600` |  |
| config.global.stores.vector_store.embedding_model | string | `"mmbert"` |  |
| config.listeners[0].address | string | `"0.0.0.0"` |  |
| config.listeners[0].name | string | `"grpc-50051"` |  |
| config.listeners[0].port | int | `50051` |  |
| config.listeners[0].timeout | string | `"300s"` |  |
| config.listeners[1].address | string | `"0.0.0.0"` |  |
| config.listeners[1].name | string | `"http-8080"` |  |
| config.listeners[1].port | int | `8080` |  |
| config.listeners[1].timeout | string | `"300s"` |  |
| config.providers.models[0].api_format | string | `"openai"` |  |
| config.providers.models[0].backend_refs[0].endpoint | string | `"http://replace-with-your-vllm-service:8000/v1"` |  |
| config.providers.models[0].backend_refs[0].provider | string | `"vllm"` |  |
| config.providers.models[0].control.retry.count | int | `2` |  |
| config.providers.models[0].control.retry.on[0] | string | `"unavailable"` |  |
| config.providers.models[0].control.retry.on[1] | string | `"timeout"` |  |
| config.providers.models[0].control.timeout.request | string | `"60s"` |  |
| config.providers.models[0].control.timeout.stream | string | `"10m"` |  |
| config.providers.models[0].name | string | `"replace-with-your-model"` |  |
| config.providers.models[0].provider_model_id | string | `"replace-with-your-model"` |  |
| config.recipes[0].description | string | `"Default routing recipe."` |  |
| config.recipes[0].name | string | `"default"` |  |
| config.recipes[0].routing.decisions[0].description | string | `"Route every request to the connected model."` |  |
| config.recipes[0].routing.decisions[0].name | string | `"default-route"` |  |
| config.recipes[0].routing.decisions[0].rules.conditions | list | `[]` |  |
| config.recipes[0].routing.decisions[0].rules.operator | string | `"AND"` |  |
| config.routing.modelCards[0].capabilities[0] | string | `"chat"` |  |
| config.routing.modelCards[0].description | string | `"Replace this example with a connected model."` |  |
| config.routing.modelCards[0].modality | string | `"text"` |  |
| config.routing.modelCards[0].name | string | `"replace-with-your-model"` |  |
| config.version | string | `"v0.3"` |  |
| dashboard.allowOpenBootstrap | bool | `false` |  |
| dashboard.enabled | bool | `false` |  |
| dashboard.envFrom | list | `[]` |  |
| dashboard.extraEnv | list | `[]` |  |
| dashboard.httpRoute.annotations | object | `{}` |  |
| dashboard.httpRoute.enabled | bool | `false` |  |
| dashboard.httpRoute.hostname | string | `"semantic-router-dashboard.local"` |  |
| dashboard.httpRoute.parentRef.name | string | `""` |  |
| dashboard.httpRoute.parentRef.namespace | string | `""` |  |
| dashboard.httpRoute.parentRef.sectionName | string | `""` |  |
| dashboard.image.pullPolicy | string | `"IfNotPresent"` |  |
| dashboard.image.repository | string | `"ghcr.io/vllm-project/semantic-router/dashboard"` |  |
| dashboard.image.tag | string | `"latest"` |  |
| dashboard.jwtSecret.existingSecret | string | `""` |  |
| dashboard.jwtSecret.existingSecretKey | string | `"jwt-secret"` |  |
| dashboard.persistence.accessMode | string | `"ReadWriteOnce"` |  |
| dashboard.persistence.annotations | object | `{}` |  |
| dashboard.persistence.enabled | bool | `false` |  |
| dashboard.persistence.existingClaim | string | `""` |  |
| dashboard.persistence.mountPath | string | `"/app/data"` |  |
| dashboard.persistence.size | string | `"1Gi"` |  |
| dashboard.persistence.storageClassName | string | `""` |  |
| dashboard.podSecurityContext.fsGroup | int | `65532` |  |
| dashboard.readonly | bool | `false` |  |
| dashboard.replicaCount | int | `1` |  |
| dashboard.resources.limits.cpu | string | `"500m"` |  |
| dashboard.resources.limits.memory | string | `"512Mi"` |  |
| dashboard.resources.requests.cpu | string | `"100m"` |  |
| dashboard.resources.requests.memory | string | `"128Mi"` |  |
| dashboard.routerTLS.ca.existingSecret | string | `""` |  |
| dashboard.routerTLS.ca.existingSecretKey | string | `"ca.crt"` |  |
| dashboard.service.port | int | `8700` |  |
| dashboard.service.targetPort | int | `8700` |  |
| dashboard.service.type | string | `"ClusterIP"` |  |
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
| dependencies.semanticCache.milvus.development.verboseErrors | bool | `true` |  |
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
| dependencies.semanticCache.redis.development.verboseErrors | bool | `true` |  |
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
| envFromSecrets | list | `[]` |  |
| env[0].name | string | `"LD_LIBRARY_PATH"` |  |
| env[0].value | string | `"/app/lib"` |  |
| env[1].name | string | `"HF_TOKEN"` |  |
| env[1].valueFrom.secretKeyRef.key | string | `"token"` |  |
| env[1].valueFrom.secretKeyRef.name | string | `"hf-token-secret"` |  |
| env[1].valueFrom.secretKeyRef.optional | bool | `true` |  |
| env[2].name | string | `"HUGGINGFACE_HUB_TOKEN"` |  |
| env[2].valueFrom.secretKeyRef.key | string | `"token"` |  |
| env[2].valueFrom.secretKeyRef.name | string | `"hf-token-secret"` |  |
| env[2].valueFrom.secretKeyRef.optional | bool | `true` |  |
| extraEnv | list | `[]` |  |
| extraVolumeMounts | list | `[]` |  |
| extraVolumes | list | `[]` |  |
| fullnameOverride | string | `""` |  |
| global.imageRegistry | string | `""` |  |
| global.namespace | string | `""` |  |
| grafana.image.tag | string | `"11.5.1"` |  |
| grafana.sidecar.datasources.enabled | bool | `true` |  |
| image.pullPolicy | string | `"IfNotPresent"` |  |
| image.repository | string | `"ghcr.io/vllm-project/semantic-router/extproc"` |  |
| image.tag | string | `""` |  |
| imagePullSecrets | list | `[]` |  |
| ingress.annotations | object | `{}` |  |
| ingress.className | string | `""` |  |
| ingress.enabled | bool | `false` |  |
| ingress.hosts[0].host | string | `"semantic-router.local"` |  |
| ingress.hosts[0].paths[0].path | string | `"/"` |  |
| ingress.hosts[0].paths[0].pathType | string | `"Prefix"` |  |
| ingress.hosts[0].paths[0].servicePort | int | `8080` |  |
| ingress.tls | list | `[]` |  |
| jaeger.allInOne.image.tag | string | `"latest"` |  |
| knowledgeBases.enabled | bool | `false` |  |
| knowledgeBases.existingConfigMap | string | `""` |  |
| knowledgeBases.mountPath | string | `"/app/config/knowledge_bases"` |  |
| livenessProbe.enabled | bool | `true` |  |
| livenessProbe.failureThreshold | int | `5` |  |
| livenessProbe.initialDelaySeconds | int | `30` |  |
| livenessProbe.periodSeconds | int | `30` |  |
| livenessProbe.timeoutSeconds | int | `10` |  |
| nameOverride | string | `""` |  |
| networkPolicy.enabled | string | `nil` |  |
| networkPolicy.ingress.backendDispatchPeers | list | `[]` |  |
| networkPolicy.ingress.extProcPeers | list | `[]` |  |
| networkPolicy.ingress.managementPeers | list | `[]` |  |
| networkPolicy.ingress.metricsPeers | list | `[]` |  |
| nodeSelector | object | `{}` |  |
| observability.alerts.enabled | bool | `false` |  |
| observability.alerts.labels | object | `{}` |  |
| observability.alerts.thresholds.cacheHitRate | float | `0.2` |  |
| observability.alerts.thresholds.completionLatencyP95Seconds | int | `30` |  |
| observability.alerts.thresholds.inflightRequests | int | `50` |  |
| observability.alerts.thresholds.requestErrorRate | float | `0.05` |  |
| observability.alerts.thresholds.routingLatencyP95Seconds | float | `0.1` |  |
| observability.alerts.thresholds.tpotP95Seconds | float | `0.25` |  |
| observability.alerts.thresholds.ttftP95Seconds | int | `5` |  |
| persistence.accessMode | string | `"ReadWriteOnce"` |  |
| persistence.annotations | object | `{}` |  |
| persistence.enabled | bool | `true` |  |
| persistence.existingClaim | string | `""` |  |
| persistence.size | string | `"10Gi"` |  |
| persistence.storageClassName | string | `"standard"` |  |
| podAnnotations | object | `{}` |  |
| podDisruptionBudget.enabled | string | `nil` |  |
| podDisruptionBudget.minAvailable | int | `1` |  |
| podSecurityContext | object | `{}` |  |
| prometheus.server.image.tag | string | `"v2.53.0"` |  |
| readinessProbe.enabled | bool | `true` |  |
| readinessProbe.failureThreshold | int | `5` |  |
| readinessProbe.initialDelaySeconds | int | `30` |  |
| readinessProbe.periodSeconds | int | `30` |  |
| readinessProbe.timeoutSeconds | int | `10` |  |
| replicaCount | int | `1` |  |
| resources.limits.cpu | string | `"2"` |  |
| resources.limits.memory | string | `"7Gi"` |  |
| resources.requests.cpu | string | `"1"` |  |
| resources.requests.memory | string | `"3Gi"` |  |
| response-api-redis.architecture | string | `"standalone"` |  |
| response-api-redis.auth.enabled | bool | `false` |  |
| safetyGuards.rejectMultiReplicaLocalLearningState | bool | `true` |  |
| securityContext.allowPrivilegeEscalation | bool | `false` |  |
| securityContext.runAsNonRoot | bool | `false` |  |
| semantic-cache-milvus.cluster.enabled | bool | `false` |  |
| semantic-cache-redis.architecture | string | `"standalone"` |  |
| semantic-cache-redis.auth.enabled | bool | `false` |  |
| service.api.port | int | `8080` |  |
| service.api.protocol | string | `"TCP"` |  |
| service.api.targetPort | int | `8080` |  |
| service.grpc.port | int | `50051` |  |
| service.grpc.protocol | string | `"TCP"` |  |
| service.grpc.targetPort | int | `50051` |  |
| service.management.port | int | `8080` |  |
| service.metrics.enabled | bool | `true` |  |
| service.metrics.port | int | `9190` |  |
| service.metrics.protocol | string | `"TCP"` |  |
| service.metrics.targetPort | int | `9190` |  |
| service.type | string | `"ClusterIP"` |  |
| serviceAccount.annotations | object | `{}` |  |
| serviceAccount.create | bool | `true` |  |
| serviceAccount.name | string | `""` |  |
| startupProbe.enabled | bool | `true` |  |
| startupProbe.failureThreshold | int | `360` |  |
| startupProbe.periodSeconds | int | `10` |  |
| startupProbe.timeoutSeconds | int | `5` |  |
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
| topologySpread.enabled | string | `nil` |  |
| topologySpread.maxSkew | int | `1` |  |
| topologySpread.topologyKey | string | `"kubernetes.io/hostname"` |  |
| topologySpread.whenUnsatisfiable | string | `"ScheduleAnyway"` |  |
