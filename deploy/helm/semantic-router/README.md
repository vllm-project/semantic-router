# semantic-router

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: latest](https://img.shields.io/badge/AppVersion-latest-informational?style=flat-square)

A Helm chart for deploying Semantic Router - an intelligent routing system for LLM applications

**Homepage:** <https://github.com/vllm-project/semantic-router>

## Dependencies and one-click deployment

This Helm chart supports deploying the full semantic-router stack via dependencies, including semantic cache and observability components. Dependencies are disabled by default; enable them as needed.

### Dependency toggles (`values.yaml`)

- Semantic cache dependency (Redis or Milvus)
- Response API dependency (Milvus or Redis)
- Observability dependency (Jaeger / Prometheus / Grafana)

> Note: The Redis storage backend for the Response API is not implemented yet. Enabling it will cause startup failures.

### Example

```
dependencies:
  semanticCache:
    redis:
      enabled: true
  responseApi:
    milvus:
      enabled: true
  observability:
    jaeger:
      enabled: true
    prometheus:
      enabled: true
    grafana:
      enabled: true
```

## CRD Management

This Helm chart includes Custom Resource Definitions (CRDs) in the `crds/` directory:

- `vllm.ai_intelligentpools.yaml` - IntelligentPool CRD
- `vllm.ai_intelligentroutes.yaml` - IntelligentRoute CRD

### Generating CRDs

CRDs are automatically generated from Go type definitions using `controller-gen`. To regenerate CRDs:

```bash
# From the repository root
make generate-crd
```

This command will:

1. Generate CRDs from `src/semantic-router/pkg/apis/vllm.ai/v1alpha1` types
2. Output to `deploy/kubernetes/crds/`
3. Copy to `deploy/helm/semantic-router/crds/` for Helm chart

### CRD Installation

CRDs in the `crds/` directory are automatically installed by Helm:

- Installed **before** other resources during `helm install`
- **Not managed** by Helm (no Helm labels/annotations)
- **Not updated** during `helm upgrade` (must be updated manually)
- **Not deleted** during `helm uninstall` (protects custom resources)

To manually update CRDs:

```bash
kubectl apply -f deploy/helm/semantic-router/crds/
```

## Maintainers

| Name | Email | Url |
| ---- | ------ | --- |
| Semantic Router Team |  | <https://github.com/vllm-project/semantic-router> |

## Source Code

- <https://github.com/vllm-project/semantic-router>

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| affinity | object | `{}` |  |
| args[0] | string | `"--secure=true"` |  |
| autoscaling.enabled | bool | `false` | Enable horizontal pod autoscaling |
| autoscaling.maxReplicas | int | `10` | Maximum number of replicas |
| autoscaling.minReplicas | int | `1` | Minimum number of replicas |
| autoscaling.targetCPUUtilizationPercentage | int | `80` | Target CPU utilization percentage |
| config.api.batch_classification.concurrency_threshold | int | `5` |  |
| config.api.batch_classification.max_batch_size | int | `100` |  |
| config.api.batch_classification.max_concurrency | int | `8` |  |
| config.api.batch_classification.metrics.detailed_goroutine_tracking | bool | `true` |  |
| config.api.batch_classification.metrics.duration_buckets[0] | float | `0.001` |  |
| config.api.batch_classification.metrics.duration_buckets[10] | int | `5` |  |
| config.api.batch_classification.metrics.duration_buckets[11] | int | `10` |  |
| config.api.batch_classification.metrics.duration_buckets[12] | int | `30` |  |
| config.api.batch_classification.metrics.duration_buckets[1] | float | `0.005` |  |
| config.api.batch_classification.metrics.duration_buckets[2] | float | `0.01` |  |
| config.api.batch_classification.metrics.duration_buckets[3] | float | `0.025` |  |
| config.api.batch_classification.metrics.duration_buckets[4] | float | `0.05` |  |
| config.api.batch_classification.metrics.duration_buckets[5] | float | `0.1` |  |
| config.api.batch_classification.metrics.duration_buckets[6] | float | `0.25` |  |
| config.api.batch_classification.metrics.duration_buckets[7] | float | `0.5` |  |
| config.api.batch_classification.metrics.duration_buckets[8] | int | `1` |  |
| config.api.batch_classification.metrics.duration_buckets[9] | float | `2.5` |  |
| config.api.batch_classification.metrics.enabled | bool | `true` |  |
| config.api.batch_classification.metrics.high_resolution_timing | bool | `false` |  |
| config.api.batch_classification.metrics.sample_rate | float | `1` |  |
| config.api.batch_classification.metrics.size_buckets[0] | int | `1` |  |
| config.api.batch_classification.metrics.size_buckets[1] | int | `2` |  |
| config.api.batch_classification.metrics.size_buckets[2] | int | `5` |  |
| config.api.batch_classification.metrics.size_buckets[3] | int | `10` |  |
| config.api.batch_classification.metrics.size_buckets[4] | int | `20` |  |
| config.api.batch_classification.metrics.size_buckets[5] | int | `50` |  |
| config.api.batch_classification.metrics.size_buckets[6] | int | `100` |  |
| config.api.batch_classification.metrics.size_buckets[7] | int | `200` |  |
| config.bert_model.model_id | string | `"models/all-MiniLM-L12-v2"` |  |
| config.bert_model.threshold | float | `0.6` |  |
| config.bert_model.use_cpu | bool | `true` |  |
| config.categories[0].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[0].model_scores[0].score | float | `0.7` |  |
| config.categories[0].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[0].name | string | `"business"` |  |
| config.categories[0].system_prompt | string | `"You are a senior business consultant and strategic advisor with expertise in corporate strategy, operations management, financial analysis, marketing, and organizational development. Provide practical, actionable business advice backed by proven methodologies and industry best practices. Consider market dynamics, competitive landscape, and stakeholder interests in your recommendations."` |  |
| config.categories[10].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[10].model_scores[0].score | float | `0.7` |  |
| config.categories[10].model_scores[0].use_reasoning | bool | `true` |  |
| config.categories[10].name | string | `"physics"` |  |
| config.categories[10].system_prompt | string | `"You are a physics expert with deep understanding of physical laws and phenomena. Provide clear explanations with mathematical derivations when appropriate."` |  |
| config.categories[11].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[11].model_scores[0].score | float | `0.6` |  |
| config.categories[11].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[11].name | string | `"computer science"` |  |
| config.categories[11].system_prompt | string | `"You are a computer science expert with knowledge of algorithms, data structures, programming languages, and software engineering. Provide clear, practical solutions with code examples when helpful."` |  |
| config.categories[12].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[12].model_scores[0].score | float | `0.5` |  |
| config.categories[12].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[12].name | string | `"philosophy"` |  |
| config.categories[12].system_prompt | string | `"You are a philosophy expert with comprehensive knowledge of philosophical traditions, ethical theories, logic, metaphysics, epistemology, political philosophy, and the history of philosophical thought. Engage with complex philosophical questions by presenting multiple perspectives, analyzing arguments rigorously, and encouraging critical thinking. Draw connections between philosophical concepts and contemporary issues while maintaining intellectual honesty about the complexity and ongoing nature of philosophical debates."` |  |
| config.categories[13].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[13].model_scores[0].score | float | `0.7` |  |
| config.categories[13].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[13].name | string | `"engineering"` |  |
| config.categories[13].system_prompt | string | `"You are an engineering expert with knowledge across multiple engineering disciplines including mechanical, electrical, civil, chemical, software, and systems engineering. Apply engineering principles, design methodologies, and problem-solving approaches to provide practical solutions. Consider safety, efficiency, sustainability, and cost-effectiveness in your recommendations. Use technical precision while explaining concepts clearly, and emphasize the importance of proper engineering practices and standards."` |  |
| config.categories[1].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[1].model_scores[0].score | float | `0.4` |  |
| config.categories[1].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[1].name | string | `"law"` |  |
| config.categories[1].system_prompt | string | `"You are a knowledgeable legal expert with comprehensive understanding of legal principles, case law, statutory interpretation, and legal procedures across multiple jurisdictions. Provide accurate legal information and analysis while clearly stating that your responses are for informational purposes only and do not constitute legal advice. Always recommend consulting with qualified legal professionals for specific legal matters."` |  |
| config.categories[2].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[2].model_scores[0].score | float | `0.6` |  |
| config.categories[2].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[2].name | string | `"psychology"` |  |
| config.categories[2].semantic_cache_enabled | bool | `true` |  |
| config.categories[2].semantic_cache_similarity_threshold | float | `0.92` |  |
| config.categories[2].system_prompt | string | `"You are a psychology expert with deep knowledge of cognitive processes, behavioral patterns, mental health, developmental psychology, social psychology, and therapeutic approaches. Provide evidence-based insights grounded in psychological research and theory. When discussing mental health topics, emphasize the importance of professional consultation and avoid providing diagnostic or therapeutic advice."` |  |
| config.categories[3].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[3].model_scores[0].score | float | `0.9` |  |
| config.categories[3].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[3].name | string | `"biology"` |  |
| config.categories[3].system_prompt | string | `"You are a biology expert with comprehensive knowledge spanning molecular biology, genetics, cell biology, ecology, evolution, anatomy, physiology, and biotechnology. Explain biological concepts with scientific accuracy, use appropriate terminology, and provide examples from current research. Connect biological principles to real-world applications and emphasize the interconnectedness of biological systems."` |  |
| config.categories[4].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[4].model_scores[0].score | float | `0.6` |  |
| config.categories[4].model_scores[0].use_reasoning | bool | `true` |  |
| config.categories[4].name | string | `"chemistry"` |  |
| config.categories[4].system_prompt | string | `"You are a chemistry expert specializing in chemical reactions, molecular structures, and laboratory techniques. Provide detailed, step-by-step explanations."` |  |
| config.categories[5].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[5].model_scores[0].score | float | `0.7` |  |
| config.categories[5].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[5].name | string | `"history"` |  |
| config.categories[5].system_prompt | string | `"You are a historian with expertise across different time periods and cultures. Provide accurate historical context and analysis."` |  |
| config.categories[6].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[6].model_scores[0].score | float | `0.7` |  |
| config.categories[6].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[6].name | string | `"other"` |  |
| config.categories[6].semantic_cache_enabled | bool | `true` |  |
| config.categories[6].semantic_cache_similarity_threshold | float | `0.75` |  |
| config.categories[6].system_prompt | string | `"You are a helpful and knowledgeable assistant. Provide accurate, helpful responses across a wide range of topics."` |  |
| config.categories[7].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[7].model_scores[0].score | float | `0.5` |  |
| config.categories[7].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[7].name | string | `"health"` |  |
| config.categories[7].semantic_cache_enabled | bool | `true` |  |
| config.categories[7].semantic_cache_similarity_threshold | float | `0.95` |  |
| config.categories[7].system_prompt | string | `"You are a health and medical information expert with knowledge of anatomy, physiology, diseases, treatments, preventive care, nutrition, and wellness. Provide accurate, evidence-based health information while emphasizing that your responses are for educational purposes only and should never replace professional medical advice, diagnosis, or treatment. Always encourage users to consult healthcare professionals for medical concerns and emergencies."` |  |
| config.categories[8].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[8].model_scores[0].score | float | `1` |  |
| config.categories[8].model_scores[0].use_reasoning | bool | `false` |  |
| config.categories[8].name | string | `"economics"` |  |
| config.categories[8].system_prompt | string | `"You are an economics expert with deep understanding of microeconomics, macroeconomics, econometrics, financial markets, monetary policy, fiscal policy, international trade, and economic theory. Analyze economic phenomena using established economic principles, provide data-driven insights, and explain complex economic concepts in accessible terms. Consider both theoretical frameworks and real-world applications in your responses."` |  |
| config.categories[9].model_scores[0].model | string | `"qwen3"` |  |
| config.categories[9].model_scores[0].score | float | `1` |  |
| config.categories[9].model_scores[0].use_reasoning | bool | `true` |  |
| config.categories[9].name | string | `"math"` |  |
| config.categories[9].system_prompt | string | `"You are a mathematics expert. Provide step-by-step solutions, show your work clearly, and explain mathematical concepts in an understandable way."` |  |
| config.classifier.category_model.category_mapping_path | string | `"models/category_classifier_modernbert-base_model/category_mapping.json"` |  |
| config.classifier.category_model.model_id | string | `"models/category_classifier_modernbert-base_model"` |  |
| config.classifier.category_model.threshold | float | `0.6` |  |
| config.classifier.category_model.use_cpu | bool | `true` |  |
| config.classifier.category_model.use_modernbert | bool | `true` |  |
| config.classifier.pii_model.model_id | string | `"models/pii_classifier_modernbert-base_presidio_token_model"` |  |
| config.classifier.pii_model.pii_mapping_path | string | `"models/pii_classifier_modernbert-base_presidio_token_model/pii_type_mapping.json"` |  |
| config.classifier.pii_model.threshold | float | `0.7` |  |
| config.classifier.pii_model.use_cpu | bool | `true` |  |
| config.classifier.pii_model.use_modernbert | bool | `true` |  |
| config.default_model | string | `"qwen3"` |  |
| config.default_reasoning_effort | string | `"high"` |  |
| config.model_config.qwen3.pii_policy.allow_by_default | bool | `true` |  |
| config.model_config.qwen3.preferred_endpoints[0] | string | `"endpoint1"` |  |
| config.model_config.qwen3.reasoning_family | string | `"qwen3"` |  |
| config.observability.tracing.enabled | bool | `true` |  |
| config.observability.tracing.exporter.endpoint | string | `"jaeger:4317"` |  |
| config.observability.tracing.exporter.insecure | bool | `true` |  |
| config.observability.tracing.exporter.type | string | `"otlp"` |  |
| config.observability.tracing.provider | string | `"opentelemetry"` |  |
| config.observability.tracing.resource.deployment_environment | string | `"development"` |  |
| config.observability.tracing.resource.service_name | string | `"vllm-semantic-router"` |  |
| config.observability.tracing.resource.service_version | string | `"v0.1.0"` |  |
| config.observability.tracing.sampling.rate | float | `1` |  |
| config.observability.tracing.sampling.type | string | `"always_on"` |  |
| config.prompt_guard.enabled | bool | `true` |  |
| config.prompt_guard.jailbreak_mapping_path | string | `"models/jailbreak_classifier_modernbert-base_model/jailbreak_type_mapping.json"` |  |
| config.prompt_guard.model_id | string | `"models/jailbreak_classifier_modernbert-base_model"` |  |
| config.prompt_guard.threshold | float | `0.7` |  |
| config.prompt_guard.use_cpu | bool | `true` |  |
| config.prompt_guard.use_modernbert | bool | `true` |  |
| config.reasoning_families.deepseek.parameter | string | `"thinking"` |  |
| config.reasoning_families.deepseek.type | string | `"chat_template_kwargs"` |  |
| config.reasoning_families.gpt-oss.parameter | string | `"reasoning_effort"` |  |
| config.reasoning_families.gpt-oss.type | string | `"reasoning_effort"` |  |
| config.reasoning_families.gpt.parameter | string | `"reasoning_effort"` |  |
| config.reasoning_families.gpt.type | string | `"reasoning_effort"` |  |
| config.reasoning_families.qwen3.parameter | string | `"enable_thinking"` |  |
| config.reasoning_families.qwen3.type | string | `"chat_template_kwargs"` |  |
| config.semantic_cache.backend_type | string | `"memory"` |  |
| config.semantic_cache.enabled | bool | `true` |  |
| config.semantic_cache.eviction_policy | string | `"fifo"` |  |
| config.semantic_cache.max_entries | int | `1000` |  |
| config.semantic_cache.similarity_threshold | float | `0.8` |  |
| config.semantic_cache.ttl_seconds | int | `3600` |  |
| config.tools.enabled | bool | `true` |  |
| config.tools.fallback_to_empty | bool | `true` |  |
| config.tools.similarity_threshold | float | `0.2` |  |
| config.tools.tools_db_path | string | `"config/tools_db.json"` |  |
| config.tools.top_k | int | `3` |  |
| config.vllm_endpoints[0].address | string | `"172.28.0.20"` |  |
| config.vllm_endpoints[0].name | string | `"endpoint1"` |  |
| config.vllm_endpoints[0].port | int | `8002` |  |
| config.vllm_endpoints[0].weight | int | `1` |  |
| env[0].name | string | `"LD_LIBRARY_PATH"` |  |
| env[0].value | string | `"/app/lib"` |  |
| fullnameOverride | string | `""` | Override the full name of the chart |
| global.namespace | string | `""` | Namespace for all resources (if not specified, uses Release.Namespace) |
| global.imageRegistry | string | `""` | Optional registry prefix applied to all images (e.g., mirror registry in China) |
| image.pullPolicy | string | `"IfNotPresent"` | Image pull policy |
| image.repository | string | `"ghcr.io/vllm-project/semantic-router/extproc"` | Image repository |
| image.tag | string | `"latest"` | Image tag (overrides the image tag whose default is the chart appVersion) |
| imagePullSecrets | list | `[]` | Image pull secrets for private registries |
| ingress.annotations | object | `{}` | Ingress annotations |
| ingress.className | string | `""` | Ingress class name |
| ingress.enabled | bool | `false` | Enable ingress |
| ingress.hosts | list | `[{"host":"semantic-router.local","paths":[{"path":"/","pathType":"Prefix","servicePort":8080}]}]` | Ingress hosts configuration |
| ingress.tls | list | `[]` | Ingress TLS configuration |
| initContainer.enabled | bool | `true` | Enable init container |
| initContainer.image | object | `{ "repository": "ghcr.io/vllm-project/semantic-router/model-downloader", "tag": "" (defaults to chart appVersion), "pullPolicy": "IfNotPresent" }` | Init container image |
| initContainer.models | list | `[{"name":"all-MiniLM-L12-v2","repo":"sentence-transformers/all-MiniLM-L12-v2"},{"name":"category_classifier_modernbert-base_model","repo":"LLM-Semantic-Router/category_classifier_modernbert-base_model"},{"name":"pii_classifier_modernbert-base_model","repo":"LLM-Semantic-Router/pii_classifier_modernbert-base_model"},{"name":"jailbreak_classifier_modernbert-base_model","repo":"LLM-Semantic-Router/jailbreak_classifier_modernbert-base_model"},{"name":"pii_classifier_modernbert-base_presidio_token_model","repo":"LLM-Semantic-Router/pii_classifier_modernbert-base_presidio_token_model"}]` | Models to download |
| initContainer.resources | object | `{"limits":{"cpu":"1000m","memory":"2Gi"},"requests":{"cpu":"500m","memory":"1Gi"}}` | Resource limits for init container |
| livenessProbe.enabled | bool | `true` | Enable liveness probe |
| livenessProbe.failureThreshold | int | `5` | Failure threshold |
| livenessProbe.initialDelaySeconds | int | `30` | Initial delay seconds |
| livenessProbe.periodSeconds | int | `30` | Period seconds |
| livenessProbe.timeoutSeconds | int | `10` | Timeout seconds |
| nameOverride | string | `""` | Override the name of the chart |
| nodeSelector | object | `{}` |  |
| persistence.accessMode | string | `"ReadWriteOnce"` | Access mode |
| persistence.annotations | object | `{}` | Annotations for PVC |
| persistence.enabled | bool | `true` | Enable persistent volume |
| persistence.existingClaim | string | `""` | Existing claim name (if provided, will use existing PVC instead of creating new one) |
| persistence.size | string | `"10Gi"` | Storage size |
| persistence.storageClassName | string | `"standard"` | Storage class name (use "-" for default storage class) |
| podAnnotations | object | `{}` |  |
| podSecurityContext | object | `{}` |  |
| readinessProbe.enabled | bool | `true` | Enable readiness probe |
| readinessProbe.failureThreshold | int | `5` | Failure threshold |
| readinessProbe.initialDelaySeconds | int | `30` | Initial delay seconds |
| readinessProbe.periodSeconds | int | `30` | Period seconds |
| readinessProbe.timeoutSeconds | int | `10` | Timeout seconds |
| replicaCount | int | `1` | Number of replicas for the deployment |
| resources.limits | object | `{"cpu":"2","memory":"6Gi"}` | Resource limits |
| resources.requests | object | `{"cpu":"1","memory":"3Gi"}` | Resource requests |
| securityContext.allowPrivilegeEscalation | bool | `false` | Allow privilege escalation |
| securityContext.runAsNonRoot | bool | `false` | Run as non-root user |
| service.api.port | int | `8080` | HTTP API port number |
| service.api.protocol | string | `"TCP"` | HTTP API protocol |
| service.api.targetPort | int | `8080` | HTTP API target port |
| service.grpc.port | int | `50051` | gRPC port number |
| service.grpc.protocol | string | `"TCP"` | gRPC protocol |
| service.grpc.targetPort | int | `50051` | gRPC target port |
| service.metrics.enabled | bool | `true` | Enable metrics service |
| service.metrics.port | int | `9190` | Metrics port number |
| service.metrics.protocol | string | `"TCP"` | Metrics protocol |
| service.metrics.targetPort | int | `9190` | Metrics target port |
| service.type | string | `"ClusterIP"` | Service type |
| serviceAccount.annotations | object | `{}` | Annotations to add to the service account |
| serviceAccount.create | bool | `true` | Specifies whether a service account should be created |
| serviceAccount.name | string | `""` | The name of the service account to use |
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
