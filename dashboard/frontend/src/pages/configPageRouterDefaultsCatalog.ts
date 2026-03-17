import type { RouterSystemKey } from './configPageRouterDefaultsSupport'
import type {
  APIConfig,
  AuthzConfig,
  CanonicalHallucinationModuleConfig,
  CanonicalSystemModels,
  EmbeddingModelsConfig,
  FeedbackDetectorConfig,
  LooperConfig,
  MemoryConfig,
  ModelSelectionConfig,
  ModalityDetectorConfig,
  ObservabilityConfig,
  PromptCompressionConfig,
  RateLimitConfig,
  ResponseAPIConfig,
  RouterCoreConfig,
  RouterReplayConfig,
  SemanticCacheConfig,
  ToolIntegrationConfig,
  VectorStoreConfig,
} from './configPageSupport'

export const PYTHON_ROUTER_KEYS: RouterSystemKey[] = [
  'router_core',
  'response_api',
  'router_replay',
  'authz',
  'ratelimit',
  'memory',
  'semantic_cache',
  'vector_store',
  'tools',
  'prompt_guard',
  'classifier',
  'hallucination_mitigation',
  'feedback_detector',
  'external_models',
  'system_models',
  'embedding_models',
  'prompt_compression',
  'modality_detector',
  'observability',
  'looper',
  'clear_route_cache',
  'model_selection',
  'api',
]

export const OPTIONAL_ROUTER_KEYS: RouterSystemKey[] = []

export const DEFAULT_SECTIONS: Record<RouterSystemKey, unknown> = {
  router_core: {
    config_source: 'file',
    auto_model_name: 'MoM',
    include_config_models_in_list: false,
    clear_route_cache: true,
    model_selection: {
      enabled: true,
      method: 'knn',
    },
  } satisfies RouterCoreConfig,
  response_api: {
    enabled: true,
    store_backend: 'memory',
    ttl_seconds: 86400,
    max_responses: 1000,
  } satisfies ResponseAPIConfig,
  router_replay: {
    store_backend: 'memory',
    ttl_seconds: 2592000,
    async_writes: false,
  } satisfies RouterReplayConfig,
  authz: {
    fail_open: false,
    identity: {
      user_id_header: 'x-authz-user-id',
      user_groups_header: 'x-authz-user-groups',
    },
    providers: [],
  } satisfies AuthzConfig,
  ratelimit: {
    fail_open: false,
    providers: [],
  } satisfies RateLimitConfig,
  memory: {
    enabled: false,
    auto_store: false,
    milvus: {
      collection: 'agentic_memory',
      dimension: 384,
    },
    default_retrieval_limit: 5,
    default_similarity_threshold: 0.7,
    extraction_batch_size: 10,
  } satisfies MemoryConfig,
  semantic_cache: {
    enabled: true,
    backend_type: 'memory',
    max_entries: 1000,
    ttl_seconds: 3600,
    eviction_policy: 'fifo',
  } satisfies SemanticCacheConfig,
  vector_store: {
    enabled: false,
    backend_type: 'memory',
    file_storage_dir: '/var/lib/vsr/data',
    max_file_size_mb: 50,
    embedding_model: 'mmbert',
    embedding_dimension: 384,
    ingestion_workers: 2,
    supported_formats: ['.txt', '.md', '.json', '.csv', '.html'],
    memory: {
      max_entries_per_store: 100000,
    },
  } satisfies VectorStoreConfig,
  tools: {
    enabled: false,
    top_k: 3,
    tools_db_path: 'config/tools_db.json',
    fallback_to_empty: true,
  } satisfies ToolIntegrationConfig,
  prompt_guard: {
    enabled: true,
    model_ref: 'prompt_guard',
    threshold: 0.7,
    use_cpu: true,
    use_mmbert_32k: true,
    jailbreak_mapping_path: 'models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json',
  },
  classifier: {
    domain: {
      model_ref: 'domain_classifier',
      threshold: 0.5,
      use_cpu: true,
      use_mmbert_32k: true,
      category_mapping_path: 'models/mmbert32k-intent-classifier-merged/category_mapping.json',
    },
    pii: {
      model_ref: 'pii_classifier',
      threshold: 0.9,
      use_cpu: true,
      use_mmbert_32k: true,
      pii_mapping_path: 'models/mmbert32k-pii-detector-merged/pii_type_mapping.json',
    },
    preference: {
      use_contrastive: false,
    },
  },
  hallucination_mitigation: {
    enabled: false,
    fact_check: {
      model_ref: 'fact_check_classifier',
      threshold: 0.6,
      use_cpu: true,
      use_mmbert_32k: true,
    },
    detector: {
      model_ref: 'hallucination_detector',
      threshold: 0.8,
      use_cpu: true,
      min_span_length: 2,
      min_span_confidence: 0.6,
      context_window_size: 50,
      enable_nli_filtering: true,
      nli_entailment_threshold: 0.75,
    },
    explainer: {
      model_ref: 'hallucination_explainer',
      threshold: 0.9,
      use_cpu: true,
    },
  } satisfies CanonicalHallucinationModuleConfig,
  feedback_detector: {
    enabled: true,
    model_ref: 'feedback_detector',
    threshold: 0.7,
    use_cpu: true,
    use_mmbert_32k: true,
  } satisfies FeedbackDetectorConfig & { model_ref?: string },
  external_models: [],
  system_models: {
    prompt_guard: 'models/mmbert32k-jailbreak-detector-merged',
    domain_classifier: 'models/mmbert32k-intent-classifier-merged',
    pii_classifier: 'models/mmbert32k-pii-detector-merged',
    fact_check_classifier: 'models/mmbert32k-factcheck-classifier-merged',
    hallucination_detector: 'models/mom-halugate-detector',
    hallucination_explainer: 'models/mom-halugate-explainer',
    feedback_detector: 'models/mmbert32k-feedback-detector-merged',
  } satisfies CanonicalSystemModels,
  embedding_models: {
    qwen3_model_path: '',
    gemma_model_path: '',
    mmbert_model_path: 'models/mom-embedding-ultra',
    multimodal_model_path: '',
    bert_model_path: '',
    use_cpu: true,
    embedding_config: {
      model_type: 'mmbert',
      preload_embeddings: true,
      target_dimension: 768,
      target_layer: 22,
      enable_soft_matching: true,
      min_score_threshold: 0.5,
    },
  } satisfies EmbeddingModelsConfig,
  prompt_compression: {
    enabled: false,
  } satisfies PromptCompressionConfig,
  modality_detector: {
    enabled: false,
  } satisfies ModalityDetectorConfig,
  observability: {
    metrics: {
      enabled: true,
    },
    tracing: {
      enabled: true,
      provider: 'opentelemetry',
      exporter: {
        type: 'otlp',
        endpoint: 'vllm-sr-jaeger:4317',
        insecure: true,
      },
      sampling: {
        type: 'always_on',
        rate: 1.0,
      },
      resource: {
        service_name: 'vllm-sr',
        service_version: 'v0.3.0',
        deployment_environment: 'development',
      },
    },
  } satisfies ObservabilityConfig,
  looper: {
    endpoint: 'http://localhost:8899/v1/chat/completions',
    timeout_seconds: 1200,
    headers: {},
  } satisfies LooperConfig,
  clear_route_cache: true,
  model_selection: {
    enabled: true,
    method: 'knn',
  } satisfies ModelSelectionConfig,
  api: {
    batch_classification: {
      metrics: {},
    },
  } satisfies APIConfig,
}

export const SECTION_META: Record<RouterSystemKey, { title: string; eyebrow: string; description: string }> = {
  router_core: {
    title: 'Router Core',
    eyebrow: 'Router',
    description: 'Core router behavior, config source, startup cache handling, and model selection strategy.',
  },
  response_api: {
    title: 'Response API',
    eyebrow: 'Services',
    description: 'Conversation chaining and persistence defaults for the OpenAI Responses API surface.',
  },
  router_replay: {
    title: 'Router Replay',
    eyebrow: 'Services',
    description: 'Persistence policy for replay records written by replay-enabled decision plugins.',
  },
  authz: {
    title: 'Authorization',
    eyebrow: 'Services',
    description: 'Header identity extraction and external authorization provider wiring.',
  },
  ratelimit: {
    title: 'Rate Limiting',
    eyebrow: 'Services',
    description: 'Per-user, group, and model request throttling rules enforced by the router.',
  },
  memory: {
    title: 'Agentic Memory',
    eyebrow: 'Stores',
    description: 'Cross-session memory extraction, storage, retrieval thresholds, and reflection policy.',
  },
  semantic_cache: {
    title: 'Semantic Cache',
    eyebrow: 'Stores',
    description: 'Similarity cache backend, retention policy, and embedding-backed cache behavior.',
  },
  vector_store: {
    title: 'Vector Store',
    eyebrow: 'Stores',
    description: 'Document ingestion, vector backend selection, and file-backed knowledge-store defaults.',
  },
  tools: {
    title: 'Tool Selection',
    eyebrow: 'Integrations',
    description: 'Automatic tool ranking, similarity thresholds, and tool database lookup settings.',
  },
  prompt_guard: {
    title: 'Prompt Guard',
    eyebrow: 'Model Catalog',
    description: 'Prompt-injection detection module settings and system-model binding.',
  },
  classifier: {
    title: 'Classifier Modules',
    eyebrow: 'Model Catalog',
    description: 'Domain, PII, MCP, and preference-classification module settings.',
  },
  hallucination_mitigation: {
    title: 'Hallucination Mitigation',
    eyebrow: 'Model Catalog',
    description: 'Fact-check, detector, and explainer modules used for hallucination review.',
  },
  feedback_detector: {
    title: 'Feedback Detector',
    eyebrow: 'Model Catalog',
    description: 'Feedback classification defaults for routing-aware user correction flows.',
  },
  external_models: {
    title: 'External Models',
    eyebrow: 'Model Catalog',
    description: 'Optional external LLM integrations used by router-owned auxiliary workflows.',
  },
  system_models: {
    title: 'System Model Bindings',
    eyebrow: 'Model Catalog',
    description: 'Stable capability-to-model bindings for the router-owned built-in model catalog.',
  },
  embedding_models: {
    title: 'Embedding Models',
    eyebrow: 'Model Catalog',
    description: 'Semantic embedding paths and embedding optimization defaults for router-owned models.',
  },
  prompt_compression: {
    title: 'Prompt Compression',
    eyebrow: 'Model Catalog',
    description: 'Optional compression pass applied before signal extraction and downstream routing.',
  },
  modality_detector: {
    title: 'Modality Detector',
    eyebrow: 'Model Catalog',
    description: 'Prompt modality detection strategy, classifier settings, and keyword fallback behavior.',
  },
  observability: {
    title: 'Observability',
    eyebrow: 'Services',
    description: 'Metrics and tracing defaults emitted by the router runtime.',
  },
  looper: {
    title: 'Looper',
    eyebrow: 'Integrations',
    description: 'Multi-model execution endpoint, timeout policy, and per-request headers.',
  },
  clear_route_cache: {
    title: 'Route Cache Reset',
    eyebrow: 'Router',
    description: 'Controls whether the route cache is cleared on startup or config reload.',
  },
  model_selection: {
    title: 'Model Selection',
    eyebrow: 'Router',
    description: 'Selection strategy and ML-backed model-ranking settings used by the router core.',
  },
  api: {
    title: 'API Service',
    eyebrow: 'Services',
    description: 'Batch-classification API limits and metrics settings when configured.',
  },
}
