import type { RouterSystemKey } from './configPageRouterDefaultsSupport'
import type {
  APIConfig,
  EmbeddingModelsConfig,
  FeedbackDetectorConfig,
  HallucinationMitigationConfig,
  LooperConfig,
  MemoryConfig,
  ModelConfig,
  ModelSelectionConfig,
  ObservabilityConfig,
  ResponseAPIConfig,
  RouterReplayConfig,
  SemanticCacheConfig,
} from './configPageSupport'

export const PYTHON_ROUTER_KEYS: RouterSystemKey[] = [
  'response_api',
  'router_replay',
  'memory',
  'semantic_cache',
  'tools',
  'prompt_guard',
  'classifier',
  'hallucination_mitigation',
  'feedback_detector',
  'external_models',
  'embedding_models',
  'observability',
  'looper',
  'clear_route_cache',
  'model_selection',
]

export const OPTIONAL_ROUTER_KEYS: RouterSystemKey[] = ['api', 'bert_model']

export const DEFAULT_SECTIONS: Record<RouterSystemKey, unknown> = {
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
  memory: {
    enabled: false,
    auto_store: false,
    milvus: {
      address: '',
      collection: 'agentic_memory',
      dimension: 384,
    },
    embedding: {
      model: 'all-MiniLM-L6-v2',
      dimension: 384,
    },
    default_retrieval_limit: 5,
    default_similarity_threshold: 0.7,
    extraction_batch_size: 10,
  } satisfies MemoryConfig,
  semantic_cache: {
    enabled: true,
    backend_type: 'memory',
    similarity_threshold: 0.8,
    max_entries: 1000,
    ttl_seconds: 3600,
    eviction_policy: 'fifo',
    use_hnsw: true,
    hnsw_m: 16,
    hnsw_ef_construction: 200,
  } satisfies SemanticCacheConfig,
  tools: {
    enabled: false,
    top_k: 3,
    similarity_threshold: 0.2,
    tools_db_path: 'config/tools_db.json',
    fallback_to_empty: true,
  },
  prompt_guard: {
    enabled: true,
    use_mmbert_32k: true,
    model_id: 'models/mmbert32k-jailbreak-detector-merged',
    jailbreak_mapping_path: 'models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json',
    threshold: 0.7,
    use_cpu: true,
  },
  classifier: {
    category_model: {
      model_id: 'models/mmbert32k-intent-classifier-merged',
      use_mmbert_32k: true,
      threshold: 0.5,
      use_cpu: true,
      category_mapping_path: 'models/mmbert32k-intent-classifier-merged/category_mapping.json',
    },
    pii_model: {
      model_id: 'models/mmbert32k-pii-detector-merged',
      use_mmbert_32k: true,
      threshold: 0.9,
      use_cpu: true,
      pii_mapping_path: 'models/mmbert32k-pii-detector-merged/pii_type_mapping.json',
    },
  },
  hallucination_mitigation: {
    enabled: false,
    fact_check_model: {
      model_id: 'models/mmbert32k-factcheck-classifier-merged',
      threshold: 0.6,
      use_cpu: true,
      use_mmbert_32k: true,
    },
    hallucination_model: {
      model_id: 'models/mom-halugate-detector',
      threshold: 0.8,
      use_cpu: true,
      min_span_length: 2,
      min_span_confidence: 0.6,
      context_window_size: 50,
      enable_nli_filtering: true,
      nli_entailment_threshold: 0.75,
    },
    nli_model: {
      model_id: 'models/mom-halugate-explainer',
      threshold: 0.9,
      use_cpu: true,
    },
  } satisfies HallucinationMitigationConfig,
  feedback_detector: {
    enabled: true,
    model_id: 'models/mmbert32k-feedback-detector-merged',
    threshold: 0.7,
    use_cpu: true,
    use_mmbert_32k: true,
  } satisfies FeedbackDetectorConfig,
  external_models: [],
  embedding_models: {
    mmbert_model_path: 'models/mom-embedding-ultra',
    use_cpu: true,
    hnsw_config: {
      model_type: 'mmbert',
      preload_embeddings: true,
      target_dimension: 768,
      target_layer: 22,
      enable_soft_matching: true,
      min_score_threshold: 0.5,
    },
  } satisfies EmbeddingModelsConfig,
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
        service_version: 'v0.2.0',
        deployment_environment: 'development',
      },
    },
  } satisfies ObservabilityConfig,
  looper: {
    enabled: true,
    endpoint: 'http://localhost:8899/v1/chat/completions',
    timeout_seconds: 1200,
    headers: {},
  } satisfies LooperConfig,
  clear_route_cache: true,
  model_selection: {
    enabled: true,
    default_algorithm: 'knn',
    llm_candidates_path: 'config/model_selection/llm_candidates.json',
    models_path: 'models/model_selection',
    training_data_path: 'config/model_selection/routing_training_data.jsonl',
    knn: {
      k: 5,
      weights: 'distance',
      metric: 'cosine',
      model_file: 'knn_model.json',
    },
    kmeans: {
      num_clusters: 8,
      efficiency_weight: 0.3,
      max_iterations: 100,
      model_file: 'kmeans_model.json',
    },
    svm: {
      kernel: 'rbf',
      c: 1.0,
      gamma: 'auto',
      model_file: 'svm_model.json',
    },
    custom_training: {
      enabled: false,
      custom_data_path: '',
      merge_with_pretrained: true,
      min_samples: 50,
    },
  } satisfies ModelSelectionConfig,
  api: {
    batch_classification: {
      max_batch_size: 100,
      concurrency_threshold: 10,
      max_concurrency: 5,
      metrics: {
        enabled: true,
        sample_rate: 0.1,
        detailed_goroutine_tracking: false,
        high_resolution_timing: true,
      },
    },
  } satisfies APIConfig,
  bert_model: {
    model_id: '',
    threshold: 0.8,
    use_cpu: true,
  } satisfies ModelConfig,
}

export const SECTION_META: Record<RouterSystemKey, { title: string; eyebrow: string; description: string }> = {
  response_api: {
    title: 'Response API',
    eyebrow: 'Runtime API',
    description: 'Conversation chaining and storage defaults for the OpenAI Responses API surface.',
  },
  router_replay: {
    title: 'Router Replay',
    eyebrow: 'Diagnostics',
    description: 'Storage policy for replay records written by the router_replay plugin.',
  },
  memory: {
    title: 'Agentic Memory',
    eyebrow: 'Memory',
    description: 'Cross-session memory defaults, Milvus connectivity, and retrieval thresholds.',
  },
  semantic_cache: {
    title: 'Semantic Cache',
    eyebrow: 'Caching',
    description: 'Similarity cache backend, retention policy, and HNSW acceleration defaults.',
  },
  tools: {
    title: 'Tool Selection',
    eyebrow: 'Runtime Assist',
    description: 'Tool auto-selection thresholds and tool database routing defaults.',
  },
  prompt_guard: {
    title: 'Prompt Guard',
    eyebrow: 'Safety',
    description: 'Global jailbreak detection defaults applied ahead of routing decisions.',
  },
  classifier: {
    title: 'Classifier Models',
    eyebrow: 'Safety',
    description: 'Category and PII classifier defaults shared by routing and guardrail flows.',
  },
  hallucination_mitigation: {
    title: 'Hallucination Mitigation',
    eyebrow: 'Safety',
    description: 'Fact-check, detector, and explainer defaults for hallucination review.',
  },
  feedback_detector: {
    title: 'Feedback Detector',
    eyebrow: 'Signals',
    description: 'Runtime defaults for classifying user feedback into router-facing labels.',
  },
  external_models: {
    title: 'External Models',
    eyebrow: 'LLM Seams',
    description: 'Optional external LLM integrations for memory, preference, and guardrail tasks.',
  },
  embedding_models: {
    title: 'Embedding Models',
    eyebrow: 'Embeddings',
    description: 'Shared embedding backends and HNSW preload behavior used across multiple features.',
  },
  observability: {
    title: 'Observability',
    eyebrow: 'Platform',
    description: 'Metrics and tracing defaults used by the local router runtime.',
  },
  looper: {
    title: 'Looper',
    eyebrow: 'Routing Engine',
    description: 'Multi-model execution endpoint and timeout defaults for looper strategies.',
  },
  clear_route_cache: {
    title: 'Route Cache Reset',
    eyebrow: 'Runtime',
    description: 'Controls whether the route cache is cleared on startup or reload.',
  },
  model_selection: {
    title: 'Model Selection',
    eyebrow: 'Routing Engine',
    description: 'ML-based model selection paths, artifacts, and custom-training knobs.',
  },
  api: {
    title: 'Batch Classification API',
    eyebrow: 'Additional API',
    description: 'Optional batch-classification runtime limits when present in the deployed config.',
  },
  bert_model: {
    title: 'Legacy Similarity BERT',
    eyebrow: 'Legacy',
    description: 'Legacy semantic-similarity model settings retained for non-template configs.',
  },
}
