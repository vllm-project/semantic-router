import {
  X_AUTHZ_USER_GROUPS,
  X_AUTHZ_USER_ID,
  X_CONVERSATION_ID,
  X_SESSION_ID,
} from '../generated/vsrHeaders'
import type { RouterSystemKey } from './configPageRouterSectionCatalog'
import type {
  APIConfig,
  AuthzConfig,
  CanonicalEmbeddingCatalogConfig,
  CanonicalSystemModels,
  LooperConfig,
  MemoryConfig,
  ModelSelectionConfig,
  ModalityDetectorConfig,
  ObservabilityConfig,
  PromptCompressionConfig,
  RateLimitConfig,
  ResponseAPIConfig,
  RouterLearningConfig,
  RouterCoreConfig,
  RouterReplayConfig,
  SemanticCacheConfig,
  VectorStoreConfig,
} from './configPageSupport'

export const DEFAULT_SECTIONS: Record<RouterSystemKey, unknown> = {
  router_core: {
    config_source: 'file',
    list_backend_models: false,
    clear_route_cache: true,
    model_selection: {
      enabled: true,
    },
  } satisfies RouterCoreConfig,
  learning: {
    enabled: false,
    adaptation: {
      enabled: true,
      candidate_set: 'decision',
      strategy: 'routing_sampling',
    },
    protection: {
      enabled: true,
      scope: 'conversation',
      identity: {
        headers: {
          session: X_SESSION_ID,
          conversation: X_CONVERSATION_ID,
        },
      },
      tuning: {},
    },
    state_store: {
      backend: 'local',
    },
  } satisfies RouterLearningConfig,
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
      user_id_header: X_AUTHZ_USER_ID,
      user_groups_header: X_AUTHZ_USER_GROUPS,
    },
    providers: [],
  } satisfies AuthzConfig,
  ratelimit: {
    fail_open: false,
    providers: [],
  } satisfies RateLimitConfig,
  management_api: {},
  startup_status: {},
  memory: {
    auto_store: false,
    milvus: {
      collection: 'agentic_memory',
    },
    default_retrieval_limit: 5,
    default_similarity_threshold: 0.7,
  } satisfies MemoryConfig,
  response_cache: {
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
    ingestion_workers: 2,
    ingestion_drain_timeout_seconds: 25,
    supported_formats: ['.txt', '.md', '.json', '.csv', '.html'],
    memory: {
      max_entries_per_store: 100000,
    },
  } satisfies VectorStoreConfig,
  external_models: [],
  admission: {},
  system_models: {
    decision_model: { deployment: 'primary' },
  } satisfies CanonicalSystemModels,
  embedding_models: {
    default: { deployment: 'vela-embedding' },
  } satisfies CanonicalEmbeddingCatalogConfig,
  prompt_compression: {
    enabled: false,
    profile: 'default',
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
        type: 'probabilistic',
        rate: 0.1,
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
  } satisfies ModelSelectionConfig,
  api: {
    batch_classification: {
      metrics: {},
    },
  } satisfies APIConfig,
}

export const SECTION_META: Record<
  RouterSystemKey,
  { title: string; eyebrow: string; description: string }
> = {
  router_core: {
    title: 'Router Core',
    eyebrow: 'Router',
    description:
      'Core router behavior, config source, startup cache handling, and model selection strategy.',
  },
  learning: {
    title: 'Router Learning',
    eyebrow: 'Router',
    description:
      'Online model-choice adaptation, conversation stability protection, and shared learning state.',
  },
  response_api: {
    title: 'Response API',
    eyebrow: 'Services',
    description:
      'Conversation chaining and persistence defaults for the OpenAI Responses API surface.',
  },
  router_replay: {
    title: 'Router Replay',
    eyebrow: 'Services',
    description:
      'Shared Replay storage and capture defaults. Each decision can override capture settings through its Replay plugin.',
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
  management_api: {
    title: 'Management API',
    eyebrow: 'Services',
    description: 'Management endpoint access, transport, and runtime controls.',
  },
  startup_status: {
    title: 'Startup Status',
    eyebrow: 'Services',
    description: 'Startup readiness and status-reporting behavior exposed by the router.',
  },
  memory: {
    title: 'Agentic Memory',
    eyebrow: 'Stores',
    description:
      'Cross-session memory extraction, storage, retrieval thresholds, and reflection policy.',
  },
  response_cache: {
    title: 'Response Cache',
    eyebrow: 'Stores',
    description: 'Similarity cache backend, retention policy, and embedding-backed cache behavior.',
  },
  vector_store: {
    title: 'Vector Store',
    eyebrow: 'Stores',
    description:
      'Document ingestion, vector backend selection, and file-backed knowledge-store defaults.',
  },
  external_models: {
    title: 'External Models',
    eyebrow: 'Model Catalog',
    description: 'Deprecated external LLM integrations, kept as the configuration declares them.',
  },
  admission: {
    title: 'Model Admission',
    eyebrow: 'Model Catalog',
    description: 'Named admission policies used to qualify models before routing.',
  },
  system_models: {
    title: 'Decision Model & Bindings',
    eyebrow: 'Model Catalog',
    description:
      'Choose the deployment that answers built-in signals and decision questions, and bind individual signals to other deployments.',
  },
  embedding_models: {
    title: 'Embedding Models',
    eyebrow: 'Model Catalog',
    description:
      'The deployment that embeds for match.recall; decisions never use it, and it starts only when a store recalls with it.',
  },
  prompt_compression: {
    title: 'Prompt Compression',
    eyebrow: 'Model Catalog',
    description: 'Optional signal-evaluation compression profiles plus hard input bounds.',
  },
  modality_detector: {
    title: 'Modality Detector',
    eyebrow: 'Model Catalog',
    description:
      'Prompt modality detection strategy, classifier settings, and keyword fallback behavior.',
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
    description: 'Turns on the selection algorithms that decisions configure.',
  },
  api: {
    title: 'API Service',
    eyebrow: 'Services',
    description: 'Batch-classification API limits and metrics settings when configured.',
  },
}
