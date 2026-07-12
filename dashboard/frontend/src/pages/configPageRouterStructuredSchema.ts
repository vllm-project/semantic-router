import type { EditFormData } from '../components/EditModal'
import type { RouterSystemKey } from './configPageRouterDefaultsSupport'
import {
  boolean,
  number,
  numberList,
  object,
  objectList,
  password,
  select,
  stringList,
  stringMap,
  text,
  type RouterStructuredFieldDefinition,
  type RouterStructuredSchema,
} from './configPageRouterStructuredSchemaPrimitives'

export type {
  RouterStructuredFieldDefinition,
  RouterStructuredKind,
  RouterStructuredSchema,
} from './configPageRouterStructuredSchemaPrimitives'

const milvusSchema = (label: string): RouterStructuredSchema =>
  object(label, {
    connection: object('Connection', {
      host: text('Host'),
      port: number('Port', { min: 1, max: 65535 }),
      database: text('Database'),
      timeout: number('Timeout'),
      auth: object('Authentication', {
        enabled: boolean('Enabled'),
        username: text('Username'),
        password: password('Password'),
      }),
      tls: object('TLS', {
        enabled: boolean('Enabled'),
        cert_file: text('Certificate File'),
        key_file: text('Key File'),
        ca_file: text('CA File'),
      }),
    }),
    collection: object('Collection', {
      name: text('Name'),
      description: text('Description'),
      vector_field: object('Vector Field', {
        name: text('Name'),
        dimension: number('Dimension', { min: 1 }),
        metric_type: select('Metric Type', ['COSINE', 'IP', 'L2']),
      }),
      index: object('Index', {
        type: text('Type'),
        params: object('Parameters', {
          M: number('M', { min: 1 }),
          efConstruction: number('efConstruction', { min: 1 }),
        }),
      }),
    }),
    search: object('Search', {
      params: object('Parameters', { ef: number('ef', { min: 1 }) }),
      topk: number('Top K', { min: 1 }),
      consistency_level: text('Consistency Level'),
    }),
    performance: object('Performance', {
      connection_pool: object('Connection Pool', {
        max_connections: number('Max Connections', { min: 1 }),
        max_idle_connections: number('Max Idle Connections', { min: 0 }),
        acquire_timeout: number('Acquire Timeout', { min: 0 }),
      }),
      batch: object('Batch', {
        insert_batch_size: number('Insert Batch Size', { min: 1 }),
        timeout: number('Timeout', { min: 0 }),
      }),
    }),
    data_management: object('Data Management', {
      ttl: object('TTL', {
        enabled: boolean('Enabled'),
        timestamp_field: text('Timestamp Field'),
        cleanup_interval: number('Cleanup Interval', { min: 0 }),
      }),
      compaction: object('Compaction', {
        enabled: boolean('Enabled'),
        interval: number('Interval', { min: 0 }),
      }),
    }),
    logging: object('Logging', {
      level: select('Level', ['debug', 'info', 'warn', 'error']),
      enable_query_log: boolean('Query Log'),
      enable_metrics: boolean('Metrics'),
    }),
    development: object('Development', {
      drop_collection_on_startup: boolean('Drop Collection On Startup'),
      auto_create_collection: boolean('Auto Create Collection'),
      verbose_errors: boolean('Verbose Errors'),
    }),
  })

const redisSchema = object('Redis Backend', {
  connection: object('Connection', {
    host: text('Host'),
    port: number('Port', { min: 1, max: 65535 }),
    database: number('Database', { min: 0 }),
    password: password('Password'),
    timeout: number('Timeout', { min: 0 }),
    tls: object('TLS', {
      enabled: boolean('Enabled'),
      cert_file: text('Certificate File'),
      key_file: text('Key File'),
      ca_file: text('CA File'),
    }),
  }),
  index: object('Index', {
    name: text('Name'),
    prefix: text('Prefix'),
    vector_field: object('Vector Field', {
      name: text('Name'),
      dimension: number('Dimension', { min: 1 }),
      metric_type: select('Metric Type', ['COSINE', 'IP', 'L2']),
    }),
    index_type: text('Index Type'),
    params: object('Parameters', {
      M: number('M', { min: 1 }),
      efConstruction: number('efConstruction', { min: 1 }),
    }),
  }),
  search: object('Search', { topk: number('Top K', { min: 1 }) }),
  development: object('Development', {
    drop_index_on_startup: boolean('Drop Index On Startup'),
    auto_create_index: boolean('Auto Create Index'),
    verbose_errors: boolean('Verbose Errors'),
  }),
  logging: object('Logging', {
    level: select('Level', ['debug', 'info', 'warn', 'error']),
    enable_query_log: boolean('Query Log'),
    enable_metrics: boolean('Metrics'),
  }),
})

const modelModule = (label: string, detector = false): RouterStructuredSchema =>
  object(label, {
    model_ref: text('Model Ref'),
    model_id: text('Model ID'),
    threshold: number('Threshold', { min: 0, max: 1, step: 0.01 }),
    use_cpu: boolean('Use CPU'),
    use_mmbert_32k: boolean('Use mmBERT 32K'),
    ...(detector
      ? {
          min_span_length: number('Min Span Length', { min: 1 }),
          min_span_confidence: number('Min Span Confidence', { min: 0, max: 1, step: 0.01 }),
          context_window_size: number('Context Window Size', { min: 1 }),
          enable_nli_filtering: boolean('Enable NLI Filtering'),
          nli_entailment_threshold: number('NLI Entailment Threshold', {
            min: 0,
            max: 1,
            step: 0.01,
          }),
        }
      : {}),
  })

const prototypeScoringSchema = object('Prototype Scoring', {
  enabled: boolean('Enabled'),
  cluster_similarity_threshold: number('Cluster Similarity Threshold', {
    min: 0,
    max: 1,
    step: 0.01,
  }),
  max_prototypes: number('Max Prototypes', { min: 1 }),
  best_weight: number('Best Weight', { min: 0, max: 1, step: 0.01 }),
  top_m: number('Top M', { min: 1 }),
  margin_threshold: number('Margin Threshold', { min: 0, step: 0.01 }),
})

const classifierModelSchema = (
  label: string,
  mappingField?: 'category_mapping_path' | 'pii_mapping_path',
): RouterStructuredSchema =>
  object(label, {
    model_ref: text('Model Ref'),
    model_id: text('Model ID'),
    threshold: number('Threshold', { min: 0, max: 1, step: 0.01 }),
    use_cpu: boolean('Use CPU'),
    use_modernbert: boolean('Use ModernBERT'),
    use_mmbert_32k: boolean('Use mmBERT 32K'),
    ...(mappingField ? { [mappingField]: text('Mapping Path') } : {}),
    ...(mappingField === 'category_mapping_path'
      ? { fallback_category: text('Fallback Category') }
      : {}),
  })

const tracingSchema = object('Tracing', {
  enabled: boolean('Enabled'),
  provider: text('Provider'),
  exporter: object('Exporter', {
    type: text('Type'),
    endpoint: text('Endpoint'),
    insecure: boolean('Insecure'),
  }),
  sampling: object('Sampling', {
    type: text('Type'),
    rate: number('Rate', { min: 0, max: 1, step: 0.01 }),
  }),
  resource: object('Resource', {
    service_name: text('Service Name'),
    service_version: text('Service Version'),
    deployment_environment: text('Deployment Environment'),
  }),
})

const metricsSchema = object('Metrics', {
  enabled: boolean('Enabled'),
  windowed_metrics: object('Windowed Metrics', {
    enabled: boolean('Enabled'),
    time_windows: stringList('Time Windows', '5m'),
    update_interval: text('Update Interval'),
    model_metrics: boolean('Model Metrics'),
    queue_depth_estimation: boolean('Queue Depth Estimation'),
    max_models: number('Max Models', { min: 1 }),
  }),
})

export const ROUTER_STRUCTURED_FIELDS: Partial<
  Record<RouterSystemKey, Record<string, RouterStructuredFieldDefinition>>
> = {
  router_core: {
    auto_model_names: {
      label: 'Auto Model Aliases',
      description: 'Accepted aliases for automatic model routing.',
      schema: stringList('Alias', 'vllm-sr/auto'),
    },
    streamed_body: {
      label: 'Streamed Body',
      description: 'Limits for streamed ext-proc request bodies.',
      schema: object('Streamed Body', {
        enabled: boolean('Enabled'),
        max_bytes: number('Max Bytes', { min: 1 }),
        timeout_sec: number('Timeout Seconds', { min: 0 }),
      }),
    },
  },
  authz: {
    identity: {
      label: 'Identity Headers',
      description: 'Headers used to resolve user and group identity.',
      schema: object('Identity Headers', {
        user_id_header: text('User ID Header'),
        user_groups_header: text('User Groups Header'),
      }),
    },
    providers: {
      label: 'Authorization Providers',
      description: 'Ordered authorization provider definitions.',
      schema: objectList(
        'Provider',
        {
          type: select('Type', ['header'], true),
          headers: stringMap('Headers'),
        },
        'type',
      ),
    },
  },
  ratelimit: {
    providers: {
      label: 'Rate Limit Providers',
      description: 'Provider connections and their typed rate-limit rules.',
      schema: objectList(
        'Provider',
        {
          type: select('Type', ['redis'], true),
          address: text('Address'),
          domain: text('Domain'),
          rules: objectList(
            'Rule',
            {
              name: text('Name', { required: true }),
              match: object('Match', {
                user: text('User'),
                group: text('Group'),
                model: text('Model'),
              }),
              requests_per_unit: number('Requests Per Unit', { min: 0 }),
              tokens_per_unit: number('Tokens Per Unit', { min: 0 }),
              unit: select('Unit', ['second', 'minute', 'hour', 'day'], true),
            },
            'name',
          ),
        },
        'type',
      ),
    },
  },
  memory: {
    milvus: {
      label: 'Milvus Memory Store',
      description: 'Connection and collection settings for agentic memory.',
      schema: object('Milvus Memory Store', {
        address: text('Address'),
        collection: text('Collection'),
        dimension: number('Dimension', { min: 1 }),
        num_partitions: number('Partitions', { min: 1 }),
      }),
    },
    quality_scoring: {
      label: 'Quality Scoring',
      description: 'Memory strength, pruning, and capacity policy.',
      schema: object('Quality Scoring', {
        initial_strength_days: number('Initial Strength Days', { min: 0 }),
        prune_threshold: number('Prune Threshold', { min: 0, max: 1, step: 0.01 }),
        max_memories_per_user: number('Max Memories Per User', { min: 1 }),
      }),
    },
    reflection: {
      label: 'Reflection',
      description: 'Reflection algorithm and memory injection limits.',
      schema: object('Reflection', {
        enabled: boolean('Enabled'),
        algorithm: text('Algorithm'),
        max_inject_tokens: number('Max Inject Tokens', { min: 0 }),
        recency_decay_days: number('Recency Decay Days', { min: 0 }),
        dedup_threshold: number('Dedup Threshold', { min: 0, max: 1, step: 0.01 }),
        block_patterns: stringList('Block Patterns', 'secret'),
      }),
    },
  },
  semantic_cache: {
    redis: {
      label: 'Redis Backend',
      description: 'Typed Redis connection, index, search, and lifecycle settings.',
      schema: redisSchema,
    },
    milvus: {
      label: 'Milvus Backend',
      description: 'Typed Milvus collection, search, and lifecycle settings.',
      schema: milvusSchema('Milvus Backend'),
    },
  },
  vector_store: {
    supported_formats: {
      label: 'Supported Formats',
      description: 'File extensions accepted by ingestion.',
      schema: stringList('File Extension', '.md'),
    },
    memory: {
      label: 'Memory Backend',
      description: 'Capacity for the in-memory vector backend.',
      schema: object('Memory Backend', {
        max_entries_per_store: number('Max Entries Per Store', { min: 1 }),
      }),
    },
    milvus: {
      label: 'Milvus Backend',
      description: 'Typed Milvus collection, search, and lifecycle settings.',
      schema: milvusSchema('Milvus Backend'),
    },
    llama_stack: {
      label: 'Llama Stack Backend',
      description: 'Llama Stack endpoint and search behavior.',
      schema: object('Llama Stack Backend', {
        endpoint: text('Endpoint', { required: true }),
        auth_token: password('Auth Token'),
        embedding_model: text('Embedding Model'),
        request_timeout_seconds: number('Request Timeout Seconds', { min: 0 }),
        search_type: text('Search Type'),
      }),
    },
  },
  tools: {
    advanced_filtering: {
      label: 'Advanced Filtering',
      description: 'Typed retrieval, weighting, category, allow/block, and history controls.',
      schema: object('Advanced Filtering', {
        enabled: boolean('Enabled'),
        retrieval_strategy: select('Retrieval Strategy', ['weighted', 'hybrid_history']),
        candidate_pool_size: number('Candidate Pool Size', { min: 0 }),
        min_lexical_overlap: number('Min Lexical Overlap', { min: 0 }),
        min_combined_score: number('Min Combined Score', { min: 0, max: 1, step: 0.01 }),
        weights: object('Weights', {
          embed: number('Embedding', { min: 0, max: 1, step: 0.01 }),
          lexical: number('Lexical', { min: 0, max: 1, step: 0.01 }),
          tag: number('Tag', { min: 0, max: 1, step: 0.01 }),
          name: number('Name', { min: 0, max: 1, step: 0.01 }),
          category: number('Category', { min: 0, max: 1, step: 0.01 }),
        }),
        use_category_filter: boolean('Use Category Filter'),
        category_confidence_threshold: number('Category Confidence Threshold', {
          min: 0,
          max: 1,
          step: 0.01,
        }),
        allow_tools: stringList('Allowed Tool', 'docs.search'),
        block_tools: stringList('Blocked Tool', 'admin.delete'),
        hybrid_history: object('Hybrid History', {
          history_horizon: number('History Horizon', { min: 0 }),
          min_history_steps: number('Min History Steps', { min: 0 }),
          history_confidence_threshold: number('History Confidence Threshold', {
            min: 0,
            max: 1,
            step: 0.01,
          }),
          weight_semantic: number('Semantic Weight', { min: 0, max: 1, step: 0.01 }),
          weight_history_transition: number('History Transition Weight', {
            min: 0,
            max: 1,
            step: 0.01,
          }),
          weight_decision_prior: number('Decision Prior Weight', {
            min: 0,
            max: 1,
            step: 0.01,
          }),
          repetition_penalty_strength: number('Repetition Penalty Strength', {
            min: 0,
            max: 1,
            step: 0.01,
          }),
        }),
      }),
    },
  },
  classifier: {
    domain: {
      label: 'Domain Module',
      description: 'Domain classifier binding, threshold, mapping, and fallback.',
      schema: classifierModelSchema('Domain Module', 'category_mapping_path'),
    },
    pii: {
      label: 'PII Module',
      description: 'PII classifier binding, threshold, and type mapping.',
      schema: classifierModelSchema('PII Module', 'pii_mapping_path'),
    },
    mcp: {
      label: 'MCP Module',
      description: 'MCP classifier transport, process, environment, and timeout settings.',
      schema: object('MCP Module', {
        enabled: boolean('Enabled'),
        transport_type: select('Transport Type', ['stdio', 'sse', 'streamable-http']),
        command: text('Command'),
        args: stringList('Argument', '-m'),
        env: stringMap('Environment Variable'),
        url: text('URL'),
        tool_name: text('Tool Name'),
        threshold: number('Threshold', { min: 0, max: 1, step: 0.01 }),
        timeout_seconds: number('Timeout Seconds', { min: 0 }),
      }),
    },
    preference: {
      label: 'Preference Module',
      description: 'Contrastive preference embedding and prototype scoring settings.',
      schema: object('Preference Module', {
        use_contrastive: boolean('Use Contrastive'),
        embedding_model: text('Embedding Model'),
        model_id: text('Model ID'),
        threshold: number('Threshold', { min: 0, max: 1, step: 0.01 }),
        use_cpu: boolean('Use CPU'),
        prototype_scoring: prototypeScoringSchema,
      }),
    },
  },
  hallucination_mitigation: {
    fact_check: {
      label: 'Fact Check Module',
      description: 'Fact-check model binding and threshold.',
      schema: modelModule('Fact Check Module'),
    },
    detector: {
      label: 'Detector Module',
      description: 'Hallucination detector model and span/NLI thresholds.',
      schema: modelModule('Detector Module', true),
    },
    explainer: {
      label: 'Explainer Module',
      description: 'Hallucination explainer model binding and threshold.',
      schema: modelModule('Explainer Module'),
    },
  },
  external_models: {
    items: {
      label: 'External Models',
      description: 'External provider models used by router-owned auxiliary workflows.',
      schema: objectList(
        'External Model',
        {
          llm_provider: text('Provider', { required: true }),
          model_role: text('Model Role', { required: true }),
          llm_endpoint: object('Endpoint', {
            address: text('Address'),
            port: number('Port', { min: 1, max: 65535 }),
            protocol: select('Protocol', ['http', 'https']),
            name: text('Name'),
            use_chat_template: boolean('Use Chat Template'),
            prompt_template: text('Prompt Template'),
          }),
          llm_model_name: text('Model Name'),
          llm_timeout_seconds: number('Timeout Seconds', { min: 0 }),
          parser_type: text('Parser Type'),
          threshold: number('Threshold', { min: 0, max: 1, step: 0.01 }),
          access_key: password('Access Key'),
          max_tokens: number('Max Tokens', { min: 1 }),
          temperature: number('Temperature', { min: 0 }),
        },
        'llm_model_name',
      ),
    },
  },
  embedding_models: {
    embedding_config: {
      label: 'Embedding Optimization',
      description: 'Embedding model, layer, dimension, and soft-matching controls.',
      schema: object('Embedding Optimization', {
        model_type: text('Model Type'),
        preload_embeddings: boolean('Preload Embeddings'),
        target_dimension: number('Target Dimension', { min: 1 }),
        target_layer: number('Target Layer', { min: 0 }),
        enable_soft_matching: boolean('Enable Soft Matching'),
        top_k: number('Top K', { min: 1 }),
        min_score_threshold: number('Min Score Threshold', { min: 0, max: 1, step: 0.01 }),
      }),
    },
  },
  prompt_compression: {
    skip_signals: {
      label: 'Skip Signals',
      description: 'Signals that bypass prompt compression.',
      schema: stringList('Signal', 'jailbreak'),
    },
  },
  modality_detector: {
    prompt_prefixes: {
      label: 'Prompt Prefixes',
      description: 'Prefixes that indicate a modality-specific request.',
      schema: stringList('Prompt Prefix', 'generate an image of '),
    },
    classifier: {
      label: 'Classifier',
      description: 'Modality classifier model and device.',
      schema: object('Classifier', {
        model_path: text('Model Path'),
        use_cpu: boolean('Use CPU'),
      }),
    },
    keywords: {
      label: 'Keywords',
      description: 'Keywords used by keyword or hybrid detection.',
      schema: stringList('Keyword', 'image'),
    },
    both_keywords: {
      label: 'Required Keyword Pairs',
      description: 'Keywords that must co-occur for a match.',
      schema: stringList('Keyword', 'draw'),
    },
  },
  observability: {
    metrics: {
      label: 'Metrics',
      description: 'Metrics and windowed aggregation settings.',
      schema: metricsSchema,
    },
    tracing: {
      label: 'Tracing',
      description: 'Tracing exporter, sampling, and resource metadata.',
      schema: tracingSchema,
    },
  },
  looper: {
    headers: {
      label: 'Headers',
      description: 'Headers sent to the Looper endpoint.',
      schema: stringMap('Header'),
    },
  },
  model_selection: {
    knn: {
      label: 'KNN',
      description: 'K-nearest-neighbor model selection settings.',
      schema: object('KNN', {
        k: number('K', { min: 1 }),
        pretrained_path: text('Pretrained Path'),
      }),
    },
    kmeans: {
      label: 'KMeans',
      description: 'KMeans model selection settings.',
      schema: object('KMeans', {
        num_clusters: number('Clusters', { min: 1 }),
        efficiency_weight: number('Efficiency Weight', { min: 0 }),
        pretrained_path: text('Pretrained Path'),
      }),
    },
    svm: {
      label: 'SVM',
      description: 'SVM model selection settings.',
      schema: object('SVM', {
        kernel: text('Kernel'),
        gamma: number('Gamma', { min: 0 }),
        pretrained_path: text('Pretrained Path'),
      }),
    },
    router_dc: {
      label: 'RouterDC',
      description: 'RouterDC similarity and contrastive settings.',
      schema: object('RouterDC', {
        temperature: number('Temperature', { min: 0 }),
        dimension_size: number('Dimension Size', { min: 1 }),
        min_similarity: number('Min Similarity', { min: 0, max: 1, step: 0.01 }),
        use_query_contrastive: boolean('Query Contrastive'),
        use_model_contrastive: boolean('Model Contrastive'),
        require_descriptions: boolean('Require Descriptions'),
        use_capabilities: boolean('Use Capabilities'),
      }),
    },
    automix: {
      label: 'AutoMix',
      description: 'AutoMix verification, escalation, and cost controls.',
      schema: object('AutoMix', {
        verification_threshold: number('Verification Threshold', { min: 0, max: 1, step: 0.01 }),
        max_escalations: number('Max Escalations', { min: 0 }),
        cost_aware_routing: boolean('Cost-Aware Routing'),
        cost_quality_tradeoff: number('Cost Quality Tradeoff', { min: 0 }),
        discount_factor: number('Discount Factor', { min: 0 }),
        use_logprob_verification: boolean('Use Logprob Verification'),
      }),
    },
    hybrid: {
      label: 'Hybrid',
      description: 'Hybrid selector weights and normalization.',
      schema: object('Hybrid', {
        experience_weight: number('Experience Weight', { min: 0 }),
        router_dc_weight: number('RouterDC Weight', { min: 0 }),
        automix_weight: number('AutoMix Weight', { min: 0 }),
        cost_weight: number('Cost Weight', { min: 0 }),
        quality_gap_threshold: number('Quality Gap Threshold', { min: 0 }),
        normalize_scores: boolean('Normalize Scores'),
      }),
    },
  },
  api: {
    batch_classification: {
      label: 'Batch Classification',
      description: 'Batch concurrency and metrics collection settings.',
      schema: object('Batch Classification', {
        max_batch_size: number('Max Batch Size', { min: 1 }),
        concurrency_threshold: number('Concurrency Threshold', { min: 1 }),
        max_concurrency: number('Max Concurrency', { min: 1 }),
        metrics: object('Metrics', {
          enabled: boolean('Enabled'),
          detailed_goroutine_tracking: boolean('Detailed Goroutine Tracking'),
          high_resolution_timing: boolean('High Resolution Timing'),
          sample_rate: number('Sample Rate', { min: 0, max: 1, step: 0.01 }),
          batch_size_ranges: objectList(
            'Batch Size Range',
            {
              min: number('Minimum', { min: 0 }),
              max: number('Maximum', { min: 0 }),
              label: text('Label', { required: true }),
            },
            'label',
          ),
          duration_buckets: numberList('Duration Buckets'),
          size_buckets: numberList('Size Buckets'),
        }),
      }),
    },
  },
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

export function createRouterStructuredValue(schema: RouterStructuredSchema): unknown {
  if (schema.defaultValue !== undefined) return structuredClone(schema.defaultValue)
  switch (schema.kind) {
    case 'boolean':
      return false
    case 'number':
      return 0
    case 'string-list':
    case 'number-list':
    case 'object-list':
      return []
    case 'string-map':
    case 'object':
      return {}
    default:
      return ''
  }
}

export function normalizeRouterStructuredValue(
  schema: RouterStructuredSchema,
  value: unknown,
  path = schema.label,
): unknown {
  if (value === undefined || value === null) {
    if (schema.required) throw new Error(`${path} is required.`)
    return undefined
  }

  switch (schema.kind) {
    case 'string':
    case 'password':
    case 'select': {
      if (typeof value !== 'string') throw new Error(`${path} must be text.`)
      if (schema.required && !value.trim()) throw new Error(`${path} is required.`)
      return value
    }
    case 'number': {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        throw new Error(`${path} must be a finite number.`)
      }
      if (schema.min !== undefined && value < schema.min)
        throw new Error(`${path} must be at least ${schema.min}.`)
      if (schema.max !== undefined && value > schema.max)
        throw new Error(`${path} must be at most ${schema.max}.`)
      return value
    }
    case 'boolean':
      if (typeof value !== 'boolean') throw new Error(`${path} must be true or false.`)
      return value
    case 'string-list': {
      if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
        throw new Error(`${path} must be a list of text values.`)
      }
      if (value.some((item) => !(item as string).trim()))
        throw new Error(`${path} cannot contain empty values.`)
      const normalized = (value as string[]).map((item) => item.trim())
      if (new Set(normalized.map((item) => item.toLocaleLowerCase())).size !== normalized.length) {
        throw new Error(`${path} must contain unique values.`)
      }
      return normalized
    }
    case 'number-list':
      if (
        !Array.isArray(value) ||
        value.some((item) => typeof item !== 'number' || !Number.isFinite(item))
      ) {
        throw new Error(`${path} must be a list of finite numbers.`)
      }
      return [...value]
    case 'string-map': {
      const record = asRecord(value)
      if (
        !record ||
        Object.entries(record).some(([key, item]) => !key.trim() || typeof item !== 'string')
      ) {
        throw new Error(`${path} must contain non-empty keys and text values.`)
      }
      return { ...record }
    }
    case 'object': {
      const record = asRecord(value)
      if (!record) throw new Error(`${path} must be an object.`)
      const normalized: Record<string, unknown> = { ...record }
      for (const [key, fieldSchema] of Object.entries(schema.fields ?? {})) {
        const fieldPath = `${path}.${key}`
        const nextValue = normalizeRouterStructuredValue(fieldSchema, record[key], fieldPath)
        if (nextValue === undefined) delete normalized[key]
        else normalized[key] = nextValue
      }
      return normalized
    }
    case 'object-list': {
      if (!Array.isArray(value)) throw new Error(`${path} must be a list.`)
      if (!schema.item) return [...value]
      return value.map((item, index) =>
        normalizeRouterStructuredValue(
          schema.item as RouterStructuredSchema,
          item,
          `${path}[${index + 1}]`,
        ),
      )
    }
  }
}

export function normalizeRouterStructuredFields(
  key: RouterSystemKey,
  data: EditFormData,
): EditFormData {
  const definitions = ROUTER_STRUCTURED_FIELDS[key]
  if (!definitions) return { ...data }
  const normalized: EditFormData = { ...data }
  for (const [name, definition] of Object.entries(definitions)) {
    const nextValue = normalizeRouterStructuredValue(
      definition.schema,
      data[name],
      definition.label,
    )
    if (nextValue === undefined) delete normalized[name]
    else normalized[name] = nextValue
  }
  return normalized
}
