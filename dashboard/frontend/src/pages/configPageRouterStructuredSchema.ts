import type { EditFormData } from '../components/EditModal'
import {
  EMBEDDING_MODELS_STRUCTURED_FIELDS,
  STORE_EMBEDDING_FIELD,
} from './configPageEmbeddingStructuredSchema'
import type { RouterSystemKey } from './configPageRouterSectionCatalog'
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
    logging: object('Logging', {
      level: select('Level', ['debug', 'info', 'warn', 'error']),
    }),
    development: object('Development', {
      drop_collection_on_startup: boolean('Drop Collection On Startup'),
      auto_create_collection: boolean('Auto Create Collection'),
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
  }),
  logging: object('Logging', {
    level: select('Level', ['debug', 'info', 'warn', 'error']),
  }),
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
    max_models: number('Max Models', { min: 1 }),
  }),
})

export const ROUTER_STRUCTURED_FIELDS: Partial<
  Record<RouterSystemKey, Record<string, RouterStructuredFieldDefinition>>
> = {
  router_core: {
    streamed_body: {
      label: 'Streamed Body',
      description: 'Limits for streamed ext-proc request bodies.',
      schema: object('Streamed Body', {
        enabled: boolean('Enabled'),
        max_bytes: number('Max Bytes', { min: 1 }),
        timeout_sec: number('Timeout Seconds', { min: 0 }),
      }),
    },
    skip_processing: {
      label: 'Skip Processing Header',
      description:
        'Allow callers to opt out of semantic-router processing with the supported header.',
      schema: object('Skip Processing Header', {
        enabled: boolean('Enabled'),
      }),
    },
  },
  learning: {
    adaptation: {
      label: 'Online Adaptation',
      description: 'How online evidence updates model selection.',
      schema: object('Online Adaptation', {
        enabled: boolean('Enabled'),
        candidate_set: select('Candidate Set', ['decision', 'tier', 'global']),
        strategy: text('Strategy'),
      }),
    },
    protection: {
      label: 'Model-switch Protection',
      description: 'Identity and stability controls that protect multi-turn continuity.',
      schema: object('Model-switch Protection', {
        enabled: boolean('Enabled'),
        scope: select('Scope', ['conversation', 'session']),
        identity: object('Identity', {
          headers: object('Headers', {
            session: text('Session Header'),
            conversation: text('Conversation Header'),
          }),
        }),
        tuning: object('Tuning', {
          idle_timeout_seconds: number('Idle Timeout Seconds', { min: 0 }),
          min_turns_before_switch: number('Minimum Turns Before Switch', { min: 0 }),
          switch_margin: number('Switch Margin', { min: 0 }),
          stability_weight: number('Stability Weight', { min: 0 }),
        }),
      }),
    },
    state_store: {
      label: 'Learning State Store',
      description: 'Shared state used to enforce protection across router replicas.',
      schema: object('Learning State Store', {
        backend: select('Backend', ['local', 'redis']),
        ttl_seconds: number('TTL Seconds', { min: 0 }),
        timeout_ms: number('Timeout Milliseconds', { min: 0 }),
        redis: object('Redis', {
          address: text('Address'),
          password: password('Password'),
          database: number('Database', { min: 0 }),
          key_prefix: text('Key Prefix'),
        }),
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
    embedding: STORE_EMBEDDING_FIELD,
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
  response_cache: {
    embedding: STORE_EMBEDDING_FIELD,
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
    embedding: STORE_EMBEDDING_FIELD,
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
  embedding_models: EMBEDDING_MODELS_STRUCTURED_FIELDS,
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
      description: 'Headers added to every Looper model call.',
      schema: stringMap('Header'),
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
    case 'json':
      return value
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
