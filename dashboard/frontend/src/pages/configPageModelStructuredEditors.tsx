import { KeyValueEditor } from '../components/KeyValueEditor'
import { ObjectListEditor, type ObjectEditorField } from '../components/ObjectListEditor'
import { StringListEditor } from '../components/StringListEditor'
import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import type {
  BackendRefEntry,
  LoRAAdapter,
  EvaluationRecordConfig,
  ModelPricing,
  ProviderReliability,
} from './configPageSupport'
import { normalizeModelBackendRefs, normalizeModelReliability } from './configPageModelFormSupport'

interface StructuredModelFieldProps {
  value: unknown
  onChange?: (value: unknown) => void
  disabled?: boolean
  readOnly?: boolean
  maskSensitive?: boolean
}

const backendRefFields: ObjectEditorField<BackendRefEntry>[] = [
  { key: 'name', label: 'Reference name', placeholder: 'local-primary' },
  { key: 'endpoint', label: 'Endpoint', placeholder: '127.0.0.1:8000', fullWidth: true },
  {
    key: 'base_url',
    label: 'Base URL',
    placeholder: 'https://api.example.com/v1',
    fullWidth: true,
  },
  { key: 'protocol', label: 'Protocol', type: 'select', options: ['http', 'https'] },
  { key: 'weight', label: 'Traffic weight', type: 'number', min: 0, step: 1, placeholder: '1' },
  { key: 'provider', label: 'Provider', placeholder: 'openai' },
  { key: 'api_version', label: 'API version', placeholder: '2025-03-01' },
  { key: 'chat_path', label: 'Chat path', placeholder: '/chat/completions', fullWidth: true },
  {
    key: 'api_key_env',
    label: 'API key environment variable',
    placeholder: 'OPENAI_API_KEY',
    fullWidth: true,
  },
  {
    key: 'api_key',
    label: 'Inline API key',
    type: 'password',
    placeholder: 'Stored in configuration',
    fullWidth: true,
  },
  { key: 'auth_header', label: 'Authentication header', placeholder: 'Authorization' },
  {
    key: 'auth_prefix',
    label: 'Authentication prefix',
    placeholder: 'Bearer',
    preserveEmpty: true,
    helpText: 'An explicit empty value disables the provider catalog default prefix.',
  },
  {
    key: 'extra_headers',
    label: 'Extra headers',
    type: 'key-value',
    fullWidth: true,
    emptyValueLabel: 'No extra headers configured.',
  },
]

interface EditableEvaluationRecord {
  model?: string
  benchmark?: string
  benchmark_profile?: string
  reasoning_effort?: string
  metrics?: Record<string, string>
  source?: string
  measured_at?: string
  metadata?: Record<string, string>
}

const evaluationFields: ObjectEditorField<EditableEvaluationRecord>[] = [
  {
    key: 'model',
    label: 'Model Card',
    placeholder: 'organization/model-id',
    required: true,
    fullWidth: true,
  },
  {
    key: 'benchmark',
    label: 'Benchmark',
    placeholder: 'organization/benchmark@1.0.0',
    required: true,
    fullWidth: true,
  },
  {
    key: 'benchmark_profile',
    label: 'Benchmark profile',
    placeholder: 'published-standard',
  },
  {
    key: 'reasoning_effort',
    label: 'Reasoning effort',
    placeholder: 'medium',
  },
  {
    key: 'metrics',
    label: 'Metrics',
    type: 'key-value',
    required: true,
    fullWidth: true,
    emptyValueLabel: 'Add at least one numeric metric.',
    keyLabel: 'Metric',
    keyPlaceholder: 'pass_at_1',
    valueLabel: 'Numeric value',
    valuePlaceholder: '0.72',
  },
  { key: 'source', label: 'Source URL', placeholder: 'https://evals.example/runs/42' },
  { key: 'measured_at', label: 'Measured at', placeholder: '2026-09-01' },
  {
    key: 'metadata',
    label: 'Subject metadata',
    type: 'key-value',
    fullWidth: true,
    emptyValueLabel: 'No optional subject metadata.',
    keyLabel: 'Field',
    keyPlaceholder: 'runtime',
    valueLabel: 'Value',
    valuePlaceholder: 'vllm',
  },
]

const loraFields: ObjectEditorField<LoRAAdapter>[] = [
  { key: 'name', label: 'Adapter name', placeholder: 'computer-science-expert', required: true },
  {
    key: 'description',
    label: 'Description',
    placeholder: 'What this adapter specializes in',
    fullWidth: true,
  },
]

const pricingFields: ObjectEditorField<ModelPricing>[] = [
  { key: 'currency', label: 'Currency', placeholder: 'USD' },
  {
    key: 'prompt_per_1m',
    label: 'Prompt / 1M tokens',
    type: 'number',
    min: 0,
    step: 0.0001,
    placeholder: '0.50',
  },
  {
    key: 'cached_input_per_1m',
    label: 'Cached input / 1M',
    type: 'number',
    min: 0,
    step: 0.0001,
    placeholder: '0.05',
  },
  {
    key: 'cache_write_per_1m',
    label: 'Cache write / 1M',
    type: 'number',
    min: 0,
    step: 0.0001,
    placeholder: '0.625',
  },
  {
    key: 'completion_per_1m',
    label: 'Completion / 1M',
    type: 'number',
    min: 0,
    step: 0.0001,
    placeholder: '1.50',
  },
]

const reliabilityFields: ObjectEditorField<ProviderReliability>[] = [
  {
    key: 'lb_policy',
    label: 'Load balancing',
    type: 'select',
    options: ['ROUND_ROBIN', 'LEAST_REQUEST', 'RING_HASH', 'MAGLEV'],
  },
  {
    key: 'retry_count',
    label: 'Max retries',
    type: 'number',
    min: 0,
    max: 5,
    step: 1,
    placeholder: '0',
  },
  {
    key: 'retry_on',
    label: 'Retry conditions',
    placeholder: '5xx,reset,connect-failure',
    fullWidth: true,
  },
  {
    key: 'health_check_path',
    label: 'Health check path',
    placeholder: '/health',
  },
  {
    key: 'health_check_interval',
    label: 'Health check interval',
    placeholder: '10s',
  },
  {
    key: 'health_check_timeout',
    label: 'Health check timeout',
    placeholder: '2s',
  },
  {
    key: 'consecutive_5xx',
    label: 'Errors before ejection',
    type: 'number',
    min: 0,
    step: 1,
    placeholder: '5',
  },
  {
    key: 'base_ejection_time',
    label: 'Base ejection time',
    placeholder: '30s',
  },
  {
    key: 'max_ejection_percent',
    label: 'Max ejection percent',
    type: 'number',
    min: 0,
    max: 100,
    step: 1,
    placeholder: '50',
  },
  {
    key: 'request_timeout',
    label: 'Request timeout',
    placeholder: '300s',
  },
  {
    key: 'stream_idle_timeout',
    label: 'Stream idle timeout',
    placeholder: '60s',
  },
  {
    key: 'connect_timeout',
    label: 'Connect timeout',
    placeholder: '10s',
  },
]

function backendRefLabel(item: BackendRefEntry, index: number): string {
  return item.name?.trim() || item.provider?.trim() || `Backend ${index + 1}`
}

function backendRefDescription(item: BackendRefEntry): string | undefined {
  return item.endpoint?.trim() || item.base_url?.trim() || 'Target required'
}

function validateBackendRef(item: BackendRefEntry): string[] {
  const errors: string[] = []
  if (!item.endpoint?.trim() && !item.base_url?.trim()) {
    errors.push('Provide an endpoint or base URL.')
  }
  if (item.protocol && item.protocol !== 'http' && item.protocol !== 'https') {
    errors.push('Protocol must be HTTP or HTTPS.')
  }
  if (item.weight !== undefined && (!Number.isFinite(item.weight) || item.weight < 0)) {
    errors.push('Traffic weight must be zero or greater.')
  }
  return errors
}

export function ModelTagsEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredModelFieldProps) {
  const tagValues = Array.isArray(value)
    ? value.filter((tag): tag is string => typeof tag === 'string')
    : normalizeStringList(value)

  return (
    <StringListEditor
      value={tagValues}
      onChange={(nextValue) => onChange?.(nextValue)}
      addLabel="Add tag"
      emptyLabel="No tags configured. Add tags for filtering and policy targeting."
      itemLabel="Tag"
      placeholder="e.g. premium"
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ModelCapabilitiesEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredModelFieldProps) {
  const capabilities = Array.isArray(value)
    ? value.filter((capability): capability is string => typeof capability === 'string')
    : normalizeStringList(value)

  return (
    <StringListEditor
      value={capabilities}
      onChange={(nextValue) => onChange?.(nextValue)}
      addLabel="Add capability"
      emptyLabel="No routing capabilities configured."
      itemLabel="Capability"
      placeholder="e.g. tool-use"
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ModelLorasEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredModelFieldProps) {
  const loras = Array.isArray(value)
    ? value.filter(
        (entry): entry is LoRAAdapter =>
          Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry),
      )
    : []

  return (
    <ObjectListEditor
      value={loras}
      onChange={(nextValue) => onChange?.(nextValue)}
      fields={loraFields}
      createItem={() => ({ name: '', description: '' })}
      addLabel="Add LoRA adapter"
      emptyLabel="No LoRA adapters configured."
      itemLabel={(item, index) => item.name?.trim() || `LoRA adapter ${index + 1}`}
      itemDescription={(item) => item.description?.trim()}
      validateItem={(item) => (item.name?.trim() ? [] : ['Adapter name is required.'])}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ModelExternalIdsEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredModelFieldProps) {
  const externalIds =
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(
          Object.entries(value).filter(
            (entry): entry is [string, string] => typeof entry[1] === 'string',
          ),
        )
      : {}

  return (
    <KeyValueEditor
      value={externalIds}
      onChange={(nextValue) => onChange?.(nextValue)}
      addLabel="Add provider ID"
      emptyLabel="No external provider IDs configured."
      keyLabel="Provider"
      keyPlaceholder="openai"
      valueLabel="Model ID"
      valuePlaceholder="gpt-4.1"
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ModelPricingEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredModelFieldProps) {
  const pricing =
    value && typeof value === 'object' && !Array.isArray(value) ? (value as ModelPricing) : {}

  return (
    <ObjectListEditor
      value={[pricing]}
      onChange={(nextValue) => onChange?.(nextValue[0] || {})}
      fields={pricingFields}
      createItem={() => ({ currency: 'USD' })}
      itemLabel={() => 'Token pricing'}
      itemDescription={(item) => `${item.currency || 'USD'} per one million tokens`}
      minItems={1}
      maxItems={1}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ModelReliabilityEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredModelFieldProps) {
  const reliability: ProviderReliability = normalizeModelReliability(value) || {}

  return (
    <ObjectListEditor<ProviderReliability>
      value={[reliability]}
      onChange={(nextValue) => onChange?.(nextValue[0] || {})}
      fields={reliabilityFields}
      createItem={() => ({})}
      itemLabel={() => 'Delivery policy'}
      itemDescription={(item) =>
        item.retry_count ? `${item.retry_count} retries` : 'Use platform defaults'
      }
      minItems={1}
      maxItems={1}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function ModelBackendRefsEditor({
  value,
  onChange,
  disabled,
  readOnly,
  maskSensitive,
}: StructuredModelFieldProps) {
  const backendRefs = normalizeModelBackendRefs(value)
  const visibleBackendRefs =
    maskSensitive && readOnly
      ? backendRefs.map((backendRef) => ({
          ...backendRef,
          endpoint: backendRef.endpoint ? '••••••••' : undefined,
          base_url: backendRef.base_url ? '••••••••' : undefined,
        }))
      : backendRefs

  return (
    <ObjectListEditor
      value={visibleBackendRefs}
      onChange={(nextValue) => onChange?.(nextValue)}
      fields={backendRefFields}
      createItem={(index) => ({
        name: `endpoint-${index + 1}`,
        endpoint: 'localhost:8000',
        protocol: 'http' as const,
        weight: 1,
        provider: 'vllm',
      })}
      addLabel="Add backend"
      emptyLabel="No provider backends configured."
      itemLabel={backendRefLabel}
      itemDescription={backendRefDescription}
      validateItem={validateBackendRef}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}

export function EvaluationRecordsEditor({
  value,
  onChange,
  disabled,
  readOnly,
}: StructuredModelFieldProps) {
  const evaluations: EditableEvaluationRecord[] = Array.isArray(value)
    ? value
        .filter(
          (entry): entry is EvaluationRecordConfig =>
            Boolean(entry) && typeof entry === 'object' && !Array.isArray(entry),
        )
        .map((entry) => ({
          model: entry.model,
          benchmark: entry.benchmark,
          benchmark_profile: entry.benchmark_profile,
          reasoning_effort: entry.reasoning_effort,
          metrics: Object.fromEntries(
            Object.entries(entry.metrics ?? {}).map(([metric, metricValue]) => [
              metric,
              String(metricValue),
            ]),
          ),
          source: entry.source,
          measured_at: entry.measured_at,
          metadata: entry.metadata
            ? Object.fromEntries(
                Object.entries(entry.metadata).map(([key, item]) => [key, String(item)]),
              )
            : undefined,
        }))
    : []

  return (
    <ObjectListEditor
      value={evaluations}
      onChange={(nextValue) => onChange?.(nextValue)}
      fields={evaluationFields}
      createItem={() => ({ model: '', benchmark: '', metrics: {} })}
      addLabel="Add record"
      emptyLabel="No operator evaluation records configured. Built-in evidence is supplied by the catalog."
      itemLabel={(item, index) =>
        [item.model?.trim(), item.benchmark?.trim()].filter(Boolean).join(' · ') ||
        `Record ${index + 1}`
      }
      itemDescription={(item) =>
        item.metrics && Object.keys(item.metrics).length > 0
          ? `${Object.keys(item.metrics).length} metric${Object.keys(item.metrics).length === 1 ? '' : 's'}`
          : 'Benchmark and metrics required'
      }
      validateItem={(item) => {
        const errors: string[] = []
        if (!item.model?.trim()) errors.push('Model Card is required.')
        if (!item.benchmark?.trim()) errors.push('Benchmark is required.')
        if (!item.metrics || Object.keys(item.metrics).length === 0) {
          errors.push('At least one metric is required.')
        } else if (Object.values(item.metrics).some((metric) => !Number.isFinite(Number(metric)))) {
          errors.push('Every metric value must be numeric.')
        }
        return errors
      }}
      disabled={disabled}
      readOnly={readOnly}
    />
  )
}
