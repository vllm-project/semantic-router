import { KeyValueEditor } from '../components/KeyValueEditor'
import { ObjectListEditor, type ObjectEditorField } from '../components/ObjectListEditor'
import { StringListEditor } from '../components/StringListEditor'
import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import type { BackendRefEntry, LoRAAdapter, ModelPricing } from './configPageSupport'
import { normalizeModelBackendRefs } from './configPageModelFormSupport'

interface StructuredModelFieldProps {
  value: unknown
  onChange?: (value: unknown) => void
  disabled?: boolean
  readOnly?: boolean
  maskSensitive?: boolean
}

const backendRefFields: ObjectEditorField<BackendRefEntry>[] = [
  { key: 'name', label: 'Reference name', placeholder: 'local-primary' },
  { key: 'backend_id', label: 'Backend ID', placeholder: 'pod-a-0' },
  { key: 'engine_kind', label: 'Engine kind', placeholder: 'vllm' },
  { key: 'type', label: 'Backend type', placeholder: 'chat' },
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
  { key: 'auth_prefix', label: 'Authentication prefix', placeholder: 'Bearer' },
  {
    key: 'extra_headers',
    label: 'Extra headers',
    type: 'key-value',
    fullWidth: true,
    emptyValueLabel: 'No extra headers configured.',
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
