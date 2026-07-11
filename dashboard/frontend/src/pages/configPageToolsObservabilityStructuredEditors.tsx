import { ObjectListEditor, type ObjectEditorField } from '../components/ObjectListEditor'
import { StringListEditor } from '../components/StringListEditor'
import type { TracingConfig } from './configPageSupport'
import {
  normalizeBatchSizeRanges,
  normalizeMetricBuckets,
  normalizeTracingExporter,
  normalizeTracingResource,
  normalizeTracingSampling,
  type BatchSizeRange,
} from './configPageToolsObservabilitySupport'

interface UnknownEditorProps {
  value: unknown
  onChange: (value: unknown) => void
}

interface ExporterEditorValue {
  type?: string
  endpoint?: string
  insecure?: string
}

const exporterFields: ObjectEditorField<ExporterEditorValue>[] = [
  { key: 'type', label: 'Exporter type', placeholder: 'otlp', required: true },
  { key: 'endpoint', label: 'Endpoint', placeholder: 'http://localhost:4318', fullWidth: true },
  { key: 'insecure', label: 'Allow insecure transport', type: 'select', options: ['true', 'false'] },
]

const samplingFields: ObjectEditorField<TracingConfig['sampling']>[] = [
  { key: 'type', label: 'Sampling type', placeholder: 'probabilistic', required: true },
  { key: 'rate', label: 'Rate', type: 'number', min: 0, max: 1, step: 0.01 },
]

const resourceFields: ObjectEditorField<TracingConfig['resource']>[] = [
  { key: 'service_name', label: 'Service name', placeholder: 'semantic-router', required: true },
  { key: 'service_version', label: 'Service version', placeholder: '1.0.0', required: true },
  {
    key: 'deployment_environment',
    label: 'Deployment environment',
    placeholder: 'production',
    required: true,
  },
]

const batchRangeFields: ObjectEditorField<BatchSizeRange>[] = [
  { key: 'label', label: 'Label', placeholder: '1-8', required: true },
  { key: 'min', label: 'Minimum batch size', type: 'number', min: 0, step: 1, required: true },
  { key: 'max', label: 'Maximum batch size', type: 'number', min: 0, step: 1, required: true },
]

function readExporter(value: unknown): TracingConfig['exporter'] {
  try {
    return normalizeTracingExporter(value)
  } catch {
    return { type: 'otlp' }
  }
}

function readSampling(value: unknown): TracingConfig['sampling'] {
  try {
    return normalizeTracingSampling(value)
  } catch {
    return { type: 'probabilistic', rate: 0.1 }
  }
}

function readResource(value: unknown): TracingConfig['resource'] {
  try {
    return normalizeTracingResource(value)
  } catch {
    return {
      service_name: 'semantic-router',
      service_version: '1.0.0',
      deployment_environment: 'production',
    }
  }
}

export function TracingExporterEditor({ value, onChange }: UnknownEditorProps) {
  const exporter = readExporter(value)
  const editorValue: ExporterEditorValue = {
    ...exporter,
    insecure: exporter.insecure === undefined ? undefined : String(exporter.insecure),
  }

  return (
    <ObjectListEditor
      value={[editorValue]}
      onChange={([next = {}]) => {
        const normalized: Record<string, unknown> = { type: next.type ?? '' }
        if (next.endpoint) normalized.endpoint = next.endpoint
        if (next.insecure === 'true') normalized.insecure = true
        if (next.insecure === 'false') normalized.insecure = false
        onChange(normalized)
      }}
      fields={exporterFields}
      createItem={() => ({ type: 'otlp' })}
      itemLabel={() => 'Trace exporter'}
      itemDescription={(item) => item.endpoint || 'Exporter endpoint not set'}
      validateItem={(item) => (item.type?.trim() ? [] : ['Exporter type is required.'])}
      minItems={1}
      maxItems={1}
    />
  )
}

export function TracingSamplingEditor({ value, onChange }: UnknownEditorProps) {
  const sampling = readSampling(value)
  return (
    <ObjectListEditor
      value={[sampling]}
      onChange={([next = { type: '' }]) => onChange(next)}
      fields={samplingFields}
      createItem={() => ({ type: 'probabilistic', rate: 0.1 })}
      itemLabel={() => 'Sampling policy'}
      itemDescription={(item) => item.type || 'Sampling type required'}
      validateItem={(item) => {
        const errors: string[] = []
        if (!item.type?.trim()) errors.push('Sampling type is required.')
        if (item.rate !== undefined && (!Number.isFinite(item.rate) || item.rate < 0 || item.rate > 1)) {
          errors.push('Sampling rate must be between 0 and 1.')
        }
        return errors
      }}
      minItems={1}
      maxItems={1}
    />
  )
}

export function TracingResourceEditor({ value, onChange }: UnknownEditorProps) {
  const resource = readResource(value)
  return (
    <ObjectListEditor
      value={[resource]}
      onChange={([next = { service_name: '', service_version: '', deployment_environment: '' }]) =>
        onChange(next)
      }
      fields={resourceFields}
      createItem={() => ({
        service_name: 'semantic-router',
        service_version: '1.0.0',
        deployment_environment: 'production',
      })}
      itemLabel={(item) => item.service_name || 'Tracing resource'}
      itemDescription={(item) => item.deployment_environment || 'Environment required'}
      validateItem={(item) => [
        ...(!item.service_name?.trim() ? ['Service name is required.'] : []),
        ...(!item.service_version?.trim() ? ['Service version is required.'] : []),
        ...(!item.deployment_environment?.trim() ? ['Deployment environment is required.'] : []),
      ]}
      minItems={1}
      maxItems={1}
    />
  )
}

export function BatchSizeRangesEditor({ value, onChange }: UnknownEditorProps) {
  let ranges: BatchSizeRange[] = []
  try {
    ranges = normalizeBatchSizeRanges(value)
  } catch {
    // Malformed legacy values can be replaced through the empty editor.
  }

  return (
    <ObjectListEditor
      value={ranges}
      onChange={onChange}
      fields={batchRangeFields}
      createItem={(index) => ({ min: index === 0 ? 1 : index * 8 + 1, max: (index + 1) * 8, label: '' })}
      addLabel="Add batch range"
      emptyLabel="No batch-size ranges configured."
      itemLabel={(item, index) => item.label || `Batch range ${index + 1}`}
      itemDescription={(item) => `${item.min}–${item.max} requests`}
      validateItem={(item) => {
        const errors: string[] = []
        if (!item.label?.trim()) errors.push('Range label is required.')
        if (!Number.isInteger(item.min) || item.min < 0) errors.push('Minimum must be a non-negative integer.')
        if (!Number.isInteger(item.max) || item.max < item.min) errors.push('Maximum must be an integer at least the minimum.')
        return errors
      }}
    />
  )
}

function readBucketValues(value: unknown, label: string, integer: boolean): string[] {
  try {
    return normalizeMetricBuckets(value, label, { integer }).map(String)
  } catch {
    if (!Array.isArray(value)) return []
    return value
      .filter((item): item is string | number => typeof item === 'string' || typeof item === 'number')
      .map(String)
  }
}

interface MetricBucketsEditorProps extends UnknownEditorProps {
  label: string
  integer?: boolean
  placeholder: string
}

export function MetricBucketsEditor({
  value,
  onChange,
  label,
  integer = false,
  placeholder,
}: MetricBucketsEditorProps) {
  return (
    <StringListEditor
      value={readBucketValues(value, label, integer)}
      onChange={onChange}
      addLabel={`Add ${label.toLocaleLowerCase().replace(/s$/, '')}`}
      emptyLabel={`No ${label.toLocaleLowerCase()} configured.`}
      itemLabel={label.replace(/s$/, '')}
      placeholder={placeholder}
      validateItem={(item) => {
        const numericValue = Number(item)
        if (!Number.isFinite(numericValue) || numericValue <= 0) return 'Value must be a positive number.'
        if (integer && !Number.isInteger(numericValue)) return 'Value must be an integer.'
        return null
      }}
    />
  )
}
