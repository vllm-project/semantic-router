import type { APIConfig, TracingConfig } from './configPageSupport'

export type BatchMetricsConfig = NonNullable<
  NonNullable<APIConfig['batch_classification']>['metrics']
>
export type BatchSizeRange = NonNullable<BatchMetricsConfig['batch_size_ranges']>[number]

function parseLegacyJson(value: unknown): unknown {
  if (typeof value !== 'string') return value
  const trimmed = value.trim()
  if (!trimmed) return undefined

  try {
    return JSON.parse(trimmed)
  } catch {
    return value
  }
}

function requireRecord(value: unknown, label: string): Record<string, unknown> {
  const parsed = parseLegacyJson(value)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a structured object.`)
  }
  return parsed as Record<string, unknown>
}

function requireText(value: unknown, label: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${label} is required.`)
  }
  return value.trim()
}

export function normalizeTracingExporter(value: unknown): TracingConfig['exporter'] {
  const exporter = requireRecord(value, 'Tracing exporter')
  const normalized: TracingConfig['exporter'] = {
    type: requireText(exporter.type, 'Tracing exporter type'),
  }

  if (exporter.endpoint !== undefined) {
    normalized.endpoint = requireText(exporter.endpoint, 'Tracing exporter endpoint')
  }
  if (exporter.insecure !== undefined) {
    if (typeof exporter.insecure !== 'boolean') {
      throw new Error('Tracing exporter insecure must be true or false.')
    }
    normalized.insecure = exporter.insecure
  }
  return normalized
}

export function normalizeTracingSampling(value: unknown): TracingConfig['sampling'] {
  const sampling = requireRecord(value, 'Tracing sampling')
  const normalized: TracingConfig['sampling'] = {
    type: requireText(sampling.type, 'Tracing sampling type'),
  }

  if (sampling.rate !== undefined) {
    if (
      typeof sampling.rate !== 'number' ||
      !Number.isFinite(sampling.rate) ||
      sampling.rate < 0 ||
      sampling.rate > 1
    ) {
      throw new Error('Tracing sampling rate must be between 0 and 1.')
    }
    normalized.rate = sampling.rate
  }
  return normalized
}

export function normalizeTracingResource(value: unknown): TracingConfig['resource'] {
  const resource = requireRecord(value, 'Tracing resource')
  return {
    service_name: requireText(resource.service_name, 'Tracing service name'),
    service_version: requireText(resource.service_version, 'Tracing service version'),
    deployment_environment: requireText(
      resource.deployment_environment,
      'Tracing deployment environment',
    ),
  }
}

export function normalizeTracingConfig(data: TracingConfig): TracingConfig {
  if (!data.provider?.trim()) throw new Error('Tracing provider is required.')
  return {
    ...data,
    provider: data.provider.trim(),
    exporter: normalizeTracingExporter(data.exporter),
    sampling: normalizeTracingSampling(data.sampling),
    resource: normalizeTracingResource(data.resource),
  }
}

export function createTracingEditorValue(data: TracingConfig): TracingConfig {
  const readOrFallback = <T,>(read: () => T, fallback: T): T => {
    try {
      return read()
    } catch {
      return fallback
    }
  }

  return {
    ...data,
    provider: data.provider?.trim() || 'otlp',
    exporter: readOrFallback(() => normalizeTracingExporter(data.exporter), { type: 'otlp' }),
    sampling: readOrFallback(() => normalizeTracingSampling(data.sampling), {
      type: 'probabilistic',
      rate: 0.1,
    }),
    resource: readOrFallback(() => normalizeTracingResource(data.resource), {
      service_name: 'semantic-router',
      service_version: '1.0.0',
      deployment_environment: 'production',
    }),
  }
}

export function normalizeBatchSizeRanges(value: unknown): BatchSizeRange[] {
  const parsed = parseLegacyJson(value)
  if (parsed === undefined || parsed === null) return []
  if (!Array.isArray(parsed)) throw new Error('Batch size ranges must be a structured list.')

  return parsed.map((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new Error(`Batch size range ${index + 1} must be a structured object.`)
    }
    const range = item as Record<string, unknown>
    if (!Number.isInteger(range.min) || (range.min as number) < 0) {
      throw new Error(`Batch size range ${index + 1} minimum must be a non-negative integer.`)
    }
    if (!Number.isInteger(range.max) || (range.max as number) < (range.min as number)) {
      throw new Error(`Batch size range ${index + 1} maximum must be an integer at least the minimum.`)
    }
    return {
      min: range.min as number,
      max: range.max as number,
      label: requireText(range.label, `Batch size range ${index + 1} label`),
    }
  })
}

export function normalizeMetricBuckets(
  value: unknown,
  label: string,
  options: { integer?: boolean } = {},
): number[] {
  const parsed = parseLegacyJson(value)
  if (parsed === undefined || parsed === null) return []
  if (!Array.isArray(parsed)) throw new Error(`${label} must be a list of numbers.`)

  const normalized = parsed.map((item, index) => {
    const numberValue = typeof item === 'number' ? item : typeof item === 'string' ? Number(item) : NaN
    if (!Number.isFinite(numberValue) || numberValue <= 0) {
      throw new Error(`${label} entry ${index + 1} must be a positive number.`)
    }
    if (options.integer && !Number.isInteger(numberValue)) {
      throw new Error(`${label} entry ${index + 1} must be an integer.`)
    }
    return numberValue
  })

  normalized.forEach((bucket, index) => {
    if (index > 0 && bucket <= normalized[index - 1]) {
      throw new Error(`${label} must be strictly increasing.`)
    }
  })
  return normalized
}

export function normalizeBatchMetrics(data: BatchMetricsConfig): BatchMetricsConfig {
  const record = data as BatchMetricsConfig & Record<string, unknown>
  const normalized: BatchMetricsConfig = { ...data }

  if (record.batch_size_ranges !== undefined) {
    normalized.batch_size_ranges = normalizeBatchSizeRanges(record.batch_size_ranges)
  }
  if (record.duration_buckets !== undefined) {
    normalized.duration_buckets = normalizeMetricBuckets(
      record.duration_buckets,
      'Duration buckets',
    )
  }
  if (record.size_buckets !== undefined) {
    normalized.size_buckets = normalizeMetricBuckets(record.size_buckets, 'Size buckets', {
      integer: true,
    })
  }
  return normalized
}
