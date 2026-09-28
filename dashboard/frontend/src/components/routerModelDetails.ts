import { getRouterModelStateLabel, type RouterModelInfo } from '../utils/routerRuntime'
import {
  formatRouterModelLabel,
  formatRouterModelTokens,
  getRouterModelArtifactPath,
  getRouterModelDevice,
  getRouterModelInputLimits,
  getRouterModelKind,
} from './routerModelPresentation'

export interface RouterModelDetailRow {
  label: string
  value: string
}

export interface RouterModelDetailSection {
  title: string
  rows: RouterModelDetailRow[]
  wide?: boolean
  collapsible?: boolean
}

const PRESENTED_METADATA = new Set([
  'binding',
  'deployment',
  'provider',
  'device',
  'precision',
  'model_type',
  'input_max_tokens',
  'document_max_tokens',
  'forward_max_tokens',
  'max_sequence_length',
  'window_size',
  'window_overlap',
  'overflow',
  'resource_id',
  'default_layer',
  'default_dimension',
])

function optionalRow(label: string, value?: string | number): RouterModelDetailRow[] {
  return value === undefined || value === '' ? [] : [{ label, value: String(value) }]
}

function consumerDetails(consumer: RouterModelInfo): string {
  const limits = getRouterModelInputLimits(consumer)
  return [
    getRouterModelStateLabel(consumer),
    consumer.metadata?.deployment,
    limits.input !== undefined &&
      `${limits.overflow === 'window' ? 'Document' : 'Input'} budget ${formatRouterModelTokens(limits.document ?? limits.input)}`,
    consumer.metadata?.default_layer && `Layer ${consumer.metadata.default_layer}`,
    consumer.metadata?.default_dimension && `${consumer.metadata.default_dimension} dimensions`,
  ]
    .filter(Boolean)
    .join(' · ')
}

export function buildRouterModelDetailSections(
  model: RouterModelInfo,
  consumers: RouterModelInfo[],
): RouterModelDetailSection[] {
  const shared = consumers.length > 1
  const metadata = Object.fromEntries(
    Object.entries(model.metadata ?? {}).filter(([key, value]) =>
      consumers.every((consumer) => consumer.metadata?.[key] === value),
    ),
  )
  const commonModel = { ...model, metadata }
  const limits = getRouterModelInputLimits(commonModel)
  const limitsVary =
    shared &&
    consumers.some((consumer) => {
      const other = getRouterModelInputLimits(consumer)
      return (other.document ?? other.input) !== (limits.document ?? limits.input)
    })
  const inputRows: RouterModelDetailRow[] = [
    {
      label: limits.overflow === 'window' ? 'Document budget' : 'Input budget',
      value: limitsVary
        ? 'Varies by consumer'
        : formatRouterModelTokens(limits.document ?? limits.input),
    },
    ...(limits.window !== undefined || limits.overflow === 'window'
      ? [{ label: 'Physical token window', value: formatRouterModelTokens(limits.window) }]
      : []),
    ...(limits.forward !== undefined && limits.forward !== limits.window
      ? [{ label: 'Single-forward capacity', value: formatRouterModelTokens(limits.forward) }]
      : []),
    ...(limits.overlap !== undefined
      ? [{ label: 'Window overlap', value: formatRouterModelTokens(limits.overlap) }]
      : []),
    ...optionalRow('Overflow policy', limits.overflow),
  ]
  const executionRows: RouterModelDetailRow[] = [
    {
      label: 'Provider',
      value: metadata.provider ? formatRouterModelLabel(metadata.provider) : 'Not reported',
    },
    { label: 'Device', value: getRouterModelDevice(commonModel).label },
    ...optionalRow('Precision', metadata.precision),
    ...optionalRow('Architecture', metadata.model_type),
    ...optionalRow('Memory usage', model.memory_usage),
  ]
  const modelRows: RouterModelDetailRow[] = [
    { label: 'Hugging Face repository', value: model.registry?.repo_id || 'Not reported' },
    { label: 'Task', value: getRouterModelKind(model) },
    ...optionalRow('Parameters', model.registry?.parameter_size),
    ...(limits.publishedContext !== undefined
      ? [
          {
            label: 'Published model context',
            value: formatRouterModelTokens(limits.publishedContext),
          },
        ]
      : []),
    ...optionalRow('Embedding dimensions', model.registry?.embedding_dim),
    ...optionalRow('Labels', model.registry?.num_classes),
    ...optionalRow('License', model.registry?.license),
  ]
  const technicalRows: RouterModelDetailRow[] = [
    ...(!shared ? [{ label: 'Runtime key', value: model.name }] : []),
    { label: 'Artifact path', value: getRouterModelArtifactPath(model) || 'Not reported' },
    ...optionalRow('Repository revision', model.registry?.revision),
    ...optionalRow('Base model', model.registry?.base_model),
    ...optionalRow('Languages', model.registry?.languages?.join(', ')),
    ...optionalRow('Datasets', model.registry?.datasets?.join(', ')),
    ...optionalRow('Resource ID', metadata.resource_id),
    ...optionalRow('Load time', model.load_time),
    ...Object.entries(metadata)
      .filter(([key]) => !PRESENTED_METADATA.has(key))
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, value]) => ({ label: formatRouterModelLabel(key), value })),
  ]
  return [
    { title: 'Execution', rows: executionRows },
    { title: 'Input limits', rows: inputRows },
    { title: 'Model metadata', rows: modelRows, wide: true },
    {
      title: shared ? `Consumers (${consumers.length})` : 'Consumer',
      wide: true,
      rows: consumers.map((consumer) => ({
        label: `${consumer.recipe || 'default'} / ${consumer.metadata?.binding || consumer.name}`,
        value: consumerDetails(consumer),
      })),
    },
    { title: 'Technical details', rows: technicalRows, wide: true, collapsible: true },
  ]
}
