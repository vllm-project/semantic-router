import type { RouterModelInfo } from '../utils/routerRuntime'

export function getRouterModelArtifactPath(model: RouterModelInfo): string {
  const runtimePath = model.model_path?.match(/path=([^,)]+)/)?.[1]?.trim()
  return (
    model.resolved_model_path || runtimePath || model.model_path || model.registry?.local_path || ''
  )
}

export function getRouterModelDisplayName(model: RouterModelInfo): string {
  // A local export path is provenance, not evidence of its upstream identity.
  return model.registry?.repo_id?.trim() || model.name
}

export function getRouterModelPreviewName(model: RouterModelInfo): {
  title: string
  subtitle?: string
} {
  const repository = model.registry?.repo_id?.trim()
  if (!repository) {
    return { title: model.name, subtitle: 'Runtime identity · repository not reported' }
  }
  const separator = repository.lastIndexOf('/')
  return {
    title: repository.slice(separator + 1),
    subtitle: separator > 0 ? repository.slice(0, separator) : undefined,
  }
}

function positiveTokenCount(value: string | number | undefined): number | undefined {
  if (value === undefined || !/^\d+$/.test(String(value))) return undefined
  const count = Number(value)
  return Number.isSafeInteger(count) && count > 0 ? count : undefined
}

export function formatRouterModelTokens(tokens?: number): string {
  return tokens === undefined ? 'Not reported' : `${tokens.toLocaleString('en-US')} tokens`
}

export function getRouterModelInputLimits(model: RouterModelInfo): {
  input?: number
  document?: number
  forward?: number
  window?: number
  overlap?: number
  publishedContext?: number
  overflow?: string
} {
  const metadata = model.metadata
  const overlap = metadata?.window_overlap
  const input =
    positiveTokenCount(metadata?.input_max_tokens) ??
    positiveTokenCount(metadata?.max_sequence_length)
  return {
    // max_sequence_length is the legacy name for the configured input budget.
    // Never use it as a physical window when a document is scanned in chunks.
    input,
    document:
      metadata?.overflow === 'window'
        ? (positiveTokenCount(metadata?.document_max_tokens) ?? input)
        : undefined,
    forward: positiveTokenCount(metadata?.forward_max_tokens),
    window: positiveTokenCount(metadata?.window_size),
    overlap: overlap === '0' ? 0 : positiveTokenCount(overlap),
    publishedContext: positiveTokenCount(model.registry?.max_context_length),
    overflow: metadata?.overflow,
  }
}

export function getRouterModelKind(model: RouterModelInfo): string {
  return formatRouterModelLabel(model.registry?.purpose || model.type)
}

export function formatRouterModelLabel(value?: string): string {
  const overrides: Record<string, string> = {
    amd: 'AMD',
    cpu: 'CPU',
    lora: 'LoRA',
    mmbert: 'mmBERT',
    nli: 'NLI',
    pii: 'PII',
    ort: 'ONNX Runtime',
    rocm: 'ROCm',
    migraphx: 'MIGraphX',
  }
  return (value || 'Unknown')
    .replace(/[_-]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => overrides[word.toLowerCase()] ?? word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

export function getRouterModelDevice(model: RouterModelInfo): {
  label: string
  isAmd: boolean
} {
  const device = model.metadata?.device?.trim()
  if (!device) return { label: 'Device not reported', isAmd: false }
  const gpu = device.match(/^(rocm|migraphx|cuda)(?::(\d+))?$/i)
  if (gpu) {
    const kind = gpu[1].toLowerCase()
    const label = { rocm: 'ROCm', migraphx: 'MIGraphX', cuda: 'CUDA' }[kind]
    return { label: `${label}${gpu[2] ? ` ${gpu[2]}` : ''}`, isAmd: kind !== 'cuda' }
  }
  return { label: device.toLowerCase() === 'cpu' ? 'CPU' : device, isAmd: false }
}
