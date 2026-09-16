import type { RouterModelInfo } from '../utils/routerRuntime'

// Local exports preserve the public Vela basename and append their derivation
// identity. Only hide this recognized suffix in the title, never in provenance.
const VELA_DERIVATION = /^(Vela-[\w.-]+)-CK-\d+[KM]?-local-[a-f\d]{8,64}$/i

function basename(value: string): string {
  return value.replace(/\\/g, '/').replace(/\/+$/, '').split('/').pop() || ''
}

export function getRouterModelArtifactPath(model: RouterModelInfo): string {
  const runtimePath = model.model_path?.match(/path=([^,)]+)/)?.[1]?.trim()
  return (
    model.resolved_model_path || runtimePath || model.model_path || model.registry?.local_path || ''
  )
}

export function isLocalDerivedRouterModel(model: RouterModelInfo): boolean {
  return VELA_DERIVATION.test(basename(getRouterModelArtifactPath(model)))
}

export function getRouterModelDisplayName(model: RouterModelInfo): string {
  const artifactName = basename(getRouterModelArtifactPath(model))
  const derived = artifactName.match(VELA_DERIVATION)
  if (derived) return derived[1]
  return basename(model.registry?.repo_id || '') || artifactName || model.name
}

export function getRouterModelPreviewName(model: RouterModelInfo): {
  title: string
  subtitle?: string
} {
  const name = getRouterModelDisplayName(model)
  const vela = name.match(/^Vela-([\d.]+)-Encoder-(\d+[MB])-(Domain|FactCheck|Feedback|Embedding)$/)
  if (!vela) return { title: name }
  return {
    title: `Vela ${vela[3] === 'FactCheck' ? 'Fact Check' : vela[3]}`,
    subtitle: `v${vela[1]} · ${vela[2]} encoder`,
  }
}

function positiveTokenCount(value: string | number | undefined): number | undefined {
  if (value === undefined || !/^\d+$/.test(String(value))) return undefined
  const count = Number(value)
  return Number.isSafeInteger(count) && count > 0 ? count : undefined
}

export function getRouterModelContext(model: RouterModelInfo): {
  tokens?: number
  source: 'runtime' | 'registry' | 'unknown'
  label: string
} {
  const runtime = positiveTokenCount(model.metadata?.max_sequence_length)
  const registry = positiveTokenCount(model.registry?.max_context_length)
  const tokens = runtime ?? registry
  return {
    tokens,
    source: runtime !== undefined ? 'runtime' : registry !== undefined ? 'registry' : 'unknown',
    label: tokens === undefined ? 'Not reported' : `${tokens.toLocaleString('en-US')} tokens`,
  }
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
