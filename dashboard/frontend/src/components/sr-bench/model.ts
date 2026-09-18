import type { Dataset, Manifest, Target, TargetMetrics } from './types'

export const DEFAULT_LIMITS: Manifest['limits'] = {
  concurrency: 1,
  total_timeout_s: 180,
  idle_timeout_s: 30,
  max_output_tokens: 4096,
  max_output_chars: 131072,
  repetition_window: 128,
  repetition_limit: 5,
  max_cost_usd: 5,
  max_run_seconds: 1800,
  max_calls_per_case: 32,
}

export function makeManifest(
  name: string,
  mode: Manifest['mode'],
  profile: string,
  dataset: Dataset | undefined,
  targets: Target[],
  limits: Manifest['limits'],
): Manifest {
  return {
    version: 'sr-bench-1.0',
    name: name.trim(),
    mode,
    profile,
    seed: 20260918,
    ...(dataset ? { dataset: { path: dataset.path, sha256: dataset.sha256 } } : {}),
    targets: targets.map((target) => ({ ...target })),
    limits: { ...limits },
    sampling: { temperature: 0, top_p: 1, max_tokens: limits.max_output_tokens, seed: 20260918 },
  }
}

export function validateManifest(manifest: Manifest): string | null {
  if (manifest.version !== 'sr-bench-1.0') return 'Use the sr-bench-1.0 manifest version.'
  if (!manifest.name?.trim()) return 'Give this run a name.'
  if (!manifest.dataset && !manifest.cases?.length)
    return 'Select a prepared dataset. Prepare datasets with the sr-bench CLI first.'
  if (!manifest.targets?.length) return 'Add at least one single model or MoM target.'
  if (new Set(manifest.targets.map((target) => target.id)).size !== manifest.targets.length)
    return 'Each target needs a unique name.'
  for (const target of manifest.targets) {
    if (!target.id.trim() || !target.model.trim() || !target.base_url.trim())
      return 'Every target needs a name, model and endpoint.'
    try {
      if (!['http:', 'https:'].includes(new URL(target.base_url).protocol))
        return 'Target endpoints must use HTTP or HTTPS.'
    } catch {
      return 'Enter a valid target endpoint URL.'
    }
  }
  if (manifest.mode === 'preview' && manifest.targets.some((target) => target.kind !== 'mom'))
    return 'Route preview requires MoM targets.'
  if (
    !manifest.limits ||
    Object.values(manifest.limits).some((value) => !Number.isFinite(value) || value <= 0)
  )
    return 'Every run limit must be a positive number.'
  if (manifest.limits.idle_timeout_s > manifest.limits.total_timeout_s)
    return 'Idle timeout must not exceed the total request deadline.'
  return null
}

export const number = (value: unknown, digits = 0): string =>
  typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: digits })
    : '—'
export const percent = (value: unknown): string =>
  typeof value === 'number' && Number.isFinite(value) ? `${number(value * 100, 2)}%` : '—'
export const money = (value: unknown): string =>
  typeof value === 'number' && Number.isFinite(value) ? `$${value.toFixed(5)}` : '—'
export const seconds = (value: unknown): string =>
  typeof value === 'number' && Number.isFinite(value) ? `${number(value, 2)} s` : '—'
export const active = (status: string) => status === 'queued' || status === 'running'

export function reportDistribution(
  targets: TargetMetrics[],
  field: 'selected_models' | 'decisions',
): Array<[string, number]> {
  return targets
    .flatMap((target) =>
      Object.entries(target[field] ?? {})
        .filter(([, count]) => Number.isFinite(count) && count > 0)
        .map(([name, count]): [string, number] => [`${target.id}: ${name}`, count]),
    )
    .sort((a, b) => b[1] - a[1])
}

export function tokenTotal(value: unknown): number | null {
  if (typeof value === 'number') return value
  if (!value || typeof value !== 'object') return null
  const usage = value as Record<string, unknown>
  if (typeof usage.total_tokens === 'number') return usage.total_tokens
  // The service normalizes these into mutually exclusive billed buckets.
  const buckets = ['input_tokens', 'cached_input_tokens', 'cache_write_tokens', 'output_tokens']
  if (buckets.every((key) => typeof usage[key] === 'number'))
    return buckets.reduce((sum, key) => sum + (usage[key] as number), 0)
  return null
}
