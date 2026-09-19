import type { Run } from './types'

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`
  if (value && typeof value === 'object')
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, entry]) => `${JSON.stringify(key)}:${canonical(entry)}`)
      .join(',')}}`
  return JSON.stringify(value) ?? 'undefined'
}

/** Collection metadata can rule a run out; the comparison service still validates full evidence. */
export function comparisonEligibility(baseline: Run | undefined, candidate: Run) {
  if (!baseline) return { eligible: false, reason: 'Choose a baseline first.' }
  if (
    baseline.status !== 'completed' ||
    baseline.manifest.mode !== 'live' ||
    !baseline.manifest.targets.some((target) => target.kind === 'single')
  )
    return {
      eligible: false,
      reason: 'Choose a completed live baseline containing a single model.',
    }
  if (baseline.id === candidate.id)
    return { eligible: false, reason: 'This is the selected baseline.' }
  if (candidate.manifest.mode !== 'live')
    return {
      eligible: false,
      reason: 'Preview and replay runs do not measure live quality or savings.',
    }
  if (candidate.status !== 'completed')
    return { eligible: false, reason: `Run is ${candidate.status}; both runs must be completed.` }
  const left = baseline.manifest as unknown as Record<string, unknown>
  const right = candidate.manifest as unknown as Record<string, unknown>
  for (const [key, label] of [
    ['case_sha256', 'Frozen cases differ'],
    ['sampling', 'Sampling settings differ'],
    ['limits', 'Execution limits differ'],
    ['benchmark_weights', 'Benchmark weights differ'],
    ['adapter_versions', 'Benchmark adapter versions differ'],
    ['benchmark_options', 'Benchmark grading or execution settings differ'],
  ]) {
    if (
      left[key] !== undefined &&
      right[key] !== undefined &&
      canonical(left[key]) !== canonical(right[key])
    )
      return { eligible: false, reason: `${label}. Use the same evaluation protocol.` }
  }
  const prices = new Map<string, string>()
  const profiles = new Map<string, string>()
  for (const manifest of [baseline.manifest, candidate.manifest]) {
    for (const target of manifest.targets) {
      for (const [model, price] of Object.entries(target.prices ?? {})) {
        const value = canonical(price)
        if (prices.has(model) && prices.get(model) !== value)
          return { eligible: false, reason: `Frozen prices differ for ${model}.` }
        prices.set(model, value)
      }
      if (target.kind === 'single') {
        const profile = canonical({ ...manifest.sampling, ...target.request_params })
        if (profiles.has(target.model) && profiles.get(target.model) !== profile)
          return { eligible: false, reason: `Request settings differ for ${target.model}.` }
        profiles.set(target.model, profile)
      }
    }
  }
  return { eligible: true, reason: 'Ready for final evidence check.' }
}
