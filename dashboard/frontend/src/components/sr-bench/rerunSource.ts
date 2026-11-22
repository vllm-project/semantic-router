import { DEFAULT_LIMITS } from './model'
import type { SamplingSettings } from './RunSettings'
import type { Manifest, Run } from './types'

/** Settings carried over from a saved run when it is evaluated again. */
export interface RerunSource {
  name: string
  profile: string
  benchmarks: string[]
  targetIDs: string[]
  limits: Manifest['limits']
  sampling: SamplingSettings
  outputPolicy: 'bounded' | 'native'
  costPolicy: 'require_priced' | 'capability_only'
  dataset?: { path: string; sha256: string }
}

export function rerunSource(run: Run): RerunSource {
  const manifest = run.manifest
  const dataset = manifest.dataset
  return {
    name: `${manifest.name} · re-run`,
    profile: manifest.profile,
    benchmarks: dataset?.benchmarks ?? [],
    targetIDs: manifest.targets.map((target) => target.id),
    limits: { ...DEFAULT_LIMITS, ...manifest.limits },
    sampling: {
      temperature: manifest.sampling?.temperature,
      top_p: manifest.sampling?.top_p,
      seed: manifest.sampling?.seed,
    },
    outputPolicy: manifest.output_policy === 'native' ? 'native' : 'bounded',
    costPolicy: manifest.cost_policy === 'capability_only' ? 'capability_only' : 'require_priced',
    ...(dataset?.path && dataset.sha256
      ? { dataset: { path: dataset.path, sha256: dataset.sha256 } }
      : {}),
  }
}
