import { describe, expect, it } from 'vitest'
import { canReuseBaseline } from './baselineReuse'
import type { Run } from './types'
import { DEFAULT_LIMITS } from './model'

const baseline: Run = {
  id: 'baseline',
  status: 'failed',
  created_at: '',
  updated_at: '',
  progress: { total: 42, completed: 41, failed: 1 },
  manifest: {
    version: 'sr-bench-1.0',
    name: 'Frozen baseline',
    mode: 'live',
    profile: 'smoke',
    seed: 42,
    targets: [{ id: 'single', model: 'provider/model', kind: 'single', base_url: 'http://model' }],
    limits: DEFAULT_LIMITS,
    sampling: { temperature: 0, max_tokens: 512 },
  },
}

describe('frozen baseline reuse', () => {
  it('keeps an explicit failed result while allowing the full frozen protocol to be reused', () => {
    expect(canReuseBaseline(baseline)).toBe(true)
    expect(baseline.status).toBe('failed')
    expect(baseline.progress).toEqual({ total: 42, completed: 41, failed: 1 })
  })

  it('allows terminated full plans but never active, preview, MoM-only or recovery subsets', () => {
    for (const status of ['completed', 'cancelled', 'interrupted'] as const)
      expect(canReuseBaseline({ ...baseline, status })).toBe(true)
    for (const status of ['queued', 'running'] as const)
      expect(canReuseBaseline({ ...baseline, status })).toBe(false)
    for (const patch of [
      { mode: 'preview' as const },
      { targets: [{ ...baseline.manifest.targets[0], kind: 'mom' as const }] },
      { recovery: { parent_run_id: 'parent' } },
      { execution_cells: [] },
    ])
      expect(canReuseBaseline({ ...baseline, manifest: { ...baseline.manifest, ...patch } })).toBe(
        false,
      )
  })
})
