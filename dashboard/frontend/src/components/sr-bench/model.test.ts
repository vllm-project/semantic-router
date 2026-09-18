import { describe, expect, it } from 'vitest'
import {
  DEFAULT_LIMITS,
  reportDistribution,
  makeManifest,
  money,
  tokenTotal,
  validateManifest,
} from './model'

const dataset = {
  id: 'quick-v1',
  path: 'datasets/quick.jsonl',
  sha256: 'a'.repeat(64),
  case_count: 500,
}
const single = {
  id: 'baseline',
  kind: 'single' as const,
  model: 'model-a',
  base_url: 'http://localhost:8000/v1',
}

describe('sr-bench run contract', () => {
  it('binds a prepared dataset and snapshots target and limit values', () => {
    const limits = { ...DEFAULT_LIMITS }
    const target = { ...single }
    const manifest = makeManifest(' Review ', 'live', 'quick', dataset, [target], limits)
    target.model = 'other-model'
    limits.max_output_tokens = 99
    expect(manifest.name).toBe('Review')
    expect(manifest.dataset).toEqual({ path: dataset.path, sha256: dataset.sha256 })
    expect(manifest.targets[0].model).toBe('model-a')
    expect(manifest.sampling.max_tokens).toBe(DEFAULT_LIMITS.max_output_tokens)
    expect(validateManifest(manifest)).toBeNull()
  })
  it('rejects invalid limits, duplicate identities and non-MoM preview', () => {
    const manifest = makeManifest('Review', 'live', 'quick', dataset, [single], DEFAULT_LIMITS)
    expect(validateManifest({ ...manifest, targets: [single, single] })).toContain('unique')
    expect(validateManifest({ ...manifest, mode: 'preview' })).toContain('MoM')
    expect(
      validateManifest({ ...manifest, limits: { ...manifest.limits, max_cost_usd: 0 } }),
    ).toContain('positive')
    expect(
      validateManifest({ ...manifest, limits: { ...manifest.limits, idle_timeout_s: 9999 } }),
    ).toContain('Idle')
  })
  it('does not turn unknown cost into free usage and sums normalized billing buckets', () => {
    expect(money(null)).toBe('—')
    expect(money(0)).toBe('$0.00000')
    expect(
      tokenTotal({
        input_tokens: 20,
        cached_input_tokens: 80,
        cache_write_tokens: 0,
        output_tokens: 20,
      }),
    ).toBe(120)
    expect(tokenTotal({ input_tokens: 100 })).toBeNull()
  })
  it('uses full report routing counts while keeping targets separate', () => {
    expect(
      reportDistribution(
        [
          { id: 'balance', selected_models: { 'model-a': 200, 'model-b': 50 } },
          { id: 'single', selected_models: { 'model-a': 250 } },
          { id: 'unknown' },
        ],
        'selected_models',
      ),
    ).toEqual([
      ['single: model-a', 250],
      ['balance: model-a', 200],
      ['balance: model-b', 50],
    ])
  })
})
