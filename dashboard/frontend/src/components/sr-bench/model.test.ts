import { describe, expect, it } from 'vitest'
import {
  DEFAULT_LIMITS,
  effectiveRequestProfile,
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
  it('preserves the registered dataset sampling seed instead of substituting a default', () => {
    const manifest = makeManifest(
      'Seeded',
      'live',
      'quick',
      { ...dataset, seed: 73 },
      [single],
      DEFAULT_LIMITS,
    )
    expect(manifest.seed).toBe(73)
    expect(manifest.sampling.seed).toBe(73)
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
  it('preserves registered overrides and rejects a fixed output profile above the run cap', () => {
    const native = {
      ...single,
      request_params: { max_tokens: 4096, temperature: 1, top_p: 0.95, seed: 42 },
    }
    const manifest = makeManifest('Native', 'live', 'quick', dataset, [native], {
      ...DEFAULT_LIMITS,
      max_output_tokens: 512,
    })
    const before = JSON.stringify(manifest)
    expect(effectiveRequestProfile(native, manifest.sampling)).toMatchObject({
      max_tokens: 4096,
      temperature: 1,
      top_p: 0.95,
      seed: 42,
    })
    expect(validateManifest(manifest)).toContain(
      'Target model-a has a registered output limit of 4096 tokens, above the run cap of 512',
    )
    expect(JSON.stringify(manifest)).toBe(before)
    expect(
      validateManifest({ ...manifest, limits: { ...manifest.limits, max_output_tokens: 4096 } }),
    ).toBeNull()
    const noFixedCap = { ...single, request_params: { temperature: 1, top_p: 0.95 } }
    expect(effectiveRequestProfile(noFixedCap, manifest.sampling).max_tokens).toBe(512)
    expect(validateManifest({ ...manifest, targets: [noFixedCap] })).toBeNull()
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
        {
          targets: [
            { ...single, id: 'single' },
            { ...single, id: 'balance', model: 'connected-balance', kind: 'mom' },
          ],
        },
      ),
    ).toEqual([
      ['model-a: model-a', 250],
      ['connected-balance: model-a', 200],
      ['connected-balance: model-b', 50],
    ])
  })
  it('validates typed sampling and limit fields before preparing a plan', () => {
    const manifest = makeManifest(
      'Typed settings',
      'live',
      'quick',
      dataset,
      [single],
      DEFAULT_LIMITS,
    )
    for (const [sampling, message] of [
      [{ temperature: -0.1 }, 'Temperature'],
      [{ temperature: 2.1 }, 'Temperature'],
      [{ temperature: Number.NaN }, 'Temperature'],
      [{ top_p: 1.1 }, 'Top P'],
      [{ seed: 1.5 }, 'Sampling seed'],
    ] as const)
      expect(
        validateManifest({ ...manifest, sampling: { ...manifest.sampling, ...sampling } }),
      ).toContain(message)
    for (const [limits, message] of [
      [{ concurrency: 33 }, 'Concurrency'],
      [{ concurrency: 1.5 }, 'Concurrency'],
      [{ max_calls_per_case: 1.5 }, 'whole numbers'],
      [{ max_output_tokens: 512.5 }, 'whole numbers'],
      [{ total_timeout_s: 1801 }, 'Request deadline'],
      [{ case_timeout_s: 0 }, 'positive'],
    ] as const)
      expect(
        validateManifest({ ...manifest, limits: { ...manifest.limits, ...limits } }),
      ).toContain(message)
    expect(
      validateManifest({
        ...manifest,
        sampling: { ...manifest.sampling, temperature: 2, top_p: 0, seed: -1 },
      }),
    ).toBeNull()
    expect(manifest.limits).not.toHaveProperty('case_timeout_s')
  })
  it('preserves omitted provider sampling defaults in a frozen manifest', () => {
    const manifest = makeManifest('Frozen', 'live', 'quick', dataset, [single], DEFAULT_LIMITS)
    delete manifest.sampling.top_p
    delete manifest.sampling.seed
    const before = JSON.stringify(manifest)
    expect(validateManifest(manifest)).toBeNull()
    expect(JSON.stringify(manifest)).toBe(before)
    expect(manifest.sampling).not.toHaveProperty('top_p')
    expect(manifest.sampling).not.toHaveProperty('seed')
  })
})
