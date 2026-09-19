import { describe, expect, it } from 'vitest'
import { comparisonEligibility } from './comparisonEligibility'
import { makeManifest, DEFAULT_LIMITS } from './model'
import type { Run } from './types'

const manifest = makeManifest(
  'Baseline',
  'live',
  'quick',
  undefined,
  [{ id: 'single', kind: 'single', model: 'model-a', base_url: '/v1' }],
  DEFAULT_LIMITS,
)
const baseline: Run = {
  id: 'baseline',
  status: 'completed',
  created_at: '',
  updated_at: '',
  manifest,
  progress: { total: 2, completed: 2, failed: 0 },
}
const candidate = (changes: Partial<Run> = {}): Run => ({
  ...baseline,
  id: 'candidate',
  ...changes,
})

describe('comparison availability from saved metadata', () => {
  it('does not let preview or unfinished evidence masquerade as comparable', () => {
    expect(comparisonEligibility(undefined, candidate()).eligible).toBe(false)
    expect(comparisonEligibility(baseline, baseline).eligible).toBe(false)
    expect(comparisonEligibility({ ...baseline, status: 'running' }, candidate()).eligible).toBe(
      false,
    )
    expect(comparisonEligibility(baseline, candidate({ status: 'cancelled' })).reason).toContain(
      'cancelled',
    )
    expect(
      comparisonEligibility(baseline, candidate({ manifest: { ...manifest, mode: 'preview' } }))
        .eligible,
    ).toBe(false)
  })
  it('uses frozen case identity and protocol, not names or total cell counts', () => {
    const source = { ...baseline, manifest: { ...manifest, case_sha256: 'same' } }
    const compatible = candidate({
      manifest: {
        ...manifest,
        case_sha256: 'same',
        name: 'Balance R2',
        targets: [{ ...manifest.targets[0], kind: 'mom' }],
      },
      progress: { total: 1, completed: 1, failed: 0 },
    })
    expect(comparisonEligibility(source, compatible).eligible).toBe(true)
    expect(
      comparisonEligibility(source, candidate({ manifest: { ...manifest, case_sha256: 'other' } }))
        .reason,
    ).toContain('Frozen cases differ')
    expect(
      comparisonEligibility(
        source,
        candidate({
          manifest: { ...manifest, sampling: { ...manifest.sampling, temperature: 0.7 } },
        }),
      ).reason,
    ).toContain('Sampling settings differ')
  })
  it('handles object key ordering and shared model price/profile differences', () => {
    const price = { input: 1, cached_input: 0.1, cache_write: 2, output: 4 }
    const source = {
      ...baseline,
      manifest: {
        ...manifest,
        targets: [{ ...manifest.targets[0], prices: { 'model-a': price } }],
      },
    }
    expect(
      comparisonEligibility(
        source,
        candidate({
          manifest: {
            ...manifest,
            targets: [
              {
                ...manifest.targets[0],
                prices: { 'model-a': { output: 4, cache_write: 2, cached_input: 0.1, input: 1 } },
              },
            ],
          },
        }),
      ).eligible,
    ).toBe(true)
    expect(
      comparisonEligibility(
        source,
        candidate({
          manifest: {
            ...manifest,
            targets: [{ ...manifest.targets[0], prices: { 'model-a': { ...price, input: 3 } } }],
          },
        }),
      ).reason,
    ).toContain('Frozen prices differ')
    expect(
      comparisonEligibility(
        source,
        candidate({
          manifest: {
            ...manifest,
            targets: [{ ...manifest.targets[0], request_params: { temperature: 1 } }],
          },
        }),
      ).reason,
    ).toContain('Request settings differ')
  })
  it('leaves absent collection details for final service validation', () => {
    expect(comparisonEligibility(baseline, candidate())).toEqual({
      eligible: true,
      reason: 'Ready for final evidence check.',
    })
  })
})
