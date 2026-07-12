import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { matchesFleetSimSearch, parseArrivalRateCheckpoints } from './fleetSimListSupport'

describe('Fleet simulator list support', () => {
  it('matches normalized terms across bounded client-side list fields', () => {
    const rows = Array.from({ length: 500 }, (_, index) => ({
      id: `job-${index + 1}`,
      type: index % 2 === 0 ? 'optimize' : 'simulate',
      status: index === 419 ? 'failed' : 'done',
    }))

    const result = rows.filter((row) =>
      matchesFleetSimSearch('JOB-420', [row.id, row.type, row.status]),
    )
    expect(result.map((row) => row.id)).toEqual(['job-420'])
  })

  it('preserves ordered numeric checkpoint payloads', () => {
    expect(parseArrivalRateCheckpoints(['100', '200.5', '500'])).toEqual([100, 200.5, 500])
  })

  it('rejects empty, malformed, non-positive, and duplicate checkpoints', () => {
    expect(() => parseArrivalRateCheckpoints([])).toThrow(/at least one/i)
    expect(() => parseArrivalRateCheckpoints(['100', 'nope'])).toThrow(/positive number/i)
    expect(() => parseArrivalRateCheckpoints(['0'])).toThrow(/positive number/i)
    expect(() => parseArrivalRateCheckpoints(['100', '100'])).toThrow(/unique/i)
  })

  it('keeps search, bounded pagination, and confirmation dialogs on every catalog', () => {
    const sources = [
      readFileSync(new URL('./FleetSimWorkloadsPage.tsx', import.meta.url), 'utf8'),
      readFileSync(new URL('./FleetSimFleetsPage.tsx', import.meta.url), 'utf8'),
      readFileSync(new URL('./FleetSimRunsPage.tsx', import.meta.url), 'utf8'),
    ]

    for (const source of sources) {
      expect(source).toContain('onSearchChange=')
      expect(source).toContain('pagination={{')
      expect(source).toContain('<ConfirmDialog')
      expect(source).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
    }
  })
})
