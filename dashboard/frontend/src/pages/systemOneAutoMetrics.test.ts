import { describe, expect, it, vi } from 'vitest'
import {
  loadSystemOneAutoMetrics,
  readAutoStages,
  readAutoValue,
  systemOneAutoQueries,
} from './systemOneAutoMetrics'
const vector = (result: unknown[]) => ({
  status: 'success',
  data: { resultType: 'vector', result },
})
describe('native auto observations', () => {
  it('keeps missing and invalid samples unavailable', () => {
    expect(readAutoValue(vector([]))).toBeNull()
    expect(readAutoValue(vector([{ value: [0, 'NaN'] }]))).toBeNull()
    expect(readAutoValue(vector([{ value: [0, '0'] }]))).toBe(0)
    expect(() => readAutoValue({ status: 'error' })).toThrow()
  })
  it('groups actual stage outcomes without counting a rejection as an error', () => {
    const item = (stage: string, outcome: string, count: string) => ({
      metric: { algorithm: 'cascade', model: 'fast', stage, outcome },
      value: [0, count],
    })
    expect(
      readAutoStages(
        vector([
          item('first', 'accepted', '7'),
          item('first', 'rejected', '2'),
          item('first', 'identity_mismatch', '1'),
          item('later', 'error', 'NaN'),
        ]),
      ),
    ).toEqual([
      {
        label: 'cascade / first · fast',
        algorithm: 'cascade',
        stage: 'first',
        model: 'fast',
        accepted: 7,
        rejected: 2,
        failed: 1,
      },
    ])
  })
  it('uses request outcomes and the specified stage window', () => {
    expect(systemOneAutoQueries(900).stages).toContain('[900s]')
    expect(systemOneAutoQueries(900).unresolved).toContain('outcome!="resolved"')
    expect(systemOneAutoQueries(900).p95).toContain('sum by (le)')
  })
  it('preserves available metrics when one query fails', async () => {
    const fetcher = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url) => {
      const query = new URL(String(url), 'http://localhost').searchParams.get('query') ?? ''
      if (query.includes('histogram_quantile')) throw new Error('Unavailable')
      return new Response(
        JSON.stringify(vector(query.includes('sum by (algorithm') ? [] : [{ value: [0, '2'] }])),
        { status: 200 },
      )
    })
    try {
      const snapshot = await loadSystemOneAutoMetrics(900, new AbortController().signal)
      expect(snapshot.unavailable).toEqual(['p95'])
      expect(snapshot.values.calls).toBe(2)
      expect(snapshot.stages).toEqual([])
    } finally {
      fetcher.mockRestore()
    }
  })
})
