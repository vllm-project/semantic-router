import { afterEach, describe, expect, it, vi } from 'vitest'
import { benchApi } from './api'
import { experimentApi, type ExperimentPage } from './experimentApi'
import {
  comparisonSelectionReady,
  readExperimentMembers,
  verifyExperimentBaselines,
} from './experimentComparison'
import type { RunChoice, RunOptions } from './types'

vi.mock('./api', () => ({ benchApi: { runOptions: vi.fn() } }))
vi.mock('./experimentApi', () => ({ experimentApi: { runs: vi.fn() } }))
const choice = (run_id: string): RunChoice => ({
  run_id,
  name: run_id,
  profile: 'smoke',
  case_count: 14,
})
const members = (ids: string[], next: number | null = null): ExperimentPage => ({
  experiment: { id: 'exp-a', name: 'Test', created_at: '1', updated_at: '2' },
  members: ids.map((run_id) => ({ run_id, role: 'candidate', hypothesis: '', linked_at: '2' })),
  next_cursor: next,
  has_more: next !== null,
})
const options = (ids: string[], next: string | null = null): RunOptions => ({
  baseline: choice('baseline'),
  baselines: [],
  options: ids.map(choice),
  next_cursor: next,
  has_more: next !== null,
  scanned_pairs: ids.length,
  scan_limited: false,
  unverified_pairs: 0,
  unverified_baselines: 0,
  model_requests: 0,
})
afterEach(() => vi.resetAllMocks())

describe('complete experiment comparison membership', () => {
  it('reads every member page before returning a scope', async () => {
    vi.mocked(experimentApi.runs)
      .mockResolvedValueOnce(
        members(
          Array.from({ length: 20 }, (_, n) => `run-${n}`),
          20,
        ),
      )
      .mockResolvedValueOnce(members(['baseline', 'candidate']))
    const signal = new AbortController().signal
    const result = await readExperimentMembers('exp-a', signal)
    expect(result.length).toBe(22)
    expect(result.some((member) => member.run_id === 'candidate')).toBe(true)
    expect(experimentApi.runs).toHaveBeenLastCalledWith('exp-a', 20, expect.any(AbortSignal))
  })

  it.each(['duplicate', 'revision', 'cursor'] as const)(
    'rejects %s inconsistency instead of exposing partial membership',
    async (problem) => {
      const second = members([problem === 'duplicate' ? 'first' : 'second'])
      if (problem === 'revision') second.experiment.updated_at = '3'
      if (problem === 'cursor') Object.assign(second, { has_more: true, next_cursor: 20 })
      vi.mocked(experimentApi.runs)
        .mockResolvedValueOnce(members(['first'], 20))
        .mockResolvedValueOnce(second)
      await expect(readExperimentMembers('exp-a', new AbortController().signal)).rejects.toThrow()
    },
  )

  it('does not continue paging an abandoned experiment', async () => {
    const controller = new AbortController()
    vi.mocked(experimentApi.runs).mockImplementationOnce(async () => {
      controller.abort()
      return members(['first'], 20)
    })
    await expect(readExperimentMembers('exp-a', controller.signal)).rejects.toThrow()
    expect(experimentApi.runs).toHaveBeenCalledTimes(1)
  })

  it('bounds large membership scans without returning a partial scope', async () => {
    vi.mocked(experimentApi.runs).mockImplementation(async (_id, after = 0) =>
      members([`run-${after}`], after + 20),
    )
    await expect(readExperimentMembers('exp-a', new AbortController().signal)).rejects.toThrow(
      'not loaded completely',
    )
    expect(experimentApi.runs).toHaveBeenCalledTimes(100)
  })
})

describe('server verified experiment pairs', () => {
  it('requires an in-experiment compatible candidate, including a later options page', async () => {
    vi.mocked(benchApi.runOptions)
      .mockResolvedValueOnce(options(['outside'], 'page-2'))
      .mockResolvedValueOnce(options(['candidate']))
    const signal = new AbortController().signal
    const result = await verifyExperimentBaselines(
      [choice('baseline'), choice('outside-baseline')],
      new Set(['baseline', 'candidate']),
      signal,
    )
    expect([...result.ids]).toEqual(['baseline'])
    expect(benchApi.runOptions).toHaveBeenLastCalledWith(
      'comparison',
      'baseline',
      'page-2',
      expect.any(AbortSignal),
    )
    expect(benchApi.runOptions).toHaveBeenCalledTimes(2)
  })

  it('does not expose a baseline with only cross-experiment or unverified pairs', async () => {
    vi.mocked(benchApi.runOptions).mockResolvedValue({
      ...options(['outside']),
      scan_limited: true,
    })
    const result = await verifyExperimentBaselines(
      [choice('baseline')],
      new Set(['baseline', 'candidate']),
      new AbortController().signal,
    )
    expect(result.ids.size).toBe(0)
    expect(result.scanLimited).toBe(true)
  })

  it('rejects an inconsistent options cursor instead of assuming no scoped match', async () => {
    vi.mocked(benchApi.runOptions).mockResolvedValue(options(['outside'], 'same'))
    await expect(
      verifyExperimentBaselines(
        [choice('baseline')],
        new Set(['baseline', 'candidate']),
        new AbortController().signal,
      ),
    ).rejects.toThrow('inconsistent comparison page')
    expect(benchApi.runOptions).toHaveBeenCalledTimes(2)
  })

  it('bounds automatic pair verification rather than silently declaring no pairs', async () => {
    vi.mocked(benchApi.runOptions).mockImplementation(async (_kind, _baseline, after) =>
      options(['outside'], String(Number(after ?? 0) + 1)),
    )
    await expect(
      verifyExperimentBaselines(
        [choice('baseline')],
        new Set(['baseline']),
        new AbortController().signal,
      ),
    ).rejects.toThrow('No incomplete selection')
    expect(benchApi.runOptions).toHaveBeenCalledTimes(100)
  })

  it('requires every selected identity to be verified, with no duplicates or automatic choice', () => {
    const baselines = [choice('baseline')]
    const candidates = [choice('candidate')]
    expect(comparisonSelectionReady('baseline', ['candidate'], baselines, candidates)).toBe(true)
    expect(comparisonSelectionReady('baseline', [], baselines, candidates)).toBe(false)
    expect(comparisonSelectionReady('baseline', ['outside'], baselines, candidates)).toBe(false)
    expect(
      comparisonSelectionReady('baseline', ['candidate', 'candidate'], baselines, candidates),
    ).toBe(false)
    expect(comparisonSelectionReady('outside', ['candidate'], baselines, candidates)).toBe(false)
  })
})
