import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { benchApi } from './api'
import type { Manifest } from './types'

describe('sr-bench evidence read deadlines', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('bounds a stalled read and does not retry the request', async () => {
    const fetch = vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          )
        }),
    )
    vi.stubGlobal('fetch', fetch)
    const outcome = expect(benchApi.runs()).rejects.toMatchObject({
      status: 408,
      message: expect.stringContaining('timed out after 30 seconds'),
    })
    await vi.advanceTimersByTimeAsync(30000)
    await outcome
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each(['compose', 'plan', 'comparison', 'replay', 'candidate'])(
    'bounds a stalled %s review without retrying or dispatching generation',
    async (operation) => {
      const fetch = vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () =>
              reject(new DOMException('Aborted', 'AbortError')),
            )
          }),
      )
      vi.stubGlobal('fetch', fetch)
      const outcome = expect(
        operation === 'candidate'
          ? benchApi.candidatePlan('baseline', {
              target_ids: ['balance'],
              mode: 'live',
              name: 'Candidate',
            })
          : operation === 'replay'
            ? benchApi.replay({
                baseline_run_id: 'baseline',
                preview_run_id: 'preview',
                idempotency_key: 'same-key',
              })
            : operation === 'compose'
              ? benchApi.composeDatasets(['a'.repeat(64)], ['mmlu-pro'])
              : operation === 'replay'
                ? 'Reconcile the saved submission'
                : operation === 'comparison'
                  ? benchApi.compare('baseline', 'candidate')
                  : benchApi.plan({} as Manifest),
      ).rejects.toMatchObject({
        status: 408,
        message: expect.stringContaining(
          operation === 'replay'
            ? 'Reconcile the saved submission'
            : operation === 'comparison'
              ? 'timed out after 30 seconds'
              : 'No model generation was requested',
        ),
      })
      await vi.advanceTimersByTimeAsync(30000)
      await outcome
      expect(fetch).toHaveBeenCalledTimes(1)
      expect(fetch.mock.calls[0][0]).not.toBe('/api/sr-bench/v1/runs')
      expect(vi.getTimerCount()).toBe(0)
    },
  )

  it('preserves caller cancellation and clears the read deadline', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () =>
              reject(new DOMException('Aborted', 'AbortError')),
            )
          }),
      ),
    )
    const controller = new AbortController()
    const outcome = expect(benchApi.runs(controller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    })
    controller.abort()
    await outcome
    expect(vi.getTimerCount()).toBe(0)
  })

  it('bounds mutation responses without retrying or claiming the mutation was stopped', async () => {
    const fetch = vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) =>
          init.signal?.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          ),
        ),
    )
    vi.stubGlobal('fetch', fetch)
    const outcome = expect(benchApi.cancel('run-existing')).rejects.toMatchObject({
      status: 408,
      dispatchStarted: undefined,
      message: expect.stringContaining('outcome is unknown'),
    })
    await vi.advanceTimersByTimeAsync(30000)
    await outcome
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(vi.getTimerCount()).toBe(0)
  })

  it('marks a lost initial submission response ambiguous without claiming dispatch was stopped', async () => {
    const fetch = vi.fn(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          )
        }),
    )
    vi.stubGlobal('fetch', fetch)
    const outcome = expect(
      benchApi.start({} as Manifest, 'same-submission-id'),
    ).rejects.toMatchObject({
      status: 408,
      dispatchStarted: undefined,
      message: expect.stringContaining('Reconcile the saved submission'),
    })
    await vi.advanceTimersByTimeAsync(30000)
    await outcome
    expect(fetch).toHaveBeenCalledTimes(1)
  })
})
