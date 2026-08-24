import { readFileSync } from 'node:fs'
import { describe, expect, it, vi } from 'vitest'

import type { ModelLiveVerification } from '../utils/modelVerificationApi'
import {
  resetModelLiveVerification,
  runModelLiveVerification,
  type ModelLiveVerificationState,
} from './useModelLiveVerification'

const evidence = (summary: string): ModelLiveVerification => ({
  verified: true,
  model: 'logical-model',
  providerModel: 'provider/real-model',
  backend: 'logical-model/primary',
  provider: 'openai',
  summary,
  verifiedAt: '2026-08-15T05:06:07Z',
  latencyMs: 18,
})

function stateHarness(initial = new Map<string, ModelLiveVerificationState>()) {
  let states = initial
  return {
    get states() {
      return states
    },
    update(
      update: (
        current: Map<string, ModelLiveVerificationState>,
      ) => Map<string, ModelLiveVerificationState>,
    ) {
      states = update(states)
    },
  }
}

describe('live model verification request state', () => {
  it('aborts the previous request and ignores its stale success', async () => {
    const activeRequests = new Map<string, AbortController>()
    const state = stateHarness()
    const pending: Array<{
      signal: AbortSignal
      resolve: (value: ModelLiveVerification) => void
    }> = []
    const verifier = vi.fn(
      (_model: string, signal?: AbortSignal) =>
        new Promise<ModelLiveVerification>((resolve) => {
          if (!signal) throw new Error('missing abort signal')
          pending.push({ signal, resolve })
        }),
    )

    const first = runModelLiveVerification('logical-model', activeRequests, state.update, verifier)
    expect(state.states.get('logical-model')).toEqual({ status: 'pending' })

    const second = runModelLiveVerification('logical-model', activeRequests, state.update, verifier)
    expect(pending[0].signal.aborted).toBe(true)

    pending[0].resolve(evidence('stale response'))
    await first
    expect(state.states.get('logical-model')).toEqual({ status: 'pending' })

    pending[1].resolve(evidence('current response'))
    await second
    expect(state.states.get('logical-model')).toMatchObject({
      status: 'verified',
      evidence: { summary: 'current response' },
    })
  })

  it('aborts in-flight work and clears evidence when the config revision changes', () => {
    const first = new AbortController()
    const second = new AbortController()
    const activeRequests = new Map([
      ['model-a', first],
      ['model-b', second],
    ])
    const state = stateHarness(
      new Map([['model-a', { status: 'verified', evidence: evidence('old config') }]]),
    )

    resetModelLiveVerification(activeRequests, state.update)

    expect(first.signal.aborted).toBe(true)
    expect(second.signal.aborted).toBe(true)
    expect(activeRequests.size).toBe(0)
    expect(state.states.size).toBe(0)

    const source = readFileSync(new URL('./useModelLiveVerification.ts', import.meta.url), 'utf8')
    expect(source).toContain('useEffect(() => abortAndClear(), [abortAndClear, configRevision])')
  })
})
