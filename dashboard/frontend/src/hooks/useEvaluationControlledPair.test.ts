import { describe, expect, it, vi } from 'vitest'

import type { EvaluationControlledPairExecution } from '../types/evaluationControlledPair'
import { handoffEvaluationControlledPair } from './evaluationControlledPairHookSupport'

describe('controlled pair readiness handoff', () => {
  it('awaits assignment and returns a retryable rationale when durable-ledger refresh fails', async () => {
    const execution = {
      id: '11111111-1111-4111-8111-111111111111',
    } as EvaluationControlledPairExecution
    const onReady = vi
      .fn<(value: EvaluationControlledPairExecution) => Promise<void>>()
      .mockRejectedValue(new Error('Durable run ledger refresh failed.'))

    await expect(handoffEvaluationControlledPair(execution, onReady)).resolves.toBe(
      'Durable run ledger refresh failed.',
    )
    expect(onReady).toHaveBeenCalledOnce()
    expect(onReady).toHaveBeenCalledWith(execution, expect.any(Function))
  })

  it('does not report ready until asynchronous assignment resolves', async () => {
    const execution = {
      id: '22222222-2222-4222-8222-222222222222',
    } as EvaluationControlledPairExecution
    let resolveAssignment: (() => void) | undefined
    const assignment = new Promise<void>((resolve) => {
      resolveAssignment = resolve
    })
    let settled = false
    const handoff = handoffEvaluationControlledPair(execution, () => assignment).then((error) => {
      settled = true
      return error
    })

    await Promise.resolve()
    expect(settled).toBe(false)
    resolveAssignment?.()
    await expect(handoff).resolves.toBeNull()
  })

  it('lets asynchronous assignment reject an unmounted or superseded generation', async () => {
    const execution = {
      id: '33333333-3333-4333-8333-333333333333',
    } as EvaluationControlledPairExecution
    let current = true
    let resolveRefresh: (() => void) | undefined
    const refresh = new Promise<void>((resolve) => {
      resolveRefresh = resolve
    })
    const bind = vi.fn()

    const handoff = handoffEvaluationControlledPair(
      execution,
      async (value, isCurrent) => {
        await refresh
        if (isCurrent()) bind(value.id)
      },
      () => current,
    )
    current = false
    resolveRefresh?.()

    await expect(handoff).resolves.toBeNull()
    expect(bind).not.toHaveBeenCalled()
  })

  it('does not invoke assignment for an already stale generation', async () => {
    const execution = {
      id: '44444444-4444-4444-8444-444444444444',
    } as EvaluationControlledPairExecution
    const onReady = vi.fn()

    await expect(
      handoffEvaluationControlledPair(execution, onReady, () => false),
    ).resolves.toBeNull()
    expect(onReady).not.toHaveBeenCalled()
  })
})
