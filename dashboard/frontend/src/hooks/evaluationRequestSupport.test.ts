import { readFileSync } from 'node:fs'
import { describe, expect, it, vi } from 'vitest'

import { createEvaluationRequest } from './evaluationRequestSupport'

describe('evaluation request controller', () => {
  it('skips hidden background requests but allows an explicit foreground load', async () => {
    const fetcher = vi.fn(async () => ['task-1'])
    const request = createEvaluationRequest(fetcher, { isHidden: () => true })

    await expect(request.run()).resolves.toBeUndefined()
    await expect(request.run({ allowHidden: true })).resolves.toEqual(['task-1'])
    expect(fetcher).toHaveBeenCalledTimes(1)
  })

  it('deduplicates overlapping requests', async () => {
    let resolveRequest: ((value: string[]) => void) | undefined
    const fetcher = vi.fn(
      () => new Promise<string[]>((resolve) => {
        resolveRequest = resolve
      }),
    )
    const request = createEvaluationRequest(fetcher, { isHidden: () => false })

    const first = request.run()
    const second = request.run()
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(request.isInFlight()).toBe(true)

    resolveRequest?.(['task-1'])
    await expect(first).resolves.toEqual(['task-1'])
    await expect(second).resolves.toEqual(['task-1'])
    expect(request.isInFlight()).toBe(false)
  })

  it('drops a stale response even when the underlying fetch ignores abort', async () => {
    let resolveRequest: ((value: string) => void) | undefined
    const request = createEvaluationRequest(
      () => new Promise<string>((resolve) => {
        resolveRequest = resolve
      }),
      { isHidden: () => false },
    )

    const stale = request.run()
    request.invalidate()
    resolveRequest?.('stale task')

    await expect(stale).resolves.toBeUndefined()
  })

  it('keeps evaluation polling and SSE visibility-aware in the hook integration', () => {
    const source = readFileSync(new URL('./useEvaluation.ts', import.meta.url), 'utf8')
    expect(source).toContain("document.addEventListener('visibilitychange'")
    expect(source).toContain('createEvaluationRequest')
    expect(source).toContain('terminalRef.current')
    expect(source).toContain('request.invalidate')
    expect(source).not.toMatch(/setInterval\(fetchTasks/)
  })
})
