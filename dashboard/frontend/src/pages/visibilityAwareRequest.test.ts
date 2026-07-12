import { describe, expect, it, vi } from 'vitest'

import { createVisibilityAwareRequest } from './visibilityAwareRequest'

describe('createVisibilityAwareRequest', () => {
  it('skips hidden-page work unless the caller explicitly allows it', async () => {
    const task = vi.fn(async () => undefined)
    const request = createVisibilityAwareRequest(task, { isHidden: () => true })

    await request.run()
    expect(task).not.toHaveBeenCalled()

    await request.run({ allowHidden: true })
    expect(task).toHaveBeenCalledTimes(1)
  })

  it('coalesces overlapping calls and permits a later refresh', async () => {
    let finishRequest: (() => void) | undefined
    const task = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finishRequest = resolve
        }),
    )
    const request = createVisibilityAwareRequest(task, { isHidden: () => false })

    const first = request.run()
    const second = request.run()
    await Promise.resolve()

    expect(second).toBe(first)
    expect(task).toHaveBeenCalledTimes(1)

    finishRequest?.()
    await first

    const third = request.run()
    await Promise.resolve()
    finishRequest?.()
    await third

    expect(task).toHaveBeenCalledTimes(2)
  })
})
