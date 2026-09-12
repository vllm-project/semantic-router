import { afterEach, describe, expect, it, vi } from 'vitest'
import { waitForKnowledgeBaseActivation } from './knowledgeBaseActivation'

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals() })

describe('knowledge base activation', () => {
  it('waits for the exact candidate rather than a different active generation', async () => {
    vi.useFakeTimers()
    const fetcher = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ active_runtime_hash: 'old' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ active_runtime_hash: 'candidate' }) })
    vi.stubGlobal('fetch', fetcher)
    const done = waitForKnowledgeBaseActivation({ activation_status: 'pending', generated_runtime_hash: 'candidate' })
    await vi.advanceTimersByTimeAsync(1000)
    await done
    expect(fetcher).toHaveBeenCalledTimes(2)
  })

  it('does not claim activation if the candidate remains pending', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ active_runtime_hash: 'old' }) }))
    const result = expect(waitForKnowledgeBaseActivation({ activation_status: 'pending', generated_runtime_hash: 'candidate' }))
      .rejects.toThrow('previous configuration remains active')
    await vi.advanceTimersByTimeAsync(20000)
    await result
  })

  it('keeps standalone and already active responses immediate', async () => {
    const fetcher = vi.fn()
    vi.stubGlobal('fetch', fetcher)
    await waitForKnowledgeBaseActivation({ activation_status: 'unknown' })
    await waitForKnowledgeBaseActivation({ activation_status: 'active' })
    expect(fetcher).not.toHaveBeenCalled()
  })
})
