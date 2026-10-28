import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  clearReplay,
  readReplay,
  replayKey,
  saveReplay,
  type PendingReplay,
} from './replaySubmission'
const request: PendingReplay = {
  version: 1,
  actorID: 'account-a',
  request: { baseline_run_id: 'baseline', preview_run_id: 'preview', idempotency_key: 'attempt-a' },
}
describe('offline estimate submission identity', () => {
  beforeEach(() => {
    const values = new Map<string, string>()
    vi.stubGlobal('sessionStorage', {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
      removeItem: (key: string) => values.delete(key),
    })
  })
  afterEach(() => vi.unstubAllGlobals())
  it('preserves a newer request when an older response arrives and isolates accounts', () => {
    const newer = { ...request, request: { ...request.request, idempotency_key: 'attempt-b' } }
    saveReplay(newer)
    expect(clearReplay(request)).toBe(false)
    expect(() => saveReplay(request)).toThrow('needs reconciliation')
    expect(readReplay('account-b').saved).toBeNull()
    expect(clearReplay({ ...newer, actorID: 'account-b' })).toBe(false)
    expect(readReplay('account-a').saved).toEqual(newer)
  })
  it('fails closed on corrupted stored requests', () => {
    sessionStorage.setItem(replayKey('account-a'), '{broken')
    expect(readReplay('account-a').error).toContain('cannot be read safely')
    expect(() => saveReplay(request)).toThrow('needs reconciliation')
    expect(clearReplay(request)).toBe(false)
  })
  it('does not permit posting without durable storage', () => {
    vi.stubGlobal('sessionStorage', {
      getItem: () => null,
      setItem: () => {
        throw new Error('Quota')
      },
    })
    expect(() => saveReplay(request)).toThrow('No request was sent')
  })
})
