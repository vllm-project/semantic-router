import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  clearRecovery,
  readRecovery,
  recoveryKey,
  saveRecovery,
  type PendingRecovery,
} from './recoverySubmission'

const saved: PendingRecovery = {
  version: 1,
  actorID: 'actor-a',
  parentID: 'parent-a',
  request: {
    mode: 'undispatched',
    plan_sha256: 'a'.repeat(64),
    idempotency_key: 'attempt-a',
    cells: [{ case_id: 'case-a', target_id: 'single' }],
  },
}

describe('recovery submission scope', () => {
  beforeEach(() => {
    const values = new Map<string, string>()
    vi.stubGlobal('sessionStorage', {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
      removeItem: (key: string) => values.delete(key),
    })
  })
  afterEach(() => vi.unstubAllGlobals())

  it('isolates accounts and parents and never clears or replaces another attempt', () => {
    saveRecovery(saved)
    expect(readRecovery('actor-b', saved.parentID).saved).toBeNull()
    expect(readRecovery(saved.actorID, 'parent-b').saved).toBeNull()
    const other = { ...saved, request: { ...saved.request, idempotency_key: 'attempt-b' } }
    expect(clearRecovery(other)).toBe(false)
    expect(() => saveRecovery(other)).toThrow('Another recovery')
    expect(readRecovery(saved.actorID, saved.parentID).saved).toEqual(saved)
    expect(clearRecovery(saved)).toBe(true)
  })

  it('keeps corrupted or invalid saved recovery fail-closed', () => {
    const key = recoveryKey(saved.actorID, saved.parentID)
    for (const raw of [
      '{broken',
      'null',
      JSON.stringify({ ...saved, actorID: 'other' }),
      JSON.stringify({ ...saved, request: { ...saved.request, mode: 'failed' } }),
    ]) {
      sessionStorage.setItem(key, raw)
      expect(readRecovery(saved.actorID, saved.parentID).error).not.toBe('')
      expect(clearRecovery(saved)).toBe(false)
      expect(() => saveRecovery(saved)).toThrow('Another recovery')
      expect(sessionStorage.getItem(key)).toBe(raw)
    }
  })
})
