import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_LIMITS, makeManifest } from './model'
import {
  clearSubmission,
  readSubmission,
  saveSubmission,
  submissionKey,
  type PendingSubmission,
} from './submission'

const manifest = makeManifest(
  'Frozen attempt',
  'live',
  'quick',
  {
    id: 'dataset',
    path: 'prepared.jsonl',
    sha256: 'a'.repeat(64),
    case_count: 1,
  },
  [{ id: 'single', kind: 'single', model: 'model', base_url: 'http://localhost:8000/v1' }],
  DEFAULT_LIMITS,
)
const request: PendingSubmission = {
  version: 1,
  actorID: 'first-account',
  idempotencyKey: 'attempt-a',
  manifest,
}

describe('durable submission identity', () => {
  beforeEach(() => {
    const values = new Map<string, string>()
    vi.stubGlobal('sessionStorage', {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
      removeItem: (key: string) => values.delete(key),
    })
  })
  afterEach(() => vi.unstubAllGlobals())

  it('does not clear or overwrite a newer pending identity', () => {
    const newer = { ...request, idempotencyKey: 'attempt-b' }
    saveSubmission(newer)
    expect(clearSubmission(request)).toBe(false)
    expect(() => saveSubmission(request)).toThrow('Another saved submission')
    expect(readSubmission(request.actorID).request).toEqual(newer)
    expect(clearSubmission(newer)).toBe(true)
  })

  it('keeps accounts separate and refuses to replace unreadable saved state', () => {
    saveSubmission(request)
    expect(clearSubmission({ ...request, actorID: 'second-account' })).toBe(false)
    expect(readSubmission(request.actorID).request).toEqual(request)
    sessionStorage.setItem(submissionKey(request.actorID), '{broken')
    expect(clearSubmission(request)).toBe(false)
    expect(() => saveSubmission(request)).toThrow('Another saved submission')
    expect(sessionStorage.getItem(submissionKey(request.actorID))).toBe('{broken')
  })
})
