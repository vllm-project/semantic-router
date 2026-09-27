import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import {
  clearExperiment,
  experimentKey,
  readExperiment,
  saveExperiment,
  type PendingExperiment,
} from './experimentSubmission'
const request: PendingExperiment = { version: 1, actorID: 'a', name: 'Recipe study', key: 'one' }
beforeEach(() => {
  const values = new Map<string, string>()
  vi.stubGlobal('sessionStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  })
})
afterEach(() => vi.unstubAllGlobals())
it('keeps a newer experiment identity and scopes requests to the current account', () => {
  const newer = { ...request, key: 'two' }
  saveExperiment(newer)
  expect(clearExperiment(request)).toBe(false)
  expect(() => saveExperiment(request)).toThrow('needs reconciliation')
  expect(readExperiment('b').saved).toBeNull()
  expect(readExperiment('a').saved).toEqual(newer)
})
it('does not overwrite unreadable identity or post without storage', () => {
  sessionStorage.setItem(experimentKey('a'), '{broken')
  expect(() => saveExperiment(request)).toThrow('No request was sent')
  expect(clearExperiment(request)).toBe(false)
  vi.stubGlobal('sessionStorage', {
    getItem: () => null,
    setItem: () => {
      throw new Error('Quota')
    },
  })
  expect(() => saveExperiment(request)).toThrow('No request was sent')
})
