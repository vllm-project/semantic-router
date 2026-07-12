import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import {
  getDecisionEntriesError,
  getMlpHiddenLayersError,
  parseMlpHiddenLayers,
  serializeMlpHiddenLayers,
} from './mlSetupStructuredFieldSupport'

describe('ML setup structured fields', () => {
  it('round-trips hidden-layer editors through the existing comma-delimited API field', () => {
    const layers = parseMlpHiddenLayers('256,128,64')
    expect(layers).toEqual([{ size: 256 }, { size: 128 }, { size: 64 }])
    expect(serializeMlpHiddenLayers(layers)).toBe('256,128,64')
    expect(getMlpHiddenLayersError('256,256')).toBeNull()
  })

  it('rejects invalid hidden-layer and decision schemas before submission', () => {
    expect(getMlpHiddenLayersError('256,0')).toMatch(/positive integer/i)
    expect(getMlpHiddenLayersError('')).toMatch(/at least one/i)
    expect(
      getDecisionEntriesError([
        { name: 'default', domains: ['general'], algorithm: 'knn', priority: 100, model_names: [] },
      ]),
    ).toMatch(/model reference/i)
    expect(
      getDecisionEntriesError([
        {
          name: 'default',
          domains: ['general', 'general'],
          algorithm: 'knn',
          priority: 100,
          model_names: ['local/model'],
        },
      ]),
    ).toMatch(/duplicated/i)
  })

  it('accepts valid list-shaped decision payloads unchanged', () => {
    expect(
      getDecisionEntriesError([
        {
          name: 'coding-route',
          domains: ['coding', 'computer-science'],
          algorithm: 'mlp',
          priority: 100,
          model_names: ['local/code', 'local/general'],
        },
      ]),
    ).toBeNull()
  })

  it('keeps ML list fields connected to structured controls', () => {
    const source = readFileSync(new URL('./MLSetupPage.tsx', import.meta.url), 'utf8')
    expect(source).toContain('<StringListEditor')
    expect(source).toContain('<ObjectListEditor')
    expect(source).toContain('getDecisionEntriesError(wizard.decisions)')
    expect(source).toContain('getMlpHiddenLayersError(wizard.mlpHiddenSizes)')
    expect(source).not.toContain('comma-separated')
  })
})
