import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  benchmarkName,
  formatContextWindow,
  formatIntelligence,
  modelHubRows,
  modelHubStats,
} from './modelHubSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog

describe('model hub support', () => {
  it('projects the complete generated inventory without maintaining another list', () => {
    const stats = modelHubStats(catalog)
    expect(stats.models).toBe(catalog.models.length)
    expect(stats.physicalModels).toBeGreaterThan(100)
    expect(stats.virtualModels).toBeGreaterThan(0)
    expect(stats.providers).toBe(catalog.providers.length)
    expect(stats.publishers).toBeGreaterThan(15)
  })

  it('searches and filters canonical model metadata', () => {
    const rows = modelHubRows(catalog, {
      query: 'glm-5.3',
      kind: 'physical',
      distribution: 'open_weights',
      lifecycle: 'supported',
      publisher: 'Z.ai',
      sort: 'name',
    })
    expect(rows.map((row) => row.model.id)).toEqual([
      'zai/glm-5.3',
      'zai/glm-5.3-flash',
    ])
    expect(rows.every((row) => row.providers.length > 0)).toBe(true)
  })

  it('sorts available intelligence scores ahead of missing evidence', () => {
    const rows = modelHubRows(catalog, {
      query: '',
      kind: 'physical',
      distribution: 'all',
      lifecycle: 'supported',
      publisher: 'all',
      sort: 'intelligence',
    })
    expect(rows[0].intelligence?.status).toBe('available')
    const firstMissing = rows.findIndex((row) => row.intelligence?.status !== 'available')
    expect(rows.slice(firstMissing).every((row) => row.intelligence?.status !== 'available')).toBe(
      true,
    )
  })

  it('keeps removed models discoverable without presenting them as supported', () => {
    const supported = modelHubRows(catalog, {
      query: 'claude sonnet 4',
      kind: 'physical',
      distribution: 'all',
      lifecycle: 'supported',
      publisher: 'all',
      sort: 'name',
    })
    const historical = modelHubRows(catalog, {
      query: 'claude sonnet 4',
      kind: 'physical',
      distribution: 'all',
      lifecycle: 'removed',
      publisher: 'all',
      sort: 'name',
    })
    expect(supported.some((row) => row.model.id === 'anthropic/claude-sonnet-4')).toBe(false)
    expect(historical.map((row) => row.model.id)).toContain('anthropic/claude-sonnet-4')
  })

  it('formats context, missing scores, and benchmark labels explicitly', () => {
    expect(formatContextWindow(1_000_000)).toBe('1M')
    expect(formatContextWindow(131_072)).toBe('131.1K')
    expect(formatContextWindow()).toBe('Not published')
    expect(formatIntelligence(null)).toBe('Not yet measured')
    expect(benchmarkName('idavidrein/gpqa-diamond@1.0.0', 'accuracy', catalog)).toBe(
      'GPQA Diamond · accuracy',
    )
  })
})
