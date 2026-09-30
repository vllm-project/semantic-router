import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import {
  modelHubArenaData,
  parseModelHubArenaRoute,
  serializeModelHubArenaRoute,
} from './modelHubArenaSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog

describe('model hub arena hierarchy', () => {
  it('derives all three ranking layers from the default catalog index', () => {
    const arena = modelHubArenaData(catalog, 'all')

    expect(arena).not.toBeNull()
    expect(arena?.overall.id).toBe('vllm-sr/intelligence@1.0.0')
    expect(arena?.overall.rows).toHaveLength(21)
    expect(arena?.capabilities.map((surface) => surface.displayName)).toEqual([
      'General',
      'Reasoning',
      'Coding',
      'Agentic',
    ])
    expect(arena?.benchmarks.map((surface) => surface.displayName)).toEqual([
      'MMLU-Pro',
      'GPQA Diamond',
      "Humanity's Last Exam",
      'LiveCodeBench v6',
      'SciCode',
      'Terminal-Bench 2.1',
    ])
  })

  it('keeps category evidence useful when Overall is incomplete', () => {
    const arena = modelHubArenaData(catalog, 'all')
    const coding = arena?.capabilities.find((surface) => surface.displayName === 'Coding')

    expect(arena?.overall.rows.some((row) => row.model.id === 'qwen/qwen3.8-27b')).toBe(false)
    expect(coding?.rows.some((row) => row.model.id === 'qwen/qwen3.8-27b')).toBe(true)
  })
})

describe('model hub arena URL state', () => {
  it('round-trips layer, selection, and scope without dropping unrelated state', () => {
    const current = new URLSearchParams('campaign=pytorchcon')
    const parameters = serializeModelHubArenaRoute(
      {
        scope: 'virtual',
        layer: 'benchmarks',
        capability: 'vllm-sr/coding@1.0.0',
        benchmark: 'livecodebench/livecodebench@6.0.0',
      },
      current,
    )

    expect(parseModelHubArenaRoute(parameters)).toEqual({
      scope: 'virtual',
      layer: 'benchmarks',
      capability: 'vllm-sr/coding@1.0.0',
      benchmark: 'livecodebench/livecodebench@6.0.0',
    })
    expect(parameters.get('campaign')).toBe('pytorchcon')
  })

  it('falls back safely and omits default values', () => {
    expect(
      parseModelHubArenaRoute(new URLSearchParams('arena=closed&arena_layer=domains')),
    ).toEqual({
      scope: 'all',
      layer: 'overall',
      capability: '',
      benchmark: '',
    })
    expect(
      serializeModelHubArenaRoute(
        {
          scope: 'all',
          layer: 'overall',
          capability: '',
          benchmark: '',
        },
        new URLSearchParams(),
      ).toString(),
    ).toBe('')
  })
})
