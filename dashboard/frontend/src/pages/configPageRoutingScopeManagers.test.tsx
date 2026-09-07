import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ConfigPageDecisionsSection from './ConfigPageDecisionsSection'
import ConfigPageProjectionsSection from './ConfigPageProjectionsSection'
import ConfigPageSignalsSection from './ConfigPageSignalsSection'
import type { ConfigData } from './configPageSupport'

const config: ConfigData = {
  providers: { models: [] },
  routing: { decisions: [] },
  entrypoints: [
    { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
    { model_names: ['vllm-sr/private'], recipe: 'privacy' },
  ],
  recipes: [
    {
      name: 'balanced',
      routing: {
        projections: {
          scores: [{ name: 'balanced-score', method: 'weighted_sum', inputs: [] }],
          mappings: [],
          partitions: [],
        },
        signals: {
          keywords: [
            {
              name: 'balanced-keyword',
              operator: 'OR',
              keywords: ['balanced'],
              case_sensitive: false,
            },
          ],
        },
        decisions: [
          {
            name: 'balanced-route',
            description: 'Balanced route',
            priority: 100,
            rules: {
              operator: 'AND',
              conditions: [{ type: 'keyword', name: 'balanced-keyword' }],
            },
            modelRefs: [{ model: 'model-a', use_reasoning: false }],
          },
        ],
      },
    },
    {
      name: 'privacy',
      routing: {
        projections: {
          scores: [{ name: 'private-score', method: 'weighted_sum', inputs: [] }],
          mappings: [],
          partitions: [],
        },
        signals: {
          pii: [{ name: 'private-pii', threshold: 0.8 }],
        },
        decisions: [
          {
            name: 'private-route',
            description: 'Private route',
            priority: 100,
            rules: {
              operator: 'AND',
              conditions: [{ type: 'pii', name: 'private-pii' }],
            },
            modelRefs: [{ model: 'model-a', use_reasoning: false }],
          },
        ],
      },
    },
  ],
}

describe('recipe-aware config managers', () => {
  it('shows the first reachable recipe signals instead of an empty default profile', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfigPageSignalsSection, {
        config,
        isPythonCLI: true,
        isReadonly: false,
        signalsSearch: '',
        onSignalsSearchChange: vi.fn(),
        saveConfig: vi.fn(),
        openEditModal: vi.fn(),
        openViewModal: vi.fn(),
        listInputToArray: vi.fn(),
        removeSignalByName: vi.fn(),
      }),
    )

    expect(markup).toContain('vllm-sr/balanced')
    expect(markup).toContain('balanced-keyword')
    expect(markup).not.toContain('private-pii')
  })

  it('shows recipe-owned decisions with an explicit routing profile selector', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfigPageDecisionsSection, {
        config,
        isPythonCLI: true,
        isReadonly: false,
        decisionsSearch: '',
        onDecisionsSearchChange: vi.fn(),
        saveConfig: vi.fn(),
        openEditModal: vi.fn(),
        openViewModal: vi.fn(),
        removeDecisionByName: vi.fn(),
        models: [],
      }),
    )

    expect(markup).toContain('Routing profile')
    expect(markup).toContain('vllm-sr/balanced')
    expect(markup).toContain('balanced-route')
    expect(markup).not.toContain('private-route')
  })

  it('shows recipe-owned projections through the same routing profile selector', () => {
    const markup = renderToStaticMarkup(
      createElement(ConfigPageProjectionsSection, {
        config,
        isReadonly: false,
        saveConfig: vi.fn(),
        openEditModal: vi.fn(),
        openViewModal: vi.fn(),
      }),
    )

    expect(markup).toContain('Routing profile')
    expect(markup).toContain('vllm-sr/balanced')
    expect(markup).toContain('balanced-score')
    expect(markup).not.toContain('private-score')
  })
})
