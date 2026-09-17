import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import HeaderDisplay from '../components/HeaderDisplay'
import { collectResponseHeaders } from '../components/chatRequestSupport'
import { DECISION_SIGNAL_TYPES, SIGNAL_TYPES } from '../generated/routerConfigContract'
import { monarchTokens } from '../lib/dslLanguage'
import { getSignalFieldSchema } from '../lib/dslSchemas'
import type { SafetySignal } from '../types/config'
import { SIGNAL_CATALOG } from './configPageSignalCatalog'
import { collectSignals } from './insightsPageSupport'
import { SIGNAL_ICONS } from './topology/constants'
import { parseConfigToTopology } from './topology/utils/topologyParser'

const signal: SafetySignal = {
  name: 'unsafe_content',
  model: 'content_safety',
  labels: ['safe', 'unsafe'],
  unsafe_labels: ['unsafe'],
  threshold: 0.8,
  hazard: {
    model: 'content_hazard',
    labels: ['violence', 'fraud', 'other'],
    categories: ['violence'],
    threshold: 0.7,
  },
}

describe('content safety signal Dashboard contract', () => {
  it('uses the generated declaration and decision inventories in catalog and DSL', () => {
    expect(SIGNAL_TYPES).toContain('safety')
    expect(DECISION_SIGNAL_TYPES).toContain('safety')
    expect(SIGNAL_CATALOG.find((entry) => entry.type === 'safety')).toMatchObject({
      label: 'Safety',
      collection: 'safety',
    })
    expect(monarchTokens.signalTypes).toContain('safety')
    const typeRule = monarchTokens.tokenizer.root.find(
      (rule) => Array.isArray(rule) && rule[1] === 'type',
    )
    const typePattern = Array.isArray(typeRule) ? typeRule[0] : undefined
    expect((typePattern as RegExp).test('safety')).toBe(true)
  })

  it('renders the generated nested hazard form and preserves it in canonical topology', () => {
    const fields = getSignalFieldSchema('safety')
    expect(fields.find((field) => field.key === 'model')?.required).not.toBe(true)
    const hazard = fields.find((field) => field.key === 'hazard')
    expect(hazard?.fields?.find((field) => field.key === 'model')?.required).not.toBe(true)
    expect(hazard?.fields?.map((field) => field.key)).toEqual(
      expect.arrayContaining(['model', 'labels', 'categories', 'threshold']),
    )
    const topology = parseConfigToTopology({
      routing: {
        signals: { safety: [signal] },
        decisions: [
          {
            name: 'safety_review',
            rules: { operator: 'AND', conditions: [{ type: 'safety', name: signal.name }] },
            modelRefs: [{ model: 'reviewer' }],
          },
        ],
      },
      providers: { models: [{ name: 'reviewer' }] },
    })
    const { name, ...signalFields } = signal
    expect(topology.signals).toEqual([
      expect.objectContaining({ type: 'safety', name, config: signalFields }),
    ])
    expect(topology.decisions[0].rules.conditions).toEqual([{ type: 'safety', name }])
    expect(SIGNAL_ICONS.safety).toBeTruthy()
  })

  it('preserves built-in model defaults without adding external model names', () => {
    const builtIn: SafetySignal = {
      name: 'built_in_safety',
      threshold: 0.8,
      hazard: { labels: ['violence', 'fraud'], categories: ['fraud'], threshold: 0.7 },
    }
    const topology = parseConfigToTopology({
      routing: { signals: { safety: [builtIn] }, decisions: [] },
      providers: { models: [] },
    })
    const { name, ...fields } = builtIn
    expect(topology.signals).toEqual([
      expect.objectContaining({ type: 'safety', name, config: fields }),
    ])
  })

  it('preserves and reveals safety matches from real response headers', () => {
    const headers = collectResponseHeaders(
      new Response(null, {
        headers: { 'x-vsr-matched-safety': signal.name, 'x-unrelated': 'hidden' },
      }),
    )
    expect(headers).toEqual({ 'x-vsr-matched-safety': signal.name })
    const markup = renderToStaticMarkup(createElement(HeaderDisplay, { headers }))
    expect(markup).toContain('Safety Signal')
    expect(markup).toContain('Unsafe Content')
    expect(markup).not.toContain('x-unrelated')
  })

  it('includes replay safety matches in the signal summary', () => {
    expect(collectSignals({ safety: [signal.name] })).toEqual(['Unsafe Content'])
  })
})
