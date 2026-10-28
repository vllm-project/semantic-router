import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import type { Signal } from './insightsPageTypes'
import { buildSignalFields, collectSignals } from './insightsRecordSignals'

const renderFields = (signals: Signal) =>
  renderToStaticMarkup(<>{buildSignalFields(signals).map((field) => field.value)}</>)

describe('recorded signal matches', () => {
  it('retains every canonical family in registry order without mutating recorded arrays', () => {
    const signals = Object.freeze(
      Object.fromEntries(
        [...ROUTER_CONFIG_EXTENSION.signals]
          .reverse()
          .map(({ type }) => [type, Object.freeze([`${type}:primary:variant`])]),
      ),
    ) as Signal
    const before = JSON.stringify(signals)
    const fields = buildSignalFields(signals)

    expect(fields.slice(1).map(({ label }) => label)).toEqual(
      ROUTER_CONFIG_EXTENSION.signals.map(({ display_name }) => `${display_name} signals`),
    )
    expect(collectSignals(signals)).toHaveLength(ROUTER_CONFIG_EXTENSION.signals.length)
    const html = renderFields(signals)
    for (const { type } of ROUTER_CONFIG_EXTENSION.signals) {
      expect(html).toContain(`title="${type}:primary:variant"`)
    }
    expect(JSON.stringify(signals)).toBe(before)
  })

  it('includes unknown recorded string arrays after known families and ignores non-array data', () => {
    const signals = {
      zeta_future: ['zeta_rule'],
      domain: ['technical'],
      alpha_future: ['custom_rule:main:exact_suffix', 'custom_rule:main:exact_suffix', 7, null],
      not_a_family: { value: 1 },
      invalid_family: 'not_an_array',
      empty_family: [],
    } as unknown as Signal
    const fields = buildSignalFields(signals)

    expect(fields.slice(1).map(({ label }) => label)).toEqual([
      'Domain signals',
      'Alpha Future signals',
      'Zeta Future signals',
    ])
    expect(collectSignals(signals)).toEqual([
      'Technical',
      'Custom Rule: Main',
      'Custom Rule: Main',
      'Zeta Rule',
    ])
    expect(renderFields(signals).match(/title="custom_rule:main:exact_suffix"/g)).toHaveLength(2)
  })

  it('explains the recorded-match boundary even when no matches were recorded', () => {
    const html = renderFields({ domain: [], complexity: [] })
    expect(collectSignals({})).toEqual([])
    expect(buildSignalFields({})).toHaveLength(1)
    expect(html).toContain('Recorded matches only. Measurements are available in Routing Metadata;')
    expect(html).toContain('unmatched or unrecorded rules are not listed.')
    expect(html).toContain('No signal matches were recorded for this request.')
    expect(html).not.toContain('Disabled')
    expect(html).not.toContain('Not evaluated')
  })

  it('escapes unknown labels and original identifiers as text', () => {
    const signals = { 'future<family>': ['rule<variant>'] } as unknown as Signal
    const html = renderFields(signals)
    expect(html).toContain('title="rule&lt;variant&gt;"')
    expect(html).not.toContain('<variant>')
  })
})
