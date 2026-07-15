import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import HeaderDisplay from './HeaderDisplay'
import HeaderReveal from './HeaderReveal'

const routingHeaders = {
  'x-vsr-schema-version': '2',
  'x-vsr-response-path': 'upstream',
  'x-vsr-selected-model': 'qwen/qwen3.5-rocm',
  'x-vsr-selected-decision': 'complex-specialist',
  'x-vsr-selected-algorithm': 'router_dc',
  'x-vsr-matched-domains': 'computer science',
}

describe('chat routing metadata', () => {
  it('hides internal schema metadata and orders the routed decision path', () => {
    const markup = renderToStaticMarkup(createElement(HeaderDisplay, { headers: routingHeaders }))

    expect(markup).not.toContain('Schema Version')
    expect(markup.indexOf('Domain')).toBeLessThan(markup.indexOf('Decision'))
    expect(markup.indexOf('Decision')).toBeLessThan(markup.indexOf('Algorithm'))
    expect(markup.indexOf('Algorithm')).toBeLessThan(markup.indexOf('Model'))
    expect(markup.indexOf('Model')).toBeLessThan(markup.indexOf('Response Path'))
  })

  it('renders the transient reveal as a signal-to-response decision path', () => {
    const markup = renderToStaticMarkup(createElement(HeaderReveal, { headers: routingHeaders }))

    expect(markup).not.toContain('Schema Version')
    expect(markup).toContain('DECISION PATH')
    expect(markup).toContain('SIGNAL')
    expect(markup).toContain('DECISION')
    expect(markup).toContain('ALGORITHM')
    expect(markup).toContain('MODEL')
    expect(markup).toContain('RESPONSE PATH')
  })

  it('uses the looper algorithm as a compatibility fallback', () => {
    const headers = {
      ...routingHeaders,
      'x-vsr-selected-algorithm': '',
      'x-vsr-looper-algorithm': 'confidence',
    }
    const markup = renderToStaticMarkup(createElement(HeaderReveal, { headers }))

    expect(markup).toContain('confidence')
  })
})
