import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import HeaderDisplay from './HeaderDisplay'

const routingHeaders = {
  'x-vsr-schema-version': '2',
  'x-vsr-response-path': 'upstream',
  'x-vsr-selected-model': 'qwen/qwen3.5-rocm',
  'x-vsr-selected-decision': 'complex-specialist',
  'x-vsr-selected-algorithm': 'router_dc',
  'x-vsr-matched-domains': 'computer science',
}

describe('chat routing metadata', () => {
  it('keeps the primary route compact and puts supporting metadata behind details', () => {
    const markup = renderToStaticMarkup(createElement(HeaderDisplay, { headers: routingHeaders }))

    expect(markup).not.toContain('Schema Version')
    expect(markup.indexOf('Decision')).toBeLessThan(markup.indexOf('Algorithm'))
    expect(markup.indexOf('Algorithm')).toBeLessThan(markup.indexOf('Model'))
    expect(markup).toContain('aria-label="Show response details"')
    expect(markup.indexOf('Model')).toBeLessThan(markup.indexOf('Domain'))
    expect(markup.indexOf('Domain')).toBeLessThan(markup.indexOf('Response Path'))
  })

  it('uses the looper algorithm when the primary header is empty', () => {
    const headers = {
      ...routingHeaders,
      'x-vsr-selected-algorithm': '',
      'x-vsr-looper-algorithm': 'confidence',
    }
    const markup = renderToStaticMarkup(createElement(HeaderDisplay, { headers }))

    expect(markup).toContain('Confidence')
  })

  it('renders internal decision and signal identifiers as human-friendly labels', () => {
    const headers = {
      ...routingHeaders,
      'x-vsr-selected-decision': 'unified_frontier_verified_answer',
      'x-vsr-matched-embeddings': 'unified_frontier_workflow_intent',
      'x-vsr-matched-fact-check': 'needs_fact_check',
      'x-vsr-matched-complexity': 'unified_frontier_complexity:medium',
    }
    const markup = renderToStaticMarkup(createElement(HeaderDisplay, { headers }))

    expect(markup).toContain('Frontier Verified Answer')
    expect(markup).toContain('Frontier Workflow Intent')
    expect(markup).toContain('Needs Fact Check')
    expect(markup).toContain('Frontier Complexity: Medium')
    expect(markup).not.toContain('unified_frontier_')
  })
})

describe('looper latency and token usage headers (#2694)', () => {
  const looperMetricsHeaders = {
    ...routingHeaders,
    'x-vsr-looper-latency-ms': '842',
    'x-vsr-looper-prompt-tokens': '512',
    'x-vsr-looper-completion-tokens': '256',
    'x-vsr-looper-total-tokens': '768',
  }

  it('renders looper latency and token usage in HeaderDisplay', () => {
    const markup = renderToStaticMarkup(
      createElement(HeaderDisplay, { headers: looperMetricsHeaders }),
    )

    expect(markup).toContain('842')
    expect(markup).toContain('512')
    expect(markup).toContain('256')
    expect(markup).toContain('768')
  })
})
