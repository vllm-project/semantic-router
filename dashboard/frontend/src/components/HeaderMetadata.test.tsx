import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import HeaderDisplay from './HeaderDisplay'
import { expandRoutingHeaders } from './vsrTrace'

const publicHeaders = {
  'x-vsr-schema-version': '3',
  'x-vsr-response-path': 'upstream',
  'x-vsr-selected-model': 'qwen/qwen3.5-rocm',
  'x-vsr-selected-decision': 'complex-specialist',
}

const routingTrace = 'algorithm=systemone, signals=("domain:computer science")'

const render = (trace: string, extra: Record<string, string> = {}) =>
  renderToStaticMarkup(
    createElement(HeaderDisplay, {
      headers: expandRoutingHeaders({ ...publicHeaders, ...extra, 'x-vsr-trace': trace }),
    }),
  )

describe('chat routing metadata', () => {
  it('labels native dispatch limits separately from consumed token usage', () => {
    const markup = render(
      `${routingTrace}, effective_input_tokens=512, effective_max_output_tokens=261632`,
    )

    expect(markup).toContain('Rendered input tokens')
    expect(markup).toContain('Output token limit')
    expect(markup).toContain('261632')
    expect(markup).not.toContain('Consumed tokens')
  })

  it('keeps the primary route compact and puts supporting metadata behind details', () => {
    const markup = render(routingTrace)

    expect(markup).not.toContain('Schema Version')
    expect(markup.indexOf('Decision')).toBeLessThan(markup.indexOf('Algorithm'))
    expect(markup.indexOf('Algorithm')).toBeLessThan(markup.indexOf('Model'))
    expect(markup).toContain('aria-label="Show response details"')
    expect(markup.indexOf('Model')).toBeLessThan(markup.indexOf('Domain'))
    expect(markup.indexOf('Domain')).toBeLessThan(markup.indexOf('Response Path'))
  })

  it('uses the looper algorithm when the trace names no selection algorithm', () => {
    const markup = render('looper=("a" "b");algorithm=confidence;iterations=2')

    expect(markup).toContain('Confidence')
  })

  it('renders internal decision and signal identifiers as human-friendly labels', () => {
    const markup = render(
      'algorithm=systemone, signals=("domain:unified_frontier_workflow_intent" "fact_check:needs_fact_check" "complexity:unified_frontier_complexity:medium")',
      { 'x-vsr-selected-decision': 'unified_frontier_verified_answer' },
    )

    expect(markup).toContain('Frontier Verified Answer')
    expect(markup).toContain('Frontier Workflow Intent')
    expect(markup).toContain('Needs Fact Check')
    expect(markup).toContain('Frontier Complexity: Medium')
    expect(markup).not.toContain('unified_frontier_')
  })

  it('shows the warnings and a truncated trace', () => {
    const markup = render(`${routingTrace}, truncated`, {
      'x-vsr-warnings': 'hallucination, protocol;action=dropped;reason="x";field="y"',
    })

    expect(markup).toContain('Warnings')
    expect(markup).toContain('hallucination, protocol')
    expect(markup).toContain('Trace Truncated')
  })

  it('shows the output kinds of the selected decision', () => {
    const markup = render(`${routingTrace}, modality=(text image)`)

    expect(markup).toContain('Output')
    expect(markup).toContain('text, image')
  })

  it('shows a matched metadata signal', () => {
    const markup = render('algorithm=static, signals=("metadata:enterprise_tenant")')

    expect(markup).toContain('Metadata Signal')
    expect(markup).toContain('Enterprise Tenant')
  })
})

describe('System One routing details', () => {
  it('labels the decision model answers, recipe, confidence and routing latency', () => {
    const markup = render(
      'confidence=0.91, algorithm=systemone, routing_latency_ms=61.0, signals=("systemone:task:stem" "systemone:difficulty" "systemone:needs:deliberation")',
      { 'x-vsr-selected-recipe': 'default' },
    )

    expect(markup).toContain('System One Answers')
    expect(markup).toContain('Recipe')
    expect(markup).toContain('Decision Confidence')
    expect(markup).toContain('Routing Latency')
    expect(markup).toContain('61.0')
  })
})

describe('looper latency and token usage (#2694)', () => {
  it('renders looper latency and token usage in HeaderDisplay', () => {
    const markup = render(
      'algorithm=escalate, looper=("a" "b");algorithm=escalate;model="b";iterations=2;latency_ms=842;prompt_tokens=512;completion_tokens=256;total_tokens=768',
    )

    expect(markup).toContain('842')
    expect(markup).toContain('512')
    expect(markup).toContain('256')
    expect(markup).toContain('768')
  })
})
