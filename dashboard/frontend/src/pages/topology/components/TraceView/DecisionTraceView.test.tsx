import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { DecisionTraceView } from './DecisionTraceView'
import type { DecisionTrace } from '../../types'

const nestedTrace: DecisionTrace[] = [
  {
    decision_name: 'coding_decision',
    matched: true,
    confidence: 0.9,
    root_trace: {
      node_type: 'OR',
      matched: true,
      confidence: 0.9,
      children: [
        {
          node_type: 'leaf',
          signal_type: 'keyword',
          signal_name: 'coding',
          matched: true,
          confidence: 1,
        },
        {
          node_type: 'leaf',
          signal_type: 'domain',
          signal_name: 'code',
          matched: false,
          confidence: 0,
        },
      ],
    },
  },
  {
    decision_name: 'reasoning_decision',
    matched: false,
    confidence: 0,
    root_trace: {
      node_type: 'leaf',
      signal_type: 'keyword',
      signal_name: 'thinking',
      matched: false,
      confidence: 0,
    },
  },
]

describe('DecisionTraceView', () => {
  it('renders nested operators and leaves for every decision in the trace', () => {
    const markup = renderToStaticMarkup(
      createElement(DecisionTraceView, {
        traces: nestedTrace,
        selectedDecisionName: 'coding_decision',
      }),
    )

    expect(markup).toContain('coding_decision')
    expect(markup).toContain('reasoning_decision')
    expect(markup).toContain('OR')
    expect(markup).toContain('keyword(coding)')
    expect(markup).toContain('domain(code)')
    expect(markup).toContain('keyword(thinking)')
  })

  it('renders nothing for an empty trace', () => {
    const markup = renderToStaticMarkup(
      createElement(DecisionTraceView, { traces: [], selectedDecisionName: null }),
    )
    expect(markup).toBe('')
  })

  it('renders a classifier label badge on leaf nodes that carry one', () => {
    const markup = renderToStaticMarkup(
      createElement(DecisionTraceView, {
        traces: [
          {
            decision_name: 'risk_decision',
            matched: true,
            confidence: 0.8,
            root_trace: {
              node_type: 'leaf',
              signal_type: 'classifier',
              signal_name: 'risk',
              label: 'RISKY',
              matched: true,
              confidence: 0.8,
            },
          },
        ],
        selectedDecisionName: 'risk_decision',
      }),
    )

    expect(markup).toContain('classifier(risk)')
    expect(markup).toContain('RISKY')
  })
})
