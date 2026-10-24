import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { DecisionTraceView } from './DecisionTraceView'

describe('Decision trace diagnostics', () => {
  it('renders a classifier timeout as unknown with fail_request, without a fabricated zero score', () => {
    const html = renderToStaticMarkup(
      <DecisionTraceView
        selectedDecision={null}
        trace={[
          {
            decision_name: 'unresolved',
            state: 'unknown',
            matched: false,
            confidence: 0,
            on_unknown: 'fail_request',
            root_trace: {
              node_type: 'AND',
              state: 'unknown',
              matched: false,
              confidence: 0,
              children: [
                {
                  node_type: 'leaf',
                  signal_type: 'classifier',
                  signal_name: 'intent',
                  label: 'coding',
                  state: 'unknown',
                  matched: false,
                  confidence: 0,
                  signal_error: 'classifier timed out',
                },
              ],
            },
          },
        ]}
      />,
    )
    expect(html).toContain('Unknown')
    expect(html).toContain('Unknown policy: fail_request')
    expect(html).toContain('classifier:intent:coding')
    expect(html).toContain('Error: classifier timed out')
    expect(html).not.toContain('0%')
    expect(html).not.toContain('No match')
    expect(html).not.toContain('Selected')
  })
  it('keeps rejected and lower priority matched decisions visible without calling them selected', () => {
    const html = renderToStaticMarkup(
      <DecisionTraceView
        selectedDecision="chosen"
        trace={[
          { decision_name: 'rejected', state: 'false', matched: false },
          { decision_name: 'chosen', state: 'true', matched: true },
          { decision_name: 'lower', state: 'true', matched: true },
        ]}
      />,
    )
    expect(html).toContain('rejected: No match')
    expect(html).toContain('chosen: Match')
    expect(html).toContain('lower: Match')
    expect(html.match(/Selected/g)).toHaveLength(1)
  })
  it('shows reported leaf scores only and tolerates absent or null trace nodes', () => {
    const html = renderToStaticMarkup(
      <DecisionTraceView
        selectedDecision="chosen"
        trace={[
          {
            decision_name: 'chosen',
            state: 'true',
            root_trace: {
              node_type: 'OR',
              state: 'true',
              confidence: 1,
              children: [
                null,
                {
                  node_type: 'leaf',
                  signal_type: 'domain',
                  signal_name: 'math',
                  state: 'true',
                  confidence: 0.8,
                  confidence_scored: true,
                },
              ],
            },
          },
        ]}
      />,
    )
    expect(html).toContain('Score 80%')
    expect(html).not.toContain('100%')
    expect(renderToStaticMarkup(<DecisionTraceView selectedDecision={null} />)).toBe('')
  })
})
