import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ConfigPageDecisionConditionsView from './ConfigPageDecisionConditionsView'

describe('ConfigPageDecisionConditionsView', () => {
  it('renders the nested built-in complex decision as a readable logic tree', () => {
    const markup = renderToStaticMarkup(
      <ConfigPageDecisionConditionsView
        conditions={[
          {
            operator: 'OR',
            conditions: [
              { type: 'projection', name: 'balance_deliberate_workload' },
              { type: 'projection', name: 'balance_needs_recovery' },
              { type: 'context', name: 'balance_context_from_30k_to_60k' },
            ],
          },
          {
            operator: 'NOT',
            conditions: [{ type: 'conversation', name: 'balance_has_images' }],
          },
          {
            operator: 'NOT',
            conditions: [{ type: 'conversation', name: 'balance_has_tools' }],
          },
        ]}
      />,
    )

    expect(markup).toContain('Any of')
    expect(markup).toContain('Not')
    expect(markup).toContain('Projection')
    expect(markup).toContain('balance_deliberate_workload')
    expect(markup).toContain('Conversation')
    expect(markup).toContain('balance_has_images')
    expect(markup).not.toContain('Unnamed condition')
    expect(markup).not.toContain('&gt;:')
  })

  it('shows typed predicate metadata without producing an empty type-name pair', () => {
    const markup = renderToStaticMarkup(
      <ConfigPageDecisionConditionsView
        conditions={[
          {
            type: 'classifier',
            name: 'request_complexity',
            label: 'hard',
            predicate: { gte: 0.8 },
            on_error: 'no_match',
          },
          {},
        ]}
      />,
    )

    expect(markup).toContain('Classifier')
    expect(markup).toContain('request_complexity')
    expect(markup).toContain('Label hard')
    expect(markup).toContain('≥ 0.8')
    expect(markup).toContain('Skip on error')
    expect(markup).toContain('Unnamed condition')
  })
})
