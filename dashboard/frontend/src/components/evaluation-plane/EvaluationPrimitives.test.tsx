import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { EvaluationTag } from './EvaluationPrimitives'

describe('EvaluationTag', () => {
  it('keeps semantic variants on the same shared tag primitive', () => {
    const markup = renderToStaticMarkup(
      createElement(
        'div',
        null,
        createElement(EvaluationTag, {
          tone: 'positive',
          title: 'Ready to collect',
          children: 'Ready',
        }),
        createElement(EvaluationTag, { tone: 'warning', mono: true, children: 'E0' }),
      ),
    )

    expect(markup.match(/data-evaluation-tag="true"/g)).toHaveLength(2)
    expect(markup).toContain('data-tone="positive"')
    expect(markup).toContain('data-tone="warning"')
    expect(markup).toContain('title="Ready to collect"')
  })
})
