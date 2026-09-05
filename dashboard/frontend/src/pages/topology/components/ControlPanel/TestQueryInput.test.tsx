import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { TestQueryInput } from './TestQueryInput'

describe('TestQueryInput', () => {
  it('disables the send action and explains why when the scope has no entrypoint', () => {
    const markup = renderToStaticMarkup(
      createElement(TestQueryInput, {
        value: 'hello',
        onChange: vi.fn(),
        onTest: vi.fn(),
        isLoading: false,
        disabledReason: 'Recipe "Balanced" has no entrypoint, so it cannot be tested.',
      }),
    )

    expect(markup).toContain('disabled=""')
    expect(markup).toContain('Recipe &quot;Balanced&quot; has no entrypoint')
  })

  it('stays enabled with non-empty input and no disabled reason', () => {
    const markup = renderToStaticMarkup(
      createElement(TestQueryInput, {
        value: 'hello',
        onChange: vi.fn(),
        onTest: vi.fn(),
        isLoading: false,
      }),
    )

    expect(markup).not.toContain('disabled=""')
  })
})
