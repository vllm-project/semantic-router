import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import ProductLoadingState from './ProductLoadingState'

describe('ProductLoadingState', () => {
  it('uses the shared Semantic Router mark and an accessible status label', () => {
    const markup = renderToStaticMarkup(<ProductLoadingState label="Loading models" />)

    expect(markup).toContain('role="status"')
    expect(markup).toContain('aria-label="Loading models"')
    expect(markup).toContain('src="/vllm-sr-logo.white.png"')
  })
})
