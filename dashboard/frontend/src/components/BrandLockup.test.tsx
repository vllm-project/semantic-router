import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import BrandLockup from './BrandLockup'

describe('shared brand lockup', () => {
  it('uses the complete website brand with an accessible home link', () => {
    const markup = renderToStaticMarkup(
      createElement(MemoryRouter, null, createElement(BrandLockup)),
    )

    expect(markup).toContain('aria-label="vLLM Semantic Router home"')
    expect(markup).toContain('src="/vllm-sr-logo.white.png"')
    expect(markup).toContain('aria-hidden="true"')
  })

  it('keeps the dashboard asset identical to the website navigation asset', () => {
    const dashboardLogo = readFileSync(
      new URL('../../public/vllm-sr-logo.white.png', import.meta.url),
    )
    const websiteLogo = readFileSync(
      new URL('../../../../website/static/img/vllm-sr-logo.white.png', import.meta.url),
    )

    expect(dashboardLogo.equals(websiteLogo)).toBe(true)
  })
})
