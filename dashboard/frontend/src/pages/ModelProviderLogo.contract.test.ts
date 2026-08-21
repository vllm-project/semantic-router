import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('provider identity marks', () => {
  it('uses the pinned LobeHub icon library with a quiet fallback', () => {
    const component = readFileSync(new URL('./ModelProviderLogo.tsx', import.meta.url), 'utf8')
    const support = readFileSync(new URL('./modelProviderLogoSupport.ts', import.meta.url), 'utf8')
    const styles = readFileSync(new URL('./ModelProviderLogo.module.css', import.meta.url), 'utf8')

    expect(support).toContain('@lobehub/icons-static-svg@')
    expect(support).toContain("openrouter: { slug: 'openrouter' }")
    expect(support).toContain("anthropic: { slug: 'anthropic' }")
    expect(support).toContain(
      "sglang: 'https://raw.githubusercontent.com/sgl-project/sgl-docs/main/favicon.png'",
    )
    expect(component).toContain('onError={() => setImageFailed(true)}')
    expect(styles).not.toContain('radial-gradient')
  })
})
