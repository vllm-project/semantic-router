import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('provider identity marks', () => {
  it('uses the pinned LobeHub icon library with a quiet fallback', () => {
    const component = readFileSync(new URL('./ModelProviderLogo.tsx', import.meta.url), 'utf8')
    const styles = readFileSync(new URL('./ModelProviderLogo.module.css', import.meta.url), 'utf8')

    expect(component).toContain('@lobehub/icons-static-svg@')
    expect(component).toContain("openrouter: { slug: 'openrouter' }")
    expect(component).toContain("anthropic: { slug: 'anthropic' }")
    expect(component).toContain('onError={() => setImageFailed(true)}')
    expect(styles).not.toContain('radial-gradient')
  })
})
