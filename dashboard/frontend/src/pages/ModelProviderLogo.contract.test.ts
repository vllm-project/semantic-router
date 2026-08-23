import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('provider identity marks', () => {
  it('renders the control-plane icon descriptor with a quiet fallback', () => {
    const component = readFileSync(new URL('./ModelProviderLogo.tsx', import.meta.url), 'utf8')
    const support = readFileSync(new URL('./modelProviderLogoSupport.ts', import.meta.url), 'utf8')
    const styles = readFileSync(new URL('./ModelProviderLogo.module.css', import.meta.url), 'utf8')

    expect(support).toContain('@lobehub/icons-static-svg@')
    expect(support).toContain("icon.source !== 'lobe'")
    expect(support).not.toContain('providerIconLobeAssets')
    expect(support).not.toContain('openrouter:')
    expect(component).toContain('icon?: ProviderCatalogIcon')
    expect(component).toContain('onError={() => setImageFailed(true)}')
    expect(styles).not.toContain('radial-gradient')
  })
})
