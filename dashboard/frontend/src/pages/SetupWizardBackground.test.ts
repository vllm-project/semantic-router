import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('setup wizard motion background', () => {
  it('lazy-loads ColorBends only when reduced motion is disabled', () => {
    const backgroundSource = readFileSync(
      new URL('./SetupWizardBackground.tsx', import.meta.url),
      'utf8',
    )
    const colorBendsSource = readFileSync(
      new URL('../components/ColorBends.tsx', import.meta.url),
      'utf8',
    )

    expect(backgroundSource).toContain("lazy(() => import('../components/ColorBends'))")
    expect(backgroundSource).toContain('reducedMotion ? null')
    expect(backgroundSource).toContain("data-motion={reducedMotion ? 'reduced' : 'animated'}")
    expect(colorBendsSource).toContain('if (!container || reducedMotion) return')
    expect(colorBendsSource.indexOf('if (!container || reducedMotion) return')).toBeLessThan(
      colorBendsSource.indexOf('new THREE.WebGLRenderer'),
    )
  })
})
