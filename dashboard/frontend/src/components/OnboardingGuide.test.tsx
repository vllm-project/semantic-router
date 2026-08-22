import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('OnboardingGuide contract', () => {
  it('uses the shared accessible drawer and persists an interrupted step', () => {
    const source = readFileSync(new URL('./OnboardingGuide.tsx', import.meta.url), 'utf8')
    const styles = readFileSync(new URL('./OnboardingGuide.module.css', import.meta.url), 'utf8')

    expect(source).toContain('useAccessibleDialog<HTMLDivElement>')
    expect(source).toContain('setOnboardingStep(stepIndex)')
    expect(source).toContain("setOnboardingStatus('dismissed')")
    expect(source).toContain('Resume product guide')
    expect(source).toContain('role="progressbar"')
    expect(source).toContain('aria-modal="true"')
    expect(source).toContain('data-testid="onboarding-guide-body"')
    expect(source).toContain('data-testid="onboarding-guide-actions"')
    expect(styles).toContain('grid-template-rows: auto auto minmax(0, 1fr) auto')
    expect(styles).toContain('overflow-y: auto')
    expect(styles).toContain('@media (prefers-reduced-motion: reduce)')
  })

  it('guides the current model-to-insights product journey', () => {
    const source = readFileSync(new URL('./OnboardingGuide.tsx', import.meta.url), 'utf8')

    for (const step of [
      'Connect your models',
      'Build a Mixture-of-Models',
      'Test your model path',
      'Give your team access',
      'See what the router saved',
    ]) {
      expect(source).toContain(step)
    }
    expect(source).toContain("route: '/access/teams'")
    expect(source).toContain("route: '/insights'")
  })
})
