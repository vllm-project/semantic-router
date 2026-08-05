import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('OnboardingGuide contract', () => {
  it('uses the shared accessible drawer and persists an interrupted step', () => {
    const source = readFileSync(new URL('./OnboardingGuide.tsx', import.meta.url), 'utf8')
    const styles = readFileSync(new URL('./OnboardingGuide.module.css', import.meta.url), 'utf8')

    expect(source).toContain('useAccessibleDialog<HTMLDivElement>')
    expect(source).toContain('setOnboardingStep(stepIndex)')
    expect(source).toContain("setOnboardingStatus('dismissed')")
    expect(source).toContain('Resume guide')
    expect(source).toContain('role="progressbar"')
    expect(source).toContain('aria-modal="true"')
    expect(source).toContain('data-testid="onboarding-guide-body"')
    expect(source).toContain('data-testid="onboarding-guide-actions"')
    expect(styles).toContain('grid-template-rows: auto auto minmax(0, 1fr) auto')
    expect(styles).toContain('overflow-y: auto')
    expect(styles).toContain('@media (prefers-reduced-motion: reduce)')
  })
})
