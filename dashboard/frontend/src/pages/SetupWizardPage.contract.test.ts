import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('SetupWizardPage async contract', () => {
  it('drops stale preset, remote import, and validation responses', () => {
    const source = readFileSync(new URL('./SetupWizardPage.tsx', import.meta.url), 'utf8')

    expect(source).toContain('presetRequestGuardRef.current.isCurrent(generation)')
    expect(source).toContain('remoteImportGuardRef.current.isCurrent(generation)')
    expect(source).toContain('validationGuardRef.current.isCurrent(generation)')
    expect(source).toContain('presetCatalogGuardRef.current.isCurrent(generation)')
    expect(source).toContain('Build your first Mixture-of-Models.')
    expect(source).toContain('aria-labelledby={`setup-step-${currentStep}-button`}')
  })
})
