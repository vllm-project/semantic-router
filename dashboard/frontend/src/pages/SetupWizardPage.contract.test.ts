import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('SetupWizardPage async contract', () => {
  const source = readFileSync(new URL('./SetupWizardPage.tsx', import.meta.url), 'utf8')

  it('drops stale preset, remote import, and validation responses', () => {
    expect(source).toContain('presetRequestGuardRef.current.isCurrent(generation)')
    expect(source).toContain('remoteImportGuardRef.current.isCurrent(generation)')
    expect(source).toContain('validationGuardRef.current.isCurrent(generation)')
    expect(source).toContain('presetCatalogGuardRef.current.isCurrent(generation)')
    expect(source).toContain('aria-labelledby={`setup-step-${currentStep}-button`}')
  })

  it('leaves setup for Home or Get started after re-reading the setup state', () => {
    expect(source).toContain('await refreshSetupState()')
    expect(source).toMatch(/leaveSetup\(['"]\/['"]\)/)
    expect(source).toMatch(/leaveSetup\(['"]\/get-started['"]\)/)
    expect(source).not.toMatch(/navigate\(['"]\/dashboard/)
  })

  it('uses the shared calm backdrop instead of a WebGL scene', () => {
    expect(source).toContain('<AuthBackdrop>')
    expect(source).not.toMatch(/three|ColorBends|SetupWizardBackground/)
  })
})
