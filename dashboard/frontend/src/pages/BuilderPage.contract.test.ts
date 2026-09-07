import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('Config Builder entry mode', () => {
  it('opens the visual workspace instead of the DSL editor', () => {
    const source = readFileSync(new URL('./BuilderPage.tsx', import.meta.url), 'utf8')

    expect(source).toMatch(/setMode\((['"])visual\1\)/)
    expect(source).not.toMatch(/setMode\((['"])dsl\1\);\s*\n\s*}, \[setMode\]\)/)
  })

  it('gates deploy on server policy, runtime storage, and deploy permission', () => {
    const source = readFileSync(new URL('./BuilderPage.tsx', import.meta.url), 'utf8')

    expect(source).toContain('serverReadonly ||')
    expect(source).toContain('!runtimeConfigWritable ||')
    expect(source).toContain('!hasDeployPermission')
    expect(source).toContain('writable runtime configuration mount')
  })
})
