import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('Plugin Operations permissions', () => {
  it('requires config.write for destructive response-cache controls', () => {
    const source = readFileSync(new URL('./ResponseCachePage.tsx', import.meta.url), 'utf8')

    expect(source).toContain('const canManage = !isReadonly && canWriteConfig(user)')
    expect(source).toContain('disabled={!canManage}')
    expect(source).toContain("disabled={!canManage || confirm !== 'flush response cache'}")
  })

  it('requires config.write for recovery invalidation but not redacted preview', () => {
    const source = readFileSync(new URL('./ContextCompressionPage.tsx', import.meta.url), 'utf8')

    expect(source).toContain('const canManage = !isReadonly && canWriteConfig(user)')
    expect(source).toContain(
      'disabled={!canManage || Object.values(recoveryScope).some((value) => !value.trim())}',
    )
    expect(source).toContain('disabled={isReadonly} onClick={() => void runPreview()}')
  })
})
