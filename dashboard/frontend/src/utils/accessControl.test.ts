import { describe, expect, it } from 'vitest'

import { canWriteConfig } from './accessControl'

describe('config write access', () => {
  it('allows explicit config writers and write-capable roles', () => {
    expect(canWriteConfig({ role: 'read', permissions: ['config.write'] })).toBe(true)
    expect(canWriteConfig({ role: 'admin' })).toBe(true)
    expect(canWriteConfig({ role: 'write' })).toBe(true)
  })

  it('keeps read-only users out of mutating dashboard controls', () => {
    expect(canWriteConfig({ role: 'read', permissions: ['config.read'] })).toBe(false)
    expect(canWriteConfig({ role: 'write', permissions: ['config.read'] })).toBe(false)
    expect(canWriteConfig({ role: 'admin', permissions: [] })).toBe(false)
    expect(canWriteConfig(null)).toBe(false)
  })
})
