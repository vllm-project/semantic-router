import { describe, expect, it } from 'vitest'

import { shortRelease } from './version'

describe('shortRelease', () => {
  it('keeps the release and leaves the pre-release and build parts to the full version', () => {
    expect(shortRelease('v0.5.0-dev.e1ba40498f')).toBe('v0.5.0')
    expect(shortRelease('0.5.0')).toBe('v0.5.0')
    expect(shortRelease('v0.10.2+build.7')).toBe('v0.10.2')
    expect(shortRelease('v1.2')).toBe('v1.2')
    expect(shortRelease(' v0.5.0rc1 ')).toBe('v0.5.0')
  })

  it('shows a version without a release as it is', () => {
    expect(shortRelease('1ba40498f')).toBe('1ba40498f')
    expect(shortRelease('latest')).toBe('latest')
    expect(shortRelease('v1')).toBe('v1')
    expect(shortRelease('—')).toBe('—')
    expect(shortRelease('  ')).toBe('—')
  })
})
