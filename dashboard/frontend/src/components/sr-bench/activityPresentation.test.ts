import { describe, expect, it } from 'vitest'
import { duration, elapsedSeconds, responseBytes } from './activityPresentation'

describe('observed call activity presentation', () => {
  it('measures elapsed independently of persisted result updates, including long requests', () => {
    const start = '2026-01-01T00:00:00Z'
    expect(duration(elapsedSeconds(start, Date.parse(start) + 5_000))).toBe('5s')
    expect(duration(elapsedSeconds(start, Date.parse(start) + 3661_000))).toBe('1h 1m 1s')
    expect(duration(elapsedSeconds(start, Date.parse(start) + 48 * 3600_000))).toBe('48h 0m 0s')
  })

  it('does not turn missing or malformed timestamps into invented activity', () => {
    expect(elapsedSeconds(undefined, Date.now())).toBeNull()
    expect(elapsedSeconds('invalid', Date.now())).toBeNull()
    expect(duration(null)).toBe('Not recorded')
    expect(duration(NaN)).toBe('Not recorded')
    expect(elapsedSeconds('2026-01-01T00:00:00Z', Date.parse('2025-12-31T23:59:59Z'))).toBe(0)
  })

  it('labels real wire bytes without estimating tokens, money, or invalid counters', () => {
    expect(responseBytes(0)).toBe('0 bytes')
    expect(responseBytes(8192)).toBe(`${(8192).toLocaleString()} bytes`)
    for (const value of [-1, NaN, Infinity, 1.5]) expect(responseBytes(value)).toBe('Not recorded')
  })
})
