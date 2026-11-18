import { describe, expect, it } from 'vitest'

import {
  formatAxisNumber,
  formatCount,
  formatMoney,
  formatPercent,
  formatRate,
  formatSeconds,
  formatSecondsAxis,
} from './metricsFormat'

describe('metrics formatting', () => {
  it('shows a quiet rate per minute or hour instead of a tiny fraction per second', () => {
    expect(formatRate(12.345)).toBe('12.3/s')
    expect(formatRate(4_200)).toBe('4.2K/s')
    expect(formatRate(0.05)).toBe('3/min')
    expect(formatRate(0.0005)).toBe('1.8/h')
    expect(formatRate(0)).toBe('0/s')
    expect(formatRate(null)).toBe('—')
  })

  it('rounds extrapolated counts and shortens large ones', () => {
    expect(formatCount(41.7)).toBe('42')
    expect(formatCount(1_000)).toBe('1K')
    expect(formatCount(1_234_567)).toBe('1.2M')
    expect(formatCount(undefined)).toBe('—')
    expect(formatAxisNumber(0.005)).toBe('0.005')
    expect(formatAxisNumber(123_000)).toBe('123K')
  })

  it('formats durations, shares and money', () => {
    expect(formatSeconds(0.0004)).toBe('<1 ms')
    expect(formatSeconds(0.42)).toBe('420 ms')
    expect(formatSeconds(1.8)).toBe('1.80 s')
    expect(formatSeconds(42.25)).toBe('42.3 s')
    expect(formatSeconds(125)).toBe('2m 5s')
    expect(formatSeconds(Number.NaN)).toBe('—')
    expect(formatSecondsAxis(0.25)).toBe('250 ms')
    expect(formatSecondsAxis(2)).toBe('2 s')
    expect(formatPercent(0.05)).toBe('5%')
    expect(formatPercent(0.1234)).toBe('12.3%')
    expect(formatPercent(0.0002)).toBe('<0.1%')
    expect(formatPercent(null)).toBe('—')
    expect(formatMoney(3.214, 'USD')).toBe('$3.21')
    expect(formatMoney(0.00423, 'USD')).toBe('$0.00423')
    expect(formatMoney(0, 'EUR')).toBe('€0.00')
    expect(formatMoney(2, 'credits')).toBe('2.00 credits')
  })
})
