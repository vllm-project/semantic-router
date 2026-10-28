import { describe, expect, it } from 'vitest'
import { changeDirection, changeSummary, formatSignedChange } from './comparisonMetrics'

describe('comparison changes', () => {
  it('preserves the sign of quality changes and cost savings', () => {
    expect(formatSignedChange(4.25)).toBe('+4.25')
    expect(formatSignedChange(-7.89)).toBe('−7.89')
    expect(formatSignedChange(-125)).toBe('−125')
    expect(changeDirection(4.25)).toBe('positive')
    expect(changeDirection(-7.89)).toBe('negative')
    expect(changeSummary(-7.89, 'quality')).toBe('Lower score')
    expect(changeSummary(-125, 'cost')).toBe('Higher cost')
    expect(changeSummary(10, 'cost')).toBe('Lower cost')
  })

  it('reserves neutral zero for an actual tie', () => {
    for (const value of [0, -0]) {
      expect(formatSignedChange(value)).toBe('0')
      expect(changeDirection(value)).toBe('neutral')
      expect(changeSummary(value, 'quality')).toBe('Same score')
      expect(changeSummary(value, 'cost')).toBe('Same cost')
    }
  })

  it('keeps subprecision changes visible instead of rounding them to zero', () => {
    expect(formatSignedChange(0.0002)).toBe('+0.0002')
    expect(formatSignedChange(-0.0002)).toBe('−0.0002')
    expect(changeDirection(-0.0002)).toBe('negative')
    expect(formatSignedChange(0.01)).toBe('+0.01')
    expect(formatSignedChange(-0.01)).toBe('−0.01')
  })

  it('keeps absent or invalid evidence separate from a zero change', () => {
    for (const value of [null, undefined, NaN, Infinity, -Infinity]) {
      expect(formatSignedChange(value)).toBe('Unknown')
      expect(changeDirection(value)).toBe('unknown')
      expect(changeSummary(value, 'quality')).toBe('Not available')
      expect(changeSummary(value, 'cost')).toBe('Not available')
    }
  })
})
