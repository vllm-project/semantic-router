import { describe, expect, it } from 'vitest'

import { isSelectedSignal, type Selection } from './builderPageTypes'

describe('isSelectedSignal', () => {
  const complexity = { signalType: 'complexity', name: 'needs_reasoning' }
  const decision = { signalType: 'decision', name: 'needs_reasoning' }

  it('tells apart signals that share a name across types', () => {
    const selection: Selection = { kind: 'signal', name: 'needs_reasoning', signalType: 'decision' }

    expect(isSelectedSignal(selection, decision)).toBe(true)
    expect(isSelectedSignal(selection, complexity)).toBe(false)
  })

  it('matches only signal selections', () => {
    expect(isSelectedSignal(null, decision)).toBe(false)
    expect(isSelectedSignal({ kind: 'signal', name: '__list__' }, decision)).toBe(false)
    expect(isSelectedSignal({ kind: 'route', name: 'needs_reasoning' }, decision)).toBe(false)
  })
})
