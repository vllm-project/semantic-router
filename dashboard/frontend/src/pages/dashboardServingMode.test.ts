import { describe, expect, it } from 'vitest'
import { describeServingMode } from './dashboardServingMode'

describe('reported serving mode', () => {
  it('describes the identified Router and engine separately', () => {
    expect(describeServingMode('router')).toEqual({
      label: 'Router mode',
      description: 'Serves System One and routes Chat through configured recipes.',
    })
    expect(describeServingMode('engine')).toEqual({
      label: 'Engine mode',
      description: 'Serves System One with routing disabled.',
    })
  })

  it.each([undefined, null, 'unknown', 'standalone', 'extproc', 'native', 'docker'])(
    'does not infer a serving mode from %s',
    (value) => {
      expect(describeServingMode(value).label).toBe('Mode unavailable')
    },
  )
})
