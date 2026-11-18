import { describe, expect, it } from 'vitest'
import { cpuRecommendedDecisionModel } from './decisionModelRecommendation'
import { DECISION_CATALOG_MODELS } from './decisionModelSupport'

const servable = (...names: string[]) => names.map((name) => ({ name, servable: true }))
const cpu = { observedDevices: ['cpu'], configuredDevices: [] }

describe('CPU decision model recommendation', () => {
  it('recommends the first servable model of the CPU order', () => {
    expect(
      cpuRecommendedDecisionModel(
        servable('vela3-fast', 'vela3.1-pico-380m', 'vela3.1-fast', 'vela3.1-pico-150m'),
        cpu,
      ),
    ).toBe('vela3.1-fast')
    expect(
      cpuRecommendedDecisionModel(
        servable('vela3-fast', 'vela3.1-pico-380m', 'vela3.1-pico-150m'),
        cpu,
      ),
    ).toBe('vela3.1-pico-150m')
    expect(cpuRecommendedDecisionModel(servable('vela3-fast', 'vela3.1-pico-380m'), cpu)).toBe(
      'vela3.1-pico-380m',
    )
  })

  it('recommends vela3.1-fast on CPU from the shipped catalog', () => {
    expect(cpuRecommendedDecisionModel(DECISION_CATALOG_MODELS, cpu)).toBe('vela3.1-fast')
  })

  it('falls back to the CPU default while no Vela 3.1 model is servable', () => {
    expect(
      cpuRecommendedDecisionModel(
        [
          { name: 'vela3.1-fast', servable: false },
          { name: 'vela3.1-pico-150m', servable: false },
          ...servable('vela3-fast'),
        ],
        cpu,
      ),
    ).toBe('vela3-fast')
    expect(cpuRecommendedDecisionModel(servable('d3-mini'), cpu)).toBeNull()
  })

  it('recommends nothing when an accelerator is in use or the hardware is unknown', () => {
    const options = servable('vela3-fast', 'vela3.1-fast')
    expect(
      cpuRecommendedDecisionModel(options, { ...cpu, observedDevices: ['cpu', 'rocm:0'] }),
    ).toBeNull()
    expect(cpuRecommendedDecisionModel(options, { ...cpu, gpuAvailable: true })).toBeNull()
    expect(
      cpuRecommendedDecisionModel(options, { observedDevices: [], configuredDevices: [] }),
    ).toBeNull()
    expect(
      cpuRecommendedDecisionModel(options, {
        observedDevices: [undefined, 'auto'],
        configuredDevices: ['auto'],
      }),
    ).toBeNull()
  })

  it('uses saved decision devices only while nothing is observed', () => {
    const options = servable('vela3-fast')
    expect(
      cpuRecommendedDecisionModel(options, { observedDevices: [], configuredDevices: ['cpu'] }),
    ).toBe('vela3-fast')
    expect(
      cpuRecommendedDecisionModel(options, {
        observedDevices: ['cuda:0'],
        configuredDevices: ['cpu'],
      }),
    ).toBeNull()
    expect(
      cpuRecommendedDecisionModel(options, {
        observedDevices: [],
        configuredDevices: ['cpu', 'cuda:1'],
      }),
    ).toBeNull()
  })
})
