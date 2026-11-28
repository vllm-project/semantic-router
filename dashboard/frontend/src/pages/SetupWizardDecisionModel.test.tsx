import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { SetupDecisionModelSection } from './SetupWizardDecisionModel'
import { memoryShortfall } from './setupWizardSupport'
import type { DecisionModelOption } from './decisionModelSupport'

const model = (name: string, label: string, minMemoryGiB: number | null): DecisionModelOption => ({
  name,
  artifact: `vllm-sr/${name}`,
  label,
  family: 'Vela 3.1',
  familyId: 'vela3_1',
  provider: 'vllm-sr',
  device: 'cpu',
  hardware: `CPU-friendly · ${minMemoryGiB} GiB memory`,
  summary: `${label} answers questions.`,
  questionTypes: ['domain'],
  modalities: ['text'],
  spanHeads: [],
  parameters: null,
  minMemoryGiB,
  backbone: 'ultrabert',
  servable: true,
})

const fast = model('vela3.1-fast', 'Vela 3.1 fast', 15)
const pico = model('vela3.1-pico-380m', 'Vela 3.1 pico 380M', 3)

describe('setup decision model memory fit', () => {
  it('describes a model that needs more memory than the host has', () => {
    expect(memoryShortfall(fast, 7.7)).toBe('Needs 15 GiB; this host has 7.7 GiB')
    expect(memoryShortfall(pico, 7.7)).toBeNull()
    expect(memoryShortfall(fast, null)).toBeNull()
  })

  it('marks options that do not fit the host memory', () => {
    const markup = renderToStaticMarkup(
      createElement(SetupDecisionModelSection, {
        value: pico.name,
        options: [pico, fast],
        recommended: [pico, fast],
        cpuOnly: true,
        memoryGiB: 7.7,
        onChange: () => undefined,
      }),
    )
    expect(markup).toContain('this host&#x27;s 7.7 GiB of memory')
    expect(markup).toContain('Needs 15 GiB; this host has 7.7 GiB')
    expect(markup.match(/Needs \d+ GiB/g)).toHaveLength(1)
  })

  it('shows no memory warning when the host memory is unknown', () => {
    const markup = renderToStaticMarkup(
      createElement(SetupDecisionModelSection, {
        value: fast.name,
        options: [fast, pico],
        recommended: [fast, pico],
        cpuOnly: true,
        onChange: () => undefined,
      }),
    )
    expect(markup).not.toContain('Needs ')
    expect(markup).not.toContain('of memory are marked')
  })
})
