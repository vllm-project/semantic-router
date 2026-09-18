import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import EvaluationExperimentCapacitySLO from './EvaluationExperimentCapacitySLO'
import type { EvaluationExperimentFormModel } from './useEvaluationExperimentForm'
import { defaultEvaluationCapacityLoadProtocol } from '../../utils/evaluationCapacitySLOContract'

function capacityForm(active: boolean): EvaluationExperimentFormModel {
  return {
    capacitySLOActive: active,
    capacityLoadProtocol: active ? defaultEvaluationCapacityLoadProtocol(8) : undefined,
    capacitySLOInput: {
      requiredConcurrency: '',
      maxLatencyP95MS: '',
      maxErrorRate: '',
      minThroughputRPS: '',
      minThroughputScalingEfficiency: '',
    },
    concurrency: 8,
    baselineLocked: false,
    setCapacitySLOField: vi.fn(),
    applyCapacitySLOPreset: vi.fn(),
  } as unknown as EvaluationExperimentFormModel
}

describe('EvaluationExperimentCapacitySLO', () => {
  it('is absent outside live Capacity and never inserts a silent passing default', () => {
    expect(
      renderToStaticMarkup(
        createElement(EvaluationExperimentCapacitySLO, { form: capacityForm(false) }),
      ),
    ).toBe('')

    const markup = renderToStaticMarkup(
      createElement(EvaluationExperimentCapacitySLO, { form: capacityForm(true) }),
    )
    expect(markup).toContain('Capacity service objective')
    expect(markup).toContain('Required for live capacity')
    expect(markup).toContain('No inferred pass criteria')
    expect(markup).toContain('Recorded capacity load plan')
    expect(markup).toContain('1 → 2 → 4 → 8 concurrent requests')
    expect(markup).toContain('100 requests × 3 independent windows (minimum 3)')
    expect(markup).toContain(
      '95% worst-window error bound · error spread ≤ 5% · throughput and p95 variation ≤ 20%',
    )
    expect(markup).toContain('Optional starting points')
    expect(markup).toContain('Latency guardrail')
    expect(markup).toContain('Balanced service')
    expect(markup).toContain('Throughput guardrail')
    expect(markup.match(/required=""/g)).toHaveLength(5)
    expect(markup).not.toContain('value="750"')
    expect(markup).not.toContain('value="0.01"')
    expect(markup).not.toMatch(/\b(?:E[0-5]|G[0-9])\b/)
  })
})
