import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type {
  SelectionCandidateRef,
  SelectionObjectiveStage,
  SelectionTrace,
} from '../types/selectionTrace'
import InsightsRecordSection from './InsightsRecordSection'
import InsightsSelectionTrace from './InsightsSelectionTrace'
import { buildRoutingExplanationSections } from './insightsPageRouting'
import type { InsightsRecord } from './insightsPageTypes'

const candidate = (
  Model: string,
  overrides: Partial<SelectionCandidateRef> = {},
): SelectionCandidateRef => ({
  Model,
  LoRAName: '',
  Weight: 0,
  UseReasoning: null,
  ReasoningDescription: '',
  ReasoningMode: '',
  ReasoningEffort: '',
  ...overrides,
})

const alpha = candidate('publisher/model-alpha')
const beta = candidate('publisher/model-beta')

function latencyStage(available: number): SelectionObjectiveStage {
  return {
    factor: 'latency',
    metric: 'ttft',
    percentile: 95,
    tolerance: 0,
    action: available === 2 ? 'applied' : 'skipped',
    reason: available === 2 ? 'tolerance_band' : 'incomplete_latency_coverage',
    available,
    total: 2,
    candidates: [
      { candidate: alpha, ...(available > 0 ? { value: 0, metric: 'ttft' } : {}) },
      {
        candidate: beta,
        ...(available === 2
          ? { value: 0.2, metric: 'ttft', elimination_reason: 'outside_tolerance' as const }
          : {}),
      },
    ],
  }
}

function renderTrace(trace: SelectionTrace, selectedModel?: string) {
  return renderToStaticMarkup(
    <InsightsSelectionTrace trace={trace} selectedModel={selectedModel} />,
  )
}

describe('recorded selection stages', () => {
  it.each([0, 1, 2])(
    'faithfully shows latency coverage %i / 2 with zero distinct from missing evidence',
    (available) => {
      const stage = latencyStage(available)
      const html = renderTrace(
        { stages: [stage], final_survivors: available === 2 ? [alpha] : [alpha, beta] },
        alpha.Model,
      )
      expect(html).toContain(`${available} / 2 measured`)
      expect(html).toContain('P95')
      expect(html).toContain('First response observation (TTFT)')
      expect(html).toContain('Inspect latency candidates')
      expect(html).not.toContain(' open=""')
      expect(html).not.toContain('<pre')
      if (available === 0) expect(html).not.toContain('<td>0 s</td>')
      else expect(html).toContain('<td>0 s</td>')
      if (available < 2) {
        expect(html).toContain('Skipped')
        expect(html).toContain(
          'Latency measurements do not cover every candidate. No candidates were removed.',
        )
        expect(html).toContain('<td>Unknown</td>')
        expect(html).toContain('Not filtered')
        expect(html).not.toContain('Removed ·')
      } else {
        expect(html).toContain('Applied')
        expect(html).toContain('Removed · outside tolerance')
        expect(html).toContain('Retained at this stage')
        expect(html).not.toContain('<td>Unknown</td>')
      }
    },
  )

  it('keeps objective survivors separate from actual selection and retains exact reasoning variants', () => {
    const low = candidate(alpha.Model, {
      UseReasoning: false,
      ReasoningEffort: 'low',
      LoRAName: 'specialist',
    })
    const high = candidate(alpha.Model, { UseReasoning: true, ReasoningEffort: 'high' })
    const html = renderTrace({ stages: [], final_survivors: [low, high] }, beta.Model)
    expect(html).toContain(`<dt>Actual selected model</dt><dd>${beta.Model}</dd>`)
    expect(html).toContain('Candidates retained by the objective')
    expect(html).toContain('Adapter: specialist · Reasoning off · Effort: low')
    expect(html).toContain('Reasoning on · Effort: high')
    expect(html).toContain('No objective stages were recorded')
    expect(html).not.toContain('Winner')
    expect(html).not.toContain('0 stages applied')
  })

  it('shows missing quality evidence and small cost forecasts without manufacturing probability or a cap', () => {
    const trace: SelectionTrace = {
      final_survivors: [alpha],
      stages: [
        {
          factor: 'quality',
          action: 'applied',
          reason: 'tolerance_band',
          tolerance: 0.07,
          available: 1,
          total: 2,
          candidates: [
            { candidate: alpha, value: 87.3 },
            { candidate: beta, elimination_reason: 'missing_measurement' },
          ],
        },
        {
          factor: 'cost',
          action: 'applied',
          reason: 'tolerance_band',
          tolerance: 0,
          available: 1,
          total: 1,
          candidates: [{ candidate: alpha, value: 0.00000009 }],
        },
      ],
    }
    const html = renderTrace(trace, alpha.Model)
    expect(html).toContain('<td>87.3</td>')
    expect(html).not.toContain('87.3%')
    expect(html).toContain('Removed · missing measurement')
    expect(html).toContain('7%')
    expect(html).toContain('$0.00000009')
    expect(html).toContain('Request cost forecast (USD)')
    expect(html).toContain(
      'Configured-rate forecast, not a provider invoice or output-token limit.',
    )
  })

  it('distinguishes per-candidate latency metrics from the configured fallback order', () => {
    const stage: SelectionObjectiveStage = {
      ...latencyStage(2),
      metric: 'tpot_then_ttft',
      candidates: [
        { candidate: alpha, value: 0, metric: 'tpot' },
        { candidate: beta, value: 0.2, metric: 'ttft', elimination_reason: 'outside_tolerance' },
      ],
    }
    const html = renderTrace({ stages: [stage], final_survivors: [alpha] })
    expect(html).toContain('TPOT, otherwise TTFT')
    expect(html).toContain('<td>0 s / token</td><td>Response duration / output token (TPOT)</td>')
    expect(html).toContain('<td>0.2 s</td><td>First response observation (TTFT)</td>')
  })

  it('uses the canonical record trace in a full-width section and omits it when unrecorded', () => {
    const record: InsightsRecord = {
      id: 'selection-record',
      timestamp: '2026-09-20T00:00:00Z',
      turn_index: 0,
      decision_tier: 0,
      decision_priority: 0,
      signals: {},
      selected_model: beta.Model,
    }
    expect(buildRoutingExplanationSections(record)).toEqual([])
    record.route_diagnostics = {
      selection_trace: { stages: [latencyStage(1)], final_survivors: [alpha, beta] },
    }
    const sections = buildRoutingExplanationSections(record)
    expect(sections).toHaveLength(1)
    expect(sections[0].title).toBe('Selection Stages')
    const html = renderToStaticMarkup(
      <InsightsRecordSection section={sections[0]} sectionIndex={0} />,
    )
    expect(html).toContain('Collapse Selection Stages')
    expect(html).toContain('Inspect latency candidates')
    expect(html).toContain(`<dt>Actual selected model</dt><dd>${beta.Model}</dd>`)
    expect(html).not.toContain('selection_trace')
  })
})
