import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationCatalog } from '../../types/evaluationPlane'
import EvaluationMethodReadiness from './EvaluationMethodReadiness'

describe('EvaluationMethodReadiness', () => {
  it('presents a declared release-check capability without exposing internal codes', () => {
    const catalog = {
      change_profiles: [
        { campaign_slots: [{ gate_id: 'G4', name: 'Workload-shift robustness' }] },
      ],
      suites: [
        {
          id: 'declared-shift',
          name: 'Declared shift',
          revision: 'declared-shift-catalog.v4',
          track_ids: ['routing'],
          modes: ['live'],
          executors: { live: 'declared-shift-broker.v3' },
          methods: [
            {
              id: 'routing.declared-shift-live.v1',
              track_id: 'routing',
              qualified_gate_ids: ['G4'],
              evidence_source: 'server_brokered_live',
              status: 'configured',
            },
          ],
        },
      ],
      targets: [
        {
          id: 'healthy-mixture',
          kind: 'mixture-of-models',
          healthy: true,
          track_ids: ['routing'],
          modes: ['live'],
          accepted_executors: { live: ['declared-shift-broker.v3'] },
        },
      ],
    } as EvaluationCatalog

    const markup = renderToStaticMarkup(createElement(EvaluationMethodReadiness, { catalog }))
    const disclosureStart = markup.indexOf('<details')
    const disclosureOpeningEnd = markup.indexOf('>', disclosureStart)
    const summaryEnd = markup.indexOf('</summary>', disclosureOpeningEnd)
    const disclosureEnd = markup.indexOf('</details>', summaryEnd)
    const searchIndex = markup.indexOf('aria-label="Search evaluation methods"')
    const tableIndex = markup.indexOf('<table', summaryEnd)
    const technicalStart = markup.indexOf('data-evaluation-technical-details="true"')
    const productRow = markup.slice(tableIndex, technicalStart)
    const technicalRow = markup.slice(technicalStart)

    expect(markup).toContain('Managed live run')
    expect(markup).toContain('Workload-shift robustness')
    expect(productRow).toContain('Declared shift')
    expect(productRow).toContain('Workload-shift robustness method')
    expect(productRow).toContain('Ready for a managed workload-shift evaluation')
    expect(markup).toContain('Imported results remain diagnostic')
    expect(markup.slice(0, disclosureStart)).toContain('<strong>1</strong> ready')
    expect(markup.slice(0, disclosureStart)).toContain('<strong>0</strong> need setup')
    expect(markup.slice(disclosureOpeningEnd, summaryEnd)).toContain(
      '<strong>Browse benchmark methods</strong>',
    )
    expect(markup.slice(disclosureOpeningEnd, summaryEnd)).toContain(
      'Search all 1 methods when you need implementation and setup details.',
    )
    expect(markup.slice(disclosureStart, disclosureOpeningEnd + 1)).not.toMatch(/\sopen(?:=|\s|>)/)
    expect(searchIndex).toBeGreaterThan(summaryEnd)
    expect(searchIndex).toBeLessThan(disclosureEnd)
    expect(tableIndex).toBeGreaterThan(summaryEnd)
    expect(tableIndex).toBeLessThan(disclosureEnd)
    expect(productRow).not.toContain('routing.declared-shift-live.v1')
    expect(productRow).not.toContain('declared-shift-broker.v3')
    expect(productRow).not.toContain('declared-shift-catalog.v4')
    expect(productRow).not.toContain('G4')
    expect(technicalRow).toContain('routing.declared-shift-live.v1')
    expect(technicalRow).toContain('declared-shift-broker.v3')
    expect(technicalRow).toContain('declared-shift-catalog.v4')
    expect(technicalRow).toContain('G4')
    expect(markup).not.toContain('<details open')
    expect(markup).not.toContain('<strong>0</strong> qualified')
  })

  it('does not report a configured live method ready without a healthy Mixture', () => {
    const catalog = {
      change_profiles: [
        { campaign_slots: [{ gate_id: 'G4', name: 'Workload-shift robustness' }] },
      ],
      suites: [
        {
          id: 'declared-shift',
          name: 'Declared shift',
          revision: 'declared-shift-catalog.v4',
          track_ids: ['routing'],
          modes: ['live'],
          executors: { live: 'declared-shift-broker.v3' },
          methods: [
            {
              id: 'routing.declared-shift-live.v1',
              track_id: 'routing',
              qualified_gate_ids: ['G4'],
              evidence_source: 'server_brokered_live',
              status: 'configured',
            },
          ],
        },
      ],
      targets: [
        {
          id: 'unhealthy-mixture',
          kind: 'mixture-of-models',
          healthy: false,
          track_ids: ['routing'],
          modes: ['live'],
          accepted_executors: { live: ['declared-shift-broker.v3'] },
        },
      ],
    } as EvaluationCatalog

    const markup = renderToStaticMarkup(createElement(EvaluationMethodReadiness, { catalog }))

    expect(markup).toContain('<strong>0</strong> ready')
    expect(markup).toContain('<strong>1</strong> need setup')
    expect(markup).toContain('Setup required')
    expect(markup).toContain('Connect a healthy Mixture that supports this method, then refresh.')
    expect(markup).not.toContain('Ready for a managed workload-shift evaluation')
  })

  it('describes sealed production evidence without claiming experiment orchestration', () => {
    const catalog = {
      change_profiles: [{ campaign_slots: [{ gate_id: 'G8', name: 'Shadow / canary' }] }],
      suites: [
        {
          id: 'production-evidence',
          name: 'Production preference evidence',
          revision: 'production-evidence.v1',
          track_ids: ['preference'],
          modes: ['live'],
          executors: { live: 'sealed-production-results.v1' },
          methods: [
            {
              id: 'preference.production-results.v1',
              track_id: 'preference',
              qualified_gate_ids: ['G8'],
              evidence_source: 'live_production',
              status: 'configured',
            },
          ],
        },
      ],
      targets: [
        {
          id: 'production-mixture',
          kind: 'mixture-of-models',
          healthy: true,
          track_ids: ['preference'],
          modes: ['live'],
          accepted_executors: { live: ['sealed-production-results.v1'] },
        },
      ],
    } as EvaluationCatalog

    const markup = renderToStaticMarkup(createElement(EvaluationMethodReadiness, { catalog }))

    expect(markup).toContain('Sealed production results')
    expect(markup).toContain(
      'Configured sealed production-results source is ready for evaluation.',
    )
    expect(markup).not.toContain('Production experiment')
    expect(markup).not.toContain('guarded production experiment')
  })

  it('uses stable setup guidance and keeps adversarial service text technical', () => {
    const methodID = 'live-agent-task.private-codename.v9'
    const suiteID = 'live-agent-tasks-internal'
    const executorID = 'worker-ledger-agent.v7'
    const rawReason = 'ledger://private-worker E5 missing executor=agent-secret-v2'
    const catalog = {
      change_profiles: [{ campaign_slots: [{ gate_id: 'G6', name: 'Fault recovery' }] }],
      suites: [
        {
          id: suiteID,
          name: 'Agent task evaluation',
          revision: 'internal-ledger-revision.v8',
          executors: { live: executorID },
          methods: [
            {
              id: methodID,
              track_id: 'agentic',
              qualified_gate_ids: ['G6'],
              evidence_source: 'live_runtime',
              status: 'data_required',
              reason: rawReason,
            },
          ],
        },
      ],
    } as EvaluationCatalog

    const markup = renderToStaticMarkup(createElement(EvaluationMethodReadiness, { catalog }))
    const technicalStart = markup.indexOf('data-evaluation-technical-details="true"')
    const productRow = markup.slice(0, technicalStart)
    const technicalRow = markup.slice(technicalStart)

    expect(productRow).toContain('Agent task evaluation')
    expect(productRow).toContain('Fault recovery method')
    expect(productRow).toContain('Agent tasks')
    expect(productRow).toContain('Live system')
    expect(productRow).toContain(
      'Connect complete repeated agent-task results for the selected Mixture, then refresh.',
    )
    for (const rawValue of [methodID, suiteID, executorID, rawReason, 'G6']) {
      expect(productRow).not.toContain(rawValue)
      expect(technicalRow).toContain(rawValue)
    }
    expect(technicalRow).toContain('live_runtime')
    expect(markup).not.toContain('<details open')
  })
})
