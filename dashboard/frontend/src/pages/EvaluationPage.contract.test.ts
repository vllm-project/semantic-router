import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const pageSource = readFileSync(new URL('./EvaluationPage.tsx', import.meta.url), 'utf8')
const typeSource = readFileSync(new URL('../types/evaluationPlane.ts', import.meta.url), 'utf8')
const formSource = readFileSync(
  new URL('../components/evaluation-plane/EvaluationExperimentForm.tsx', import.meta.url),
  'utf8',
)
const reportSource = readFileSync(
  new URL('../components/evaluation-plane/EvaluationReportView.tsx', import.meta.url),
  'utf8',
)
const navigationSource = readFileSync(
  new URL('../components/evaluation-plane/EvaluationNavigation.tsx', import.meta.url),
  'utf8',
)

describe('Evaluation Plane browser contract', () => {
  it('keeps RBAC, server read-only policy, and explicit lifecycle confirmation', () => {
    expect(pageSource).toContain('canWriteEvaluation')
    expect(pageSource).toContain('canRunEvaluation')
    expect(pageSource).toContain('!readonlyLoading && !serverReadonly')
    expect(pageSource.match(/<ConfirmDialog/g)).toHaveLength(2)
    expect(pageSource).toContain('cancelRun')
    expect(pageSource).toContain('plane.createRun({ ...request, auto_start: false })')
    expect(pageSource).toContain('plane.startRun(pendingRun.id)')
    expect(pageSource).not.toMatch(/\b(?:window\.)?confirm\s*\(/)
  })

  it('exposes the complete information architecture and server-catalog target seam', () => {
    for (const label of ['Overview', 'New experiment', 'Runs', 'Reports', 'Compare']) {
      expect(navigationSource).toContain(`label: '${label}'`)
    }
    expect(formSource).toContain('catalog.targets.map')
    expect(formSource).toContain('target_id: targetID')
    expect(formSource).toContain('catalog.change_profiles.map')
    expect(formSource).toContain('change_profile: changeProfile')
    expect(formSource).not.toMatch(/endpoint|target_url/i)
  })

  it('keeps hidden grading outside the TypeScript browser contract', () => {
    expect(typeSource).not.toMatch(
      /casegrading|hidden[_ ]?(?:label|grading)|answer[_ ]?key|reference[_ ]?answer/i,
    )
    expect(typeSource).toContain("'routing'")
    expect(typeSource).toContain("'capacity'")
  })

  it('models gates, three cost ledgers, provenance, artifacts, and recommendations', () => {
    for (const disposition of ['required', 'advisory', 'not_applicable', 'waived']) {
      expect(typeSource).toContain(`'${disposition}'`)
    }
    for (const verdict of ['pass', 'fail', 'unavailable', 'waived', 'not_applicable']) {
      expect(typeSource).toContain(`'${verdict}'`)
    }
    expect(typeSource).toContain('runtime: EvaluationCostAmount')
    expect(typeSource).toContain('evaluation_overhead: EvaluationCostAmount')
    expect(typeSource).toContain('capacity_tco: EvaluationCostAmount')
    expect(typeSource).toContain('gate_contract_version: string')
    expect(typeSource).toContain('change_profile: EvaluationChangeProfileId')
    expect(typeSource).toContain('evidence_refs: string[]')
    expect(typeSource).toContain('coverage?: EvaluationCoverage')
    expect(reportSource).toContain('report.provenance')
    expect(reportSource).toContain('workload_snapshot_digest')
    expect(reportSource).toContain('report.artifacts')
    expect(reportSource).toContain('getEvaluationArtifactURL')
    expect(reportSource).toContain('report.recommendations')
    expect(reportSource).toContain('requiredUnavailable')
    expect(reportSource).toContain('effectiveGateVerdict')
  })
})
