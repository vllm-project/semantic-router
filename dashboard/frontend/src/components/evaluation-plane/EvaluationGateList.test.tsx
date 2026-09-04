import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationGate } from '../../types/evaluationReport'
import EvaluationGateList from './EvaluationGateList'

describe('EvaluationGateList', () => {
  it('keeps implementation identities behind technical details', () => {
    const supportingRecord = 'artifact://evaluation/raw-run-identifier'
    const gate: EvaluationGate = {
      id: 'G3',
      name: 'Offline value',
      disposition: 'required',
      verdict: 'pass',
      change_profile: 'recipe',
      contract_version: 'evaluation-release-gates.v2',
      evidence_refs: [supportingRecord],
      owner: 'recipe-and-model-pool',
      rationale: 'Server-attested E0 evidence met G3 controlled-pair-contract-v2.',
      description: 'evaluation-release-gates.v2 check definition',
      observed: 0.02,
      threshold: {
        operator: 'private-threshold-operator',
        value: 0.01,
        unit: 'non-inferiority-headroom',
      },
    }

    const markup = renderToStaticMarkup(createElement(EvaluationGateList, { gates: [gate] }))
    const technicalDetailsStart = markup.indexOf('<details')
    const supportingRecordIndex = markup.indexOf(supportingRecord)
    const technicalDetailsEnd = markup.indexOf('</details>', supportingRecordIndex)

    expect(markup).toContain('<strong>Offline value</strong>')
    expect(markup).toContain('Required check')
    expect(markup).toContain('1 supporting record')
    expect(markup.slice(0, technicalDetailsStart)).toContain('Observed 0.02 · Target 0.01')
    expect(markup.slice(0, technicalDetailsStart)).not.toContain('private-threshold-operator')
    expect(markup.slice(0, technicalDetailsStart)).not.toContain('non-inferiority-headroom')
    expect(markup.match(/data-evaluation-tag="true"/g)).toHaveLength(1)
    expect(markup.slice(0, technicalDetailsStart)).toContain(gate.name)
    expect(markup.slice(0, technicalDetailsStart)).not.toContain(gate.id)
    expect(markup.slice(0, technicalDetailsStart)).not.toContain(gate.contract_version)
    expect(markup.slice(0, technicalDetailsStart)).not.toContain(gate.rationale)
    expect(markup.slice(0, technicalDetailsStart)).not.toContain(gate.description)
    expect(markup.slice(0, technicalDetailsStart)).not.toContain(gate.owner)
    expect(markup.slice(0, technicalDetailsStart)).not.toContain(supportingRecord)
    expect(markup.slice(technicalDetailsStart, supportingRecordIndex)).toMatch(
      /<summary[^>]*>Technical details<\/summary>/,
    )
    expect(markup.slice(technicalDetailsStart, supportingRecordIndex)).toContain(
      '<strong>Evaluation owner</strong><span>recipe-and-model-pool</span>',
    )
    expect(markup.slice(technicalDetailsStart, supportingRecordIndex)).toContain(
      'private-threshold-operator 0.01 non-inferiority-headroom',
    )
    expect(technicalDetailsEnd).toBeGreaterThan(supportingRecordIndex)
    expect(
      markup.slice(technicalDetailsStart, markup.indexOf('>', technicalDetailsStart) + 1),
    ).not.toMatch(/\sopen(?:=|\s|>)/)
  })
})
