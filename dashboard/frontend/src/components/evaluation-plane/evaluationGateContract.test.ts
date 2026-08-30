import { describe, expect, it } from 'vitest'

import type { EvaluationGate } from '../../types/evaluationPlane'
import {
  gateApplicabilityForProfile,
  GATE_CONTRACT_DEFINITIONS,
  SUPPORTED_GATE_CONTRACT_VERSION,
} from './evaluationGateContract'
import { effectiveGateVerdict } from './evaluationPresentation'

function gate(disposition: EvaluationGate['disposition'], verdict: EvaluationGate['verdict']) {
  return {
    id: 'G4',
    name: 'Robustness / OOD',
    disposition,
    verdict,
    change_profile: 'recipe',
    contract_version: SUPPORTED_GATE_CONTRACT_VERSION,
    evidence_refs: ['records.jsonl'],
  } satisfies EvaluationGate
}

describe('evaluation gate contract presentation', () => {
  it('keeps every profile explicit across the complete G0-G9 contract', () => {
    expect(GATE_CONTRACT_DEFINITIONS.map((entry) => entry.id)).toEqual([
      'G0',
      'G1',
      'G2',
      'G3',
      'G4',
      'G5',
      'G6',
      'G7',
      'G8',
      'G9',
    ])
    expect(gateApplicabilityForProfile('recipe').map((entry) => entry.disposition)).toEqual([
      'required',
      'required',
      'required',
      'required',
      'required',
      'required',
      'not_applicable',
      'required',
      'advisory',
      'not_applicable',
    ])
    expect(gateApplicabilityForProfile('online_adaptation')).toHaveLength(10)
    expect(
      gateApplicabilityForProfile('online_adaptation').every(
        (entry) => entry.disposition === 'required',
      ),
    ).toBe(true)
  })

  it('never presents unavailable required evidence as a pass', () => {
    expect(effectiveGateVerdict('pass', [gate('required', 'unavailable')])).toBe('unavailable')
    expect(effectiveGateVerdict('pass', [gate('advisory', 'unavailable')])).toBe('pass')
    expect(effectiveGateVerdict('pass', [gate('required', 'fail')])).toBe('fail')
  })
})
