import { describe, expect, it } from 'vitest'

import {
  assertProjectionCalibration,
  assertProjectionInputs,
  assertProjectionMappingOutputCount,
  assertProjectionMembers,
  assertProjectionOutputs,
  assertProjectionPartitionSettings,
  normalizeProjectionCalibration,
  normalizeProjectionInputs,
  normalizeProjectionMembers,
  normalizeProjectionOutputs,
} from './configPageProjectionFormSupport'

describe('projection form support', () => {
  it('normalizes and validates partition members', () => {
    const members = normalizeProjectionMembers([' support ', 'account', 'support'])
    expect(members).toEqual(['support', 'account'])
    expect(() => assertProjectionMembers(members, 'support')).not.toThrow()
    expect(() => assertProjectionMembers(members, 'other')).toThrow(/must also appear/i)
    expect(() => assertProjectionPartitionSettings('softmax_exclusive', 0)).toThrow(
      /greater than zero/i,
    )
  })

  it('normalizes typed score inputs and validates source-specific fields', () => {
    const inputs = normalizeProjectionInputs([
      { type: 'embedding', name: ' support ', weight: 0.5, value_source: 'confidence' },
      { type: 'kb_metric', kb: 'privacy', metric: 'risk', weight: 1, value_source: 'score' },
    ])
    expect(inputs[0]).toMatchObject({ type: 'embedding', name: 'support', weight: 0.5 })
    expect(() => assertProjectionInputs(inputs)).not.toThrow()
    expect(() =>
      assertProjectionInputs(normalizeProjectionInputs([{ type: 'kb_metric', weight: 1 }])),
    ).toThrow(/knowledge base/i)
  })

  it('validates optional calibration and threshold output bands', () => {
    const calibration = normalizeProjectionCalibration({ method: 'sigmoid_distance', slope: 6 })
    expect(() => assertProjectionCalibration(calibration)).not.toThrow()
    expect(() => assertProjectionCalibration({ method: 'other' })).toThrow(/sigmoid_distance/i)

    const outputs = normalizeProjectionOutputs([
      { name: 'fast', lt: 0.25 },
      { name: 'escalated', gte: 0.25 },
    ])
    expect(() => assertProjectionOutputs(outputs)).not.toThrow()
    expect(() => assertProjectionMappingOutputCount('multi_emit', outputs)).not.toThrow()
    expect(() => assertProjectionMappingOutputCount('multi_emit', outputs.slice(0, 1))).toThrow(
      /at least two/i,
    )
    expect(() =>
      assertProjectionOutputs(
        normalizeProjectionOutputs([{ name: 'invalid', gt: 0.2, gte: 0.25 }]),
      ),
    ).toThrow(/either gt or gte/i)
  })
})
