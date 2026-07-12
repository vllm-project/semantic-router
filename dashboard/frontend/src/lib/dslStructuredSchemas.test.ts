import { describe, expect, it } from 'vitest'

import {
  ALGORITHM_TYPES,
  getAlgorithmFieldSchema,
  getPluginFieldSchema,
  getSignalFieldSchema,
  PLUGIN_TYPES,
  SIGNAL_TYPES,
  type FieldSchema,
} from './dslSchemas'

function requireField(schema: FieldSchema[], key: string): FieldSchema {
  const field = schema.find((candidate) => candidate.key === key)
  if (!field) throw new Error(`Missing schema field ${key}`)
  return field
}

function flattenSchema(schema: FieldSchema[]): FieldSchema[] {
  return schema.flatMap((field) => [field, ...flattenSchema(field.fields || [])])
}

describe('DSL structured field schemas', () => {
  it('does not expose raw JSON controls for known algorithm, signal, or plugin fields', () => {
    const schemas = [
      ...ALGORITHM_TYPES.flatMap((type) => getAlgorithmFieldSchema(type)),
      ...SIGNAL_TYPES.flatMap((type) => getSignalFieldSchema(type)),
      ...PLUGIN_TYPES.flatMap((type) => getPluginFieldSchema(type)),
    ]

    expect(flattenSchema(schemas).map((field) => field.type)).not.toContain('json')
  })

  it('describes workflow and multi-factor structures with typed nested schemas', () => {
    const workflows = getAlgorithmFieldSchema('workflows')
    const planner = requireField(workflows, 'planner')
    const roles = requireField(workflows, 'roles')
    const final = requireField(workflows, 'final')
    expect(planner.type).toBe('object')
    expect(requireField(planner.fields || [], 'max_completion_tokens').type).toBe('number')
    expect(roles.type).toBe('object[]')
    expect(requireField(roles.fields || [], 'models').type).toBe('string[]')
    expect(final.type).toBe('object')

    const multiFactor = getAlgorithmFieldSchema('multi_factor')
    expect(requireField(multiFactor, 'weights').type).toBe('object')
    expect(requireField(multiFactor, 'slo').fields?.map((field) => field.key)).toEqual([
      'max_tpot_ms',
      'max_ttft_ms',
      'max_cost_per_1m',
      'max_inflight',
    ])
  })

  it('maps stable signal and header contracts to object and object-list editors', () => {
    const domainScores = requireField(getSignalFieldSchema('domain'), 'model_scores')
    expect(domainScores.type).toBe('object[]')
    expect(domainScores.fields?.map((field) => field.key)).toEqual([
      'model',
      'score',
      'use_reasoning',
    ])

    const structureFeature = requireField(getSignalFieldSchema('structure'), 'feature')
    const source = requireField(structureFeature.fields || [], 'source')
    expect(source.type).toBe('object')
    expect(requireField(source.fields || [], 'sequences').type).toBe('string[][]')
    expect(requireField(getSignalFieldSchema('complexity'), 'composer').type).toBe('rule')
    expect(requireField(getSignalFieldSchema('authz'), 'subjects').type).toBe('object[]')
    expect(requireField(getSignalFieldSchema('kb'), 'target').type).toBe('object')

    const headerMutation = getPluginFieldSchema('header_mutation')
    expect(requireField(headerMutation, 'add').type).toBe('object[]')
    expect(requireField(headerMutation, 'update').type).toBe('object[]')
    expect(requireField(headerMutation, 'delete').type).toBe('string[]')
  })
})
