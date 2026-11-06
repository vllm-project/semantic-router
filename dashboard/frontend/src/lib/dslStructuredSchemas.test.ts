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
  it('uses JSON controls only for recursive or deliberately open payloads', () => {
    const schemas = [
      ...ALGORITHM_TYPES.flatMap((type) => getAlgorithmFieldSchema(type)),
      ...SIGNAL_TYPES.flatMap((type) => getSignalFieldSchema(type)),
      ...PLUGIN_TYPES.flatMap((type) => getPluginFieldSchema(type)),
    ]

    // The recursive rule tree: response_guard's rules[].when. The open
    // payload: rag's backend_config.
    expect(
      flattenSchema(schemas)
        .filter((field) => field.type === 'json')
        .map((field) => field.key),
    ).toEqual(['conditions', 'backend_config'])
  })

  it('describes workflow and multi-factor structures with typed nested schemas', () => {
    const workflows = getAlgorithmFieldSchema('workflows')
    expect(requireField(workflows, 'minimum_candidates')).toMatchObject({ type: 'number', min: 1 })
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
    const quality = requireField(multiFactor, 'quality')
    expect(quality.type).toBe('object')
    expect(quality.fields?.map((field) => field.key)).toEqual([
      'index',
      'on_missing',
      'min_coverage',
      'min_score',
    ])
    expect(requireField(quality.fields || [], 'on_missing').options).toEqual([
      'exclude',
      'disable_quality',
    ])

    const prompt = getAlgorithmFieldSchema('prompt')
    const promptConfig = requireField(prompt, 'prompt')
    expect(promptConfig.type).toBe('object')
    expect(promptConfig.fields?.map((field) => field.key)).toEqual([
      'model',
      'instructions',
      'timeout_seconds',
    ])
    expect(prompt.some((field) => field.key === 'model')).toBe(false)
  })

  it('maps stable signal and header contracts to object and object-list editors', () => {
    const piiInput = requireField(getSignalFieldSchema('pii'), 'input')
    expect(piiInput.type).toBe('object')
    expect(piiInput.fields?.map((field) => field.key)).toEqual(
      expect.arrayContaining(['history', 'tool_results']),
    )

    const domainScores = requireField(getSignalFieldSchema('domain'), 'model_scores')
    expect(domainScores.type).toBe('object[]')
    expect(domainScores.fields?.map((field) => field.key)).toEqual([
      'model',
      'score',
      'use_reasoning',
    ])

    for (const measure of ['count', 'density']) {
      const predicate = requireField(getSignalFieldSchema('keyword'), measure)
      expect(predicate.type).toBe('object')
      expect(predicate.fields?.map((field) => field.key)).toEqual(['gt', 'gte', 'lt', 'lte'])
    }
    expect(requireField(getSignalFieldSchema('authz'), 'subjects').type).toBe('object[]')

    const metadataPredicate = requireField(getSignalFieldSchema('metadata'), 'predicate')
    expect(metadataPredicate.fields?.map((field) => field.key)).toEqual(['equals', 'in', 'exists'])

    const conversationFeature = requireField(getSignalFieldSchema('conversation'), 'feature')
    const conversationSource = requireField(conversationFeature.fields || [], 'source')
    expect(requireField(conversationSource.fields || [], 'type').options).toEqual(
      expect.arrayContaining(['tool_choice_required', 'tool_choice_none', 'flow_tool_state']),
    )

    const headerMutation = getPluginFieldSchema('header_mutation')
    expect(requireField(headerMutation, 'add').type).toBe('object[]')
    expect(requireField(headerMutation, 'update').type).toBe('object[]')
    expect(requireField(headerMutation, 'delete').type).toBe('string[]')

    const tools = getPluginFieldSchema('tools')
    expect(requireField(tools, 'strip_tool_history').type).toBe('boolean')
    const dynamicRetrieval = requireField(tools, 'dynamic_retrieval')
    expect(dynamicRetrieval.type).toBe('object')
    expect(requireField(dynamicRetrieval.fields || [], 'history_window').type).toBe('number')
    expect(requireField(dynamicRetrieval.fields || [], 'weights').type).toBe('object')
  })

  it('offers every conversation source type and non_user as a role', () => {
    const conversationFeature = requireField(getSignalFieldSchema('conversation'), 'feature')
    const source = requireField(conversationFeature.fields || [], 'source')
    const sourceType = requireField(source.fields || [], 'type')
    const role = requireField(source.fields || [], 'role')

    expect(sourceType.options).toEqual([
      'message',
      'tool_definition',
      'tool_choice_required',
      'tool_choice_none',
      'assistant_tool_call',
      'assistant_tool_cycle',
      'active_tool_loop',
      'flow_tool_state',
    ])
    expect(role.options).toContain('non_user')
  })
})
