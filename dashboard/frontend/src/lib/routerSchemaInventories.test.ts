import { describe, expect, it } from 'vitest'

import { requiredStructuredFieldErrors } from '../pages/builderPageStructuredFieldSupport'
import {
  DECISION_ACTION_SCHEMA,
  DECISION_ADAPTATIONS_SCHEMA,
  DECISION_FALLBACK_SCHEMA,
  DECISION_IDENTITY_SCHEMA,
  DECISION_MODEL_REF_SCHEMA,
  DECISION_OUTPUT_CONTRACT_SCHEMA,
  DECISION_RELIABILITY_SCHEMA,
  DECISION_RULE_ON_UNKNOWN_OPTIONS,
} from './decisionSchemas'
import * as builderSchemas from './dslMutations'
import * as sharedSchemas from './dslSchemas'
import type { FieldSchema } from './dslSchemaTypes'
import { getProjectionFieldSchema } from './projectionSchemas'
import { projectionFieldsFromRouterSchema, routerConfigFieldsForRef } from './routerConfigSchema'
import { requiredFieldErrors } from './schemaFieldValidation'

function keys(fields: readonly FieldSchema[]): string[] {
  return fields.map((field) => field.key)
}

function expectWithinContract(rendered: readonly FieldSchema[], generated: readonly FieldSchema[]) {
  const allowed = new Set(keys(generated))
  expect(keys(rendered).filter((key) => !allowed.has(key))).toEqual([])
  for (const field of rendered) {
    const source = generated.find((candidate) => candidate.key === field.key)
    if (field.fields && source?.fields) expectWithinContract(field.fields, source.fields)
  }
}

describe('one inventory for config managers and the Builder', () => {
  it('serves signal, plugin and algorithm schemas from the same module to both', () => {
    expect(builderSchemas.getSignalFieldSchema).toBe(sharedSchemas.getSignalFieldSchema)
    expect(builderSchemas.getPluginFieldSchema).toBe(sharedSchemas.getPluginFieldSchema)
    expect(builderSchemas.getAlgorithmFieldSchema).toBe(sharedSchemas.getAlgorithmFieldSchema)
  })

  it('keeps decision schemas to the fields the Router contract declares', () => {
    const cases: Array<[readonly FieldSchema[], string]> = [
      [DECISION_ACTION_SCHEMA, '#/$defs/DecisionAction'],
      [DECISION_ADAPTATIONS_SCHEMA, '#/$defs/DecisionAdaptationsConfig'],
      [DECISION_RELIABILITY_SCHEMA, '#/$defs/DecisionReliability'],
      [DECISION_FALLBACK_SCHEMA, '#/$defs/FallbackOverride'],
      [DECISION_OUTPUT_CONTRACT_SCHEMA, '#/$defs/OutputContractSpec'],
      [DECISION_MODEL_REF_SCHEMA, '#/$defs/ModelRef'],
      [DECISION_IDENTITY_SCHEMA, '#/$defs/Decision'],
    ]
    for (const [schema, ref] of cases) expectWithinContract(schema, routerConfigFieldsForRef(ref))
    expect(keys(DECISION_MODEL_REF_SCHEMA)).toEqual(
      keys(routerConfigFieldsForRef('#/$defs/ModelRef')),
    )
    expect(DECISION_RULE_ON_UNKNOWN_OPTIONS).toEqual(['no_match', 'match', 'fail_request'])
  })

  it('edits every projection field the contract declares, and no other', () => {
    for (const collection of ['partitions', 'scores', 'mappings'] as const) {
      const generated = projectionFieldsFromRouterSchema(collection)
      expect(keys(getProjectionFieldSchema(collection))).toEqual(keys(generated))
      expectWithinContract(getProjectionFieldSchema(collection), generated)
    }
  })

  it('validates required fields with one implementation', () => {
    const schema: FieldSchema[] = [
      { key: 'name', label: 'Role Name', type: 'string', required: true },
      { key: 'models', label: 'Models', type: 'string[]', required: true },
      {
        key: 'planner',
        label: 'Planner',
        type: 'object',
        fields: [{ key: 'model', label: 'Planner Model', type: 'string', required: true }],
      },
    ]
    const value = { name: '', models: [], planner: {} }
    expect(requiredStructuredFieldErrors(schema, value)).toEqual(requiredFieldErrors(schema, value))
    expect(requiredFieldErrors(schema, value)).toEqual([
      'Role Name is required.',
      'Models is required.',
      'Planner > Planner Model is required.',
    ])
  })
})
