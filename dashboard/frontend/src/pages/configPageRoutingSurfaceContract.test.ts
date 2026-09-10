import { describe, expect, it } from 'vitest'

import {
  ALGORITHM_TYPES,
  DECISION_SIGNAL_TYPES,
  PLUGIN_TYPES,
  ROUTER_CONFIG_EXTENSION,
  ROUTER_CONFIG_SCHEMA,
  SIGNAL_TYPES,
} from '../generated/routerConfigContract'
import { getAlgorithmFieldSchema } from '../lib/dslAlgorithmSchemas'
import type { FieldSchema } from '../lib/dslSchemaTypes'
import { getPluginFieldSchema, getSignalFieldSchema } from '../lib/dslSchemas'
import {
  algorithmFieldsFromRouterSchema,
  pluginFieldsFromRouterSchema,
  signalFieldsFromRouterSchema,
} from '../lib/routerConfigSchema'
import { PROJECTION_INPUT_TYPES } from './configPageProjectionFormSupport'
import { SIGNAL_CATALOG } from './configPageSignalCatalog'

function expectSameFieldInventory(rendered: FieldSchema[], generated: FieldSchema[]) {
  expect(rendered.map((field) => field.key)).toEqual(generated.map((field) => field.key))
  for (const generatedField of generated) {
    const renderedField = rendered.find((field) => field.key === generatedField.key)
    expect(renderedField).toBeDefined()
    if (generatedField.fields) {
      expectSameFieldInventory(renderedField?.fields ?? [], generatedField.fields)
    }
  }
}

describe('generated Dashboard routing contract', () => {
  it('uses one generated inventory for every routing surface', () => {
    expect(SIGNAL_CATALOG.map((entry) => entry.type)).toEqual(SIGNAL_TYPES)
    expect(PROJECTION_INPUT_TYPES).toEqual(ROUTER_CONFIG_EXTENSION.projection_input_types)
    expect(ROUTER_CONFIG_EXTENSION.algorithms.map((entry) => entry.type)).toEqual(ALGORITHM_TYPES)
    expect(ROUTER_CONFIG_EXTENSION.plugins.map((entry) => entry.type)).toEqual(PLUGIN_TYPES)
    expect(DECISION_SIGNAL_TYPES).toContain('projection')
    expect(DECISION_SIGNAL_TYPES).not.toContain('hallucination')
  })

  it('resolves every generated field schema used by management forms', () => {
    for (const signal of SIGNAL_TYPES)
      expect(getSignalFieldSchema(signal).length).toBeGreaterThan(0)
    for (const algorithm of ALGORITHM_TYPES)
      expect(getAlgorithmFieldSchema(algorithm)).toBeDefined()
    for (const plugin of PLUGIN_TYPES)
      expect(getPluginFieldSchema(plugin).length).toBeGreaterThan(0)
  })

  it('does not let presentation metadata invent Router configuration fields', () => {
    for (const signal of SIGNAL_TYPES) {
      expectSameFieldInventory(getSignalFieldSchema(signal), signalFieldsFromRouterSchema(signal))
    }
    for (const algorithm of ALGORITHM_TYPES) {
      expectSameFieldInventory(
        getAlgorithmFieldSchema(algorithm),
        algorithmFieldsFromRouterSchema(algorithm),
      )
    }
    for (const plugin of PLUGIN_TYPES) {
      expectSameFieldInventory(getPluginFieldSchema(plugin), pluginFieldsFromRouterSchema(plugin))
    }
  })

  it('includes the product setup surface in the canonical document', () => {
    expect(ROUTER_CONFIG_SCHEMA.properties).toHaveProperty('setup')
  })
})
