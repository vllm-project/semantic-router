import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import * as sharedSchemas from '@/lib/dslSchemas'
import { getProjectionFieldSchema } from '@/lib/projectionSchemas'
import { projectionFieldsFromRouterSchema, routerConfigFieldAtPath } from '@/lib/routerConfigSchema'
import { requiredFieldErrors } from '@/lib/schemaFieldValidation'
import {
  DSL_MODEL_FIELD_KEYS,
  algorithmFieldSchema,
  fieldProblems,
  modelFieldSchema,
  pluginFieldSchema,
  projectionFieldSchema,
  projectionScoreToFields,
  signalFieldSchema,
} from './editorSchemas'

const read = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8')
const keys = (fields: ReadonlyArray<{ key: string }>) => fields.map((field) => field.key)

// One base for the Builder and the config pages: the same schema modules, the
// same field renderer and the same validators, never a Builder-only copy.
describe('the config editor shares its schema base with the config pages', () => {
  it('reads signal, plugin and algorithm schemas from the shared module', () => {
    expect(signalFieldSchema).toBe(sharedSchemas.getSignalFieldSchema)
    expect(pluginFieldSchema).toBe(sharedSchemas.getPluginFieldSchema)
    expect(algorithmFieldSchema).toBe(sharedSchemas.getAlgorithmFieldSchema)
  })

  it('validates required fields with the shared validator', () => {
    expect(fieldProblems).toBe(requiredFieldErrors)
  })

  it('edits projections with the Projections page schema, minus the header name', () => {
    for (const [kind, collection] of [
      ['partition', 'partitions'],
      ['score', 'scores'],
      ['mapping', 'mappings'],
    ] as const) {
      const page = getProjectionFieldSchema(collection).filter((field) => field.key !== 'name')
      expect(projectionFieldSchema(kind)).toEqual(page)
      expect(keys(projectionFieldSchema(kind))).toEqual(
        keys(projectionFieldsFromRouterSchema(collection)).filter((key) => key !== 'name'),
      )
    }
  })

  it('edits MODEL blocks with the provider model contract, narrowed to the DSL fields', () => {
    const contract = routerConfigFieldAtPath(['providers', 'models'])?.fields ?? []
    expect(keys(modelFieldSchema())).toEqual([...DSL_MODEL_FIELD_KEYS])
    for (const field of modelFieldSchema()) {
      expect(contract.find((candidate) => candidate.key === field.key)).toEqual(field)
    }
  })

  it('renders every entity form with the shared schema field editor', () => {
    for (const source of ['./EditorForms.tsx', './EditorRouteForm.tsx']) {
      const contents = read(source)
      expect(contents).toContain("from '../ConfigPageSchemaFieldsEditor'")
      expect(contents).not.toMatch(
        /builderPage(FieldControls|StructuredFieldControls|FormPrimitives)/,
      )
    }
  })

  it('edits route conditions with the Decisions page rule editor and writer', () => {
    expect(read('./EditorRouteForm.tsx')).toContain(
      "import { RuleTreeEditor } from '../ConfigPageDecisionRulesEditor'",
    )
    expect(read('./editorRules.ts')).toContain("from '../configPageDecisionRuleSupport'")
  })

  it('shows router settings with the Router settings schema, fields and validator', () => {
    const contents = read('./EditorRouterSettings.tsx')
    expect(contents).toContain("from '../configPageRouterSettingsModel'")
    expect(contents).toContain("from '../ConfigPageRouterConfigSectionFields'")
    expect(contents).toContain("from '../configPageRouterSettingsValidation'")
    expect(contents).toContain('readOnly: true')
  })

  it('keeps no Builder-only global settings editor or schema renderer', () => {
    const pages = readdirSync(new URL('..', import.meta.url), { withFileTypes: true }).map(
      (entry) => entry.name,
    )
    expect(pages.filter((file) => file.startsWith('builderPageGlobalSettings'))).toEqual([])
    expect(pages).not.toContain('builderPageSharedDslEditors.tsx')
    expect(pages).not.toContain('builderPageGenericFieldsEditor.tsx')
  })
})

describe('projection fields of the AST', () => {
  it('writes score inputs with the canonical keys', () => {
    expect(
      projectionScoreToFields({
        name: 'difficulty',
        method: 'weighted_sum',
        inputs: [
          { signalType: 'keyword', signalName: 'hard', weight: 0.6, valueSource: 'confidence' },
          { signalType: 'context', signalName: '', weight: 0.4, match: 1, miss: 0 },
        ],
        pos: { Line: 1, Column: 1 },
      }),
    ).toEqual({
      method: 'weighted_sum',
      inputs: [
        { type: 'keyword', name: 'hard', weight: 0.6, value_source: 'confidence' },
        { type: 'context', weight: 0.4, match: 1, miss: 0 },
      ],
    })
  })
})
