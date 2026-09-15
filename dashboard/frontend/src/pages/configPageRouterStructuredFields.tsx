import type { FieldConfig } from '../components/EditModal'
import { routerConfigFieldAtPath } from '../lib/routerConfigSchema'
import type { FieldSchema } from '../lib/dslSchemas'
import ConfigPageRouterStructuredEditor from './ConfigPageRouterStructuredEditor'
import {
  ROUTER_STRUCTURED_FIELDS,
  type RouterStructuredFieldDefinition,
  type RouterStructuredSchema,
} from './configPageRouterStructuredSchema'
import {
  routerStructuredFieldSchemaPath,
  type RouterSystemKey,
} from './configPageRouterSectionCatalog'

function generatedStructuredSchema(field: FieldSchema): RouterStructuredSchema {
  const common = {
    label: field.label,
    description: field.description,
    required: field.required,
    placeholder: field.placeholder,
    min: field.min,
    max: field.max,
  }
  switch (field.type) {
    case 'string':
      return { ...common, kind: 'string' }
    case 'number':
      return { ...common, kind: 'number' }
    case 'boolean':
      return { ...common, kind: 'boolean' }
    case 'select':
      return { ...common, kind: 'select', options: field.options }
    case 'string[]':
      return { ...common, kind: 'string-list', defaultValue: [] }
    case 'number[]':
      return { ...common, kind: 'number-list', defaultValue: [] }
    case 'key-value':
      return { ...common, kind: 'string-map', defaultValue: {} }
    case 'object':
      return {
        ...common,
        kind: 'object',
        defaultValue: {},
        fields: Object.fromEntries(
          (field.fields ?? []).map((child) => [child.key, generatedStructuredSchema(child)]),
        ),
      }
    case 'object[]':
      return {
        ...common,
        kind: 'object-list',
        defaultValue: [],
        addLabel: field.addLabel,
        emptyLabel: field.emptyLabel,
        itemLabelKey: field.itemLabelKey,
        item: {
          kind: 'object',
          label: field.itemLabel ?? field.label,
          defaultValue: {},
          fields: Object.fromEntries(
            (field.fields ?? []).map((child) => [child.key, generatedStructuredSchema(child)]),
          ),
        },
      }
    default:
      return { ...common, kind: 'json' }
  }
}

function mergeStructuredSchema(
  generated: RouterStructuredSchema,
  curated: RouterStructuredSchema,
): RouterStructuredSchema {
  const merged: RouterStructuredSchema = {
    ...generated,
    ...curated,
    description: curated.description ?? generated.description,
    required: Boolean(generated.required || curated.required),
    options: generated.options?.length ? generated.options : curated.options,
    min: generated.min ?? curated.min,
    max: generated.max ?? curated.max,
  }
  if (generated.kind === 'object' && curated.kind === 'object') {
    const generatedFields = generated.fields ?? {}
    const curatedFields = curated.fields ?? {}
    merged.fields = { ...generatedFields }
    for (const [name, schema] of Object.entries(curatedFields)) {
      merged.fields[name] = generatedFields[name]
        ? mergeStructuredSchema(generatedFields[name], schema)
        : schema
    }
  }
  if (
    generated.kind === 'object-list' &&
    curated.kind === 'object-list' &&
    generated.item &&
    curated.item
  ) {
    merged.item = mergeStructuredSchema(generated.item, curated.item)
  }
  return merged
}

export function getRouterStructuredFieldDefinition(
  key: RouterSystemKey,
  name: string,
): RouterStructuredFieldDefinition {
  const definition = ROUTER_STRUCTURED_FIELDS[key]?.[name]
  if (!definition) {
    throw new Error(`Missing structured field schema for ${key}.${name}`)
  }
  const generated = routerConfigFieldAtPath(routerStructuredFieldSchemaPath(key, name))
  if (!generated) return definition
  return {
    ...definition,
    description: definition.description || generated.description || '',
    schema: mergeStructuredSchema(generatedStructuredSchema(generated), definition.schema),
  }
}

export function routerStructuredField(key: RouterSystemKey, name: string): FieldConfig {
  const definition = getRouterStructuredFieldDefinition(key, name)
  return {
    name,
    label: definition.label,
    type: 'custom',
    description: definition.description,
    required: definition.schema.required,
    customRender: (value, onChange) => (
      <ConfigPageRouterStructuredEditor
        schema={definition.schema}
        value={value}
        onChange={onChange}
      />
    ),
  }
}
