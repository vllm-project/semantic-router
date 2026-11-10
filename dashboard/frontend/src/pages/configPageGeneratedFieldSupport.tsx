import type { FieldConfig } from '../components/EditModal'
import { routerConfigFieldAtPath, routerConfigFieldsAtPath } from '../lib/routerConfigSchema'
import type { FieldSchema } from '../lib/dslSchemas'
import { FieldEditor } from './builderPageFormPrimitives'

function schemaFieldConfig(field: FieldSchema, name = field.key): FieldConfig {
  return {
    name,
    label: field.label,
    required: field.required,
    description: field.description,
    type: 'custom',
    customRender: (value, onChange) => (
      <FieldEditor schema={{ ...field, key: name }} value={value} onChange={onChange} />
    ),
  }
}

function withoutNestedFields(field: FieldSchema, omitted: ReadonlySet<string>): FieldSchema {
  if (!field.fields) return field
  return {
    ...field,
    fields: field.fields
      .filter((child) => !omitted.has(child.key))
      .map((child) => withoutNestedFields(child, omitted)),
  }
}

export function mergeGeneratedRouterFields(
  path: readonly string[],
  curated: FieldConfig[],
  options: { omit?: readonly string[]; omitNested?: readonly string[] } = {},
): FieldConfig[] {
  const represented = new Set([...curated.map((field) => field.name), ...(options.omit ?? [])])
  const nested = new Set(options.omitNested ?? [])
  const generated = routerConfigFieldsAtPath(path)
    .filter((field) => !represented.has(field.key))
    .map((field) => schemaFieldConfig(withoutNestedFields(field, nested)))
  return [...curated, ...generated]
}

export function generatedRouterValueField(
  path: readonly string[],
  name: string,
  label?: string,
): FieldConfig {
  const generated = routerConfigFieldAtPath(path)
  if (!generated) {
    throw new Error(`Missing generated Router configuration schema for ${path.join('.')}`)
  }
  return schemaFieldConfig({ ...generated, label: label ?? generated.label }, name)
}
