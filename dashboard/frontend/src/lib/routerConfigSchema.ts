import { ROUTER_CONFIG_EXTENSION, ROUTER_CONFIG_SCHEMA } from '../generated/routerConfigContract'
import type { FieldSchema } from './dslSchemaTypes'

interface JSONSchemaNode {
  $ref?: string
  type?: string
  title?: string
  description?: string
  enum?: unknown[]
  const?: unknown
  properties?: Record<string, JSONSchemaNode>
  required?: string[]
  items?: JSONSchemaNode
  additionalProperties?: boolean | JSONSchemaNode
  anyOf?: JSONSchemaNode[]
  oneOf?: JSONSchemaNode[]
  minimum?: number
  maximum?: number
}

interface RouterConfigDocument extends JSONSchemaNode {
  $defs: Record<string, JSONSchemaNode>
}

const document = ROUTER_CONFIG_SCHEMA as unknown as RouterConfigDocument

function humanize(key: string): string {
  return key
    .split('_')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

export function resolveRouterConfigSchema(schemaOrRef: JSONSchemaNode | string): JSONSchemaNode {
  const schema = typeof schemaOrRef === 'string' ? { $ref: schemaOrRef } : schemaOrRef
  if (!schema.$ref) return schema
  const prefix = '#/$defs/'
  if (!schema.$ref.startsWith(prefix)) {
    throw new Error(`Unsupported Router config schema reference: ${schema.$ref}`)
  }
  const definition = document.$defs[schema.$ref.slice(prefix.length)]
  if (!definition) throw new Error(`Missing Router config schema definition: ${schema.$ref}`)
  return definition
}

function concreteSchema(schema: JSONSchemaNode): JSONSchemaNode {
  const resolved = resolveRouterConfigSchema(schema)
  const branches = resolved.oneOf ?? resolved.anyOf
  if (!branches) return resolved
  const concrete = branches.find((branch) => branch.type !== 'null')
  return concrete ? resolveRouterConfigSchema(concrete) : resolved
}

function arrayFieldType(items: JSONSchemaNode | undefined): FieldSchema['type'] {
  if (!items) return 'json'
  const item = concreteSchema(items)
  if (item.type === 'string') return 'string[]'
  if (item.type === 'number' || item.type === 'integer') return 'number[]'
  if (item.type === 'array' && concreteSchema(item.items ?? {}).type === 'string') {
    return 'string[][]'
  }
  if (item.type === 'object' || item.properties) return 'object[]'
  return 'json'
}

function fieldFromSchema(
  key: string,
  source: JSONSchemaNode,
  required: boolean,
  ancestorRefs: ReadonlySet<string>,
): FieldSchema {
  const nextRefs = new Set(ancestorRefs)
  if (source.$ref) {
    if (nextRefs.has(source.$ref)) {
      return { key, label: humanize(key), type: 'json', required }
    }
    nextRefs.add(source.$ref)
  }
  const schema = concreteSchema(source)
  const label = schema.title || humanize(key)
  const common = {
    key,
    label,
    required,
    description: schema.description,
    min: schema.minimum,
    max: schema.maximum,
  }
  const options = (schema.enum ?? (schema.const === undefined ? [] : [schema.const])).filter(
    (value): value is string => typeof value === 'string',
  )
  if (options.length > 0) return { ...common, type: 'select', options }

  if (schema.type === 'boolean') return { ...common, type: 'boolean' }
  if (schema.type === 'number' || schema.type === 'integer') return { ...common, type: 'number' }
  if (schema.type === 'array') {
    const type = arrayFieldType(schema.items)
    if (type === 'object[]') {
      if (schema.items?.$ref && nextRefs.has(schema.items.$ref)) {
        return { ...common, type: 'json' }
      }
      const item = concreteSchema(schema.items ?? {})
      return {
        ...common,
        type,
        fields: fieldsFromObjectSchema(item, nextRefs),
        addLabel: `Add ${label.toLocaleLowerCase()}`,
        emptyLabel: `No ${label.toLocaleLowerCase()} configured.`,
      }
    }
    return { ...common, type }
  }
  if (schema.type === 'object' || schema.properties) {
    if (schema.properties && Object.keys(schema.properties).length > 0) {
      return { ...common, type: 'object', fields: fieldsFromObjectSchema(schema, nextRefs) }
    }
    if (
      typeof schema.additionalProperties === 'object' &&
      concreteSchema(schema.additionalProperties).type === 'string'
    ) {
      return { ...common, type: 'key-value' }
    }
    return { ...common, type: 'json' }
  }
  return { ...common, type: 'string' }
}

function fieldsFromObjectSchema(
  schemaOrRef: JSONSchemaNode | string,
  ancestorRefs: ReadonlySet<string> = new Set(),
): FieldSchema[] {
  const nextRefs = new Set(ancestorRefs)
  const rootRef = typeof schemaOrRef === 'string' ? schemaOrRef : schemaOrRef.$ref
  if (rootRef) nextRefs.add(rootRef)
  const schema = concreteSchema(
    typeof schemaOrRef === 'string' ? { $ref: schemaOrRef } : schemaOrRef,
  )
  const required = new Set(schema.required ?? [])
  return Object.entries(schema.properties ?? {}).map(([key, property]) =>
    fieldFromSchema(key, property, required.has(key), nextRefs),
  )
}

export function routerConfigFieldsForRef(
  schemaRef: string,
  options: { omit?: readonly string[] } = {},
): FieldSchema[] {
  const omitted = new Set(options.omit ?? [])
  return fieldsFromObjectSchema(schemaRef).filter((field) => !omitted.has(field.key))
}

function schemaAtPath(path: readonly string[]): JSONSchemaNode | undefined {
  let current: JSONSchemaNode = document
  for (const segment of path) {
    const resolved = concreteSchema(current)
    const property = resolved.properties?.[segment]
    if (!property) return undefined
    current = property
  }
  return current
}

export function routerConfigFieldsAtPath(
  path: readonly string[],
  options: { omit?: readonly string[] } = {},
): FieldSchema[] {
  const schema = schemaAtPath(path)
  if (!schema) return []
  const omitted = new Set(options.omit ?? [])
  return fieldsFromObjectSchema(schema).filter((field) => !omitted.has(field.key))
}

export function routerConfigFieldAtPath(path: readonly string[]): FieldSchema | undefined {
  if (path.length === 0) return undefined
  const schema = schemaAtPath(path)
  if (!schema) return undefined
  return fieldFromSchema(path[path.length - 1], schema, false, new Set())
}

export function mergeRouterFieldSchemas(
  generated: FieldSchema[],
  curated: FieldSchema[],
): FieldSchema[] {
  const curatedByKey = new Map(curated.map((field) => [field.key, field]))
  const merged = generated.map((field) => {
    const override = curatedByKey.get(field.key)
    if (!override) return field
    curatedByKey.delete(field.key)
    const fields =
      field.fields && override.fields
        ? mergeRouterFieldSchemas(field.fields, override.fields)
        : (override.fields ?? field.fields)
    return { ...field, ...override, fields }
  })
  return [...merged, ...curatedByKey.values()]
}

export function signalFieldsFromRouterSchema(signalType: string): FieldSchema[] {
  const surface = ROUTER_CONFIG_EXTENSION.signals.find((entry) => entry.type === signalType)
  if (!surface) return []
  return routerConfigFieldsForRef(surface.schema_ref, { omit: ['name'] })
}

export function algorithmFieldsFromRouterSchema(algorithmType: string): FieldSchema[] {
  const surface = ROUTER_CONFIG_EXTENSION.algorithms.find((entry) => entry.type === algorithmType)
  if (!surface) return []
  const payloadKeys = ROUTER_CONFIG_EXTENSION.algorithms.flatMap((entry) =>
    'config_field' in entry && typeof entry.config_field === 'string' ? [entry.config_field] : [],
  )
  const common = routerConfigFieldsForRef('#/$defs/AlgorithmConfig', {
    omit: ['type', ...payloadKeys],
  })
  if (!('schema_ref' in surface) || typeof surface.schema_ref !== 'string') return common
  const payloadFields = routerConfigFieldsForRef(surface.schema_ref)
  if (
    'payload_shape' in surface &&
    surface.payload_shape === 'nested' &&
    'config_field' in surface &&
    typeof surface.config_field === 'string'
  ) {
    return [
      ...common,
      {
        key: surface.config_field,
        label: humanize(surface.config_field),
        type: 'object',
        fields: payloadFields,
      },
    ]
  }
  return [...common, ...payloadFields]
}

export function pluginFieldsFromRouterSchema(pluginType: string): FieldSchema[] {
  const surface = ROUTER_CONFIG_EXTENSION.plugins.find((entry) => entry.type === pluginType)
  return surface ? routerConfigFieldsForRef(surface.schema_ref) : []
}

export function projectionFieldsFromRouterSchema(collection: string): FieldSchema[] {
  const surface = ROUTER_CONFIG_EXTENSION.projections.find(
    (entry) => entry.collection === collection,
  )
  return surface ? routerConfigFieldsForRef(surface.schema_ref) : []
}
