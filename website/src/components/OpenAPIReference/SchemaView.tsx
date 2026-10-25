import React, { useState } from 'react'

import styles from './styles.module.css'

export interface OpenAPISchema {
  $ref?: string
  type?: string
  format?: string
  description?: string
  nullable?: boolean
  enum?: Array<string | number | boolean | null>
  properties?: Record<string, OpenAPISchema>
  required?: string[]
  items?: OpenAPISchema
  additionalProperties?: boolean | OpenAPISchema
  oneOf?: OpenAPISchema[]
  anyOf?: OpenAPISchema[]
  allOf?: OpenAPISchema[]
}

type SchemaDefinitions = Record<string, OpenAPISchema>
const maxSchemaDepth = 10
const documentURL = '/openapi/apiserver/apiserver.openapi.json'

function referenceName(reference: string): string {
  return reference.split('/').pop()?.replace(/~1/g, '/').replace(/~0/g, '~') ?? reference
}

function resolveSchema(schema: OpenAPISchema, definitions: SchemaDefinitions): OpenAPISchema {
  if (!schema.$ref?.startsWith('#/components/schemas/')) return schema
  return { ...definitions[referenceName(schema.$ref)], ...schema }
}

export function schemaType(schema?: OpenAPISchema, depth = 0): string {
  if (!schema) return 'unspecified'
  let type = schema.type ?? 'value'
  if (schema.$ref) type = referenceName(schema.$ref)
  else if (schema.oneOf) type = `one of ${schema.oneOf.length} alternatives`
  else if (schema.anyOf) type = `any of ${schema.anyOf.length} alternatives`
  else if (schema.allOf) type = `all of ${schema.allOf.length} schemas`
  else if (schema.type === 'array') type = depth < maxSchemaDepth ? `array<${schemaType(schema.items, depth + 1)}>` : 'array'
  if (schema.format) type += ` · ${schema.format}`
  return schema.nullable ? `${type} | null` : type
}

export function exampleValue(schema: OpenAPISchema | undefined, definitions: SchemaDefinitions, field = 'value', depth = 0, references: string[] = []): unknown {
  if (!schema || depth >= maxSchemaDepth) return {}
  if (schema.$ref && references.includes(schema.$ref)) return {}
  const nextReferences = schema.$ref ? [...references, schema.$ref] : references
  const resolved = resolveSchema(schema, definitions)
  const example = (child: OpenAPISchema | undefined, name = field) => exampleValue(child, definitions, name, depth + 1, nextReferences)
  if (resolved.enum?.length) return resolved.enum[0]
  if (resolved.oneOf?.length) return example(resolved.oneOf[0])
  if (resolved.anyOf?.length) return example(resolved.anyOf[0])
  if (resolved.type === 'array') return [example(resolved.items, 'item')]
  if (resolved.type === 'object' || resolved.properties) {
    const required = new Set(resolved.required ?? [])
    const properties = Object.entries(resolved.properties ?? {})
    let selected = properties.filter(([name]) => required.has(name))
    if (!selected.length && properties.length) {
      const preferredNames = ['text', 'messages', 'query', 'yaml', 'name', 'file_id', 'premise']
      const preferred = preferredNames
        .map(name => properties.find(([candidate]) => candidate === name))
        .find(Boolean)
      selected = [preferred ?? properties[0]]
    }
    return Object.fromEntries(selected.map(([name, child]) => [name, example(child, name)]))
  }
  if (resolved.type === 'boolean') return false
  if (resolved.type === 'integer' || resolved.type === 'number') return 0
  return `<${field}>`
}

interface SchemaViewProps {
  schema?: OpenAPISchema
  name: string
  definitions: SchemaDefinitions
  required?: boolean
  depth?: number
  references?: string[]
}

export default function SchemaView({ schema, name, definitions, required, depth = 0, references = [] }: SchemaViewProps) {
  const [expanded, setExpanded] = useState(depth === 0)
  const resolved = schema ? resolveSchema(schema, definitions) : undefined
  const recursive = !!schema?.$ref && references.includes(schema.$ref)
  const nextReferences = schema?.$ref ? [...references, schema.$ref] : references
  const children = Object.entries(resolved?.properties ?? {}).map(([field, child]) => ({
    name: field, schema: child, required: resolved?.required?.includes(field),
  }))
  if (resolved?.items) children.push({ name: '[item]', schema: resolved.items, required: undefined })
  if (typeof resolved?.additionalProperties === 'object') {
    children.push({ name: '[key]', schema: resolved.additionalProperties, required: undefined })
  }
  for (const [kind, alternatives] of [['oneOf', resolved?.oneOf], ['anyOf', resolved?.anyOf], ['allOf', resolved?.allOf]] as const) {
    alternatives?.forEach((child, index) => children.push({ name: `${kind} · ${index + 1}`, schema: child, required: undefined }))
  }
  const label = (
    <>
      <code>{name}</code>
      <span>{schemaType(resolved)}</span>
      {required ? <b>required</b> : null}
    </>
  )
  const annotations = (
    <>
      {resolved?.description ? <p>{resolved.description}</p> : null}
      {resolved?.enum?.length
        ? (
            <p>
              Allowed values:
              {' '}
              <code>{resolved.enum.map(value => JSON.stringify(value)).join(', ')}</code>
            </p>
          )
        : null}
      {resolved?.additionalProperties === true ? <p>Additional fields are allowed.</p> : null}
      {resolved?.additionalProperties === false ? <p>Additional fields are not allowed.</p> : null}
      {schema?.$ref
        ? (
            <p>
              Reference:
              {' '}
              <code>{schema.$ref}</code>
            </p>
          )
        : null}
    </>
  )
  if (!children.length || recursive || depth >= maxSchemaDepth) {
    return (
      <div className={styles.schemaField}>
        <div className={styles.schemaLabel}>{label}</div>
        {annotations}
        {recursive || depth >= maxSchemaDepth
          ? (
              <p>
                Further nesting is available in the
                {' '}
                <a href={documentURL}>generated JSON</a>
                .
              </p>
            )
          : null}
      </div>
    )
  }
  return (
    <details className={styles.schemaField} open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}>
      <summary>{label}</summary>
      {expanded
        ? (
            <>
              {annotations}
              <div className={styles.schemaChildren}>
                {children.map(child => (
                  <SchemaView key={child.name} {...child} definitions={definitions} depth={depth + 1} references={nextReferences} />
                ))}
              </div>
            </>
          )
        : null}
    </details>
  )
}
