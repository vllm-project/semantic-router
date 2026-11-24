export interface ConfigSchemaIndexSection {
  path: string
  title: string
  description?: string
  required: boolean
  href: string
}

export interface ConfigSchemaSurfaceIndex {
  count: number
  names: string[]
  href_template: string
}

export interface ConfigSchemaIndex {
  contract_version: string
  config_version: string
  schema_id: string
  schema_etag?: string
  default_view: string
  sections: ConfigSchemaIndexSection[]
  surfaces: Record<string, ConfigSchemaSurfaceIndex>
}

export interface ConfigSchemaNode {
  $ref?: string
  type?: string | string[]
  title?: string
  description?: string
  default?: unknown
  const?: unknown
  enum?: unknown[]
  minimum?: number
  maximum?: number
  properties?: Record<string, ConfigSchemaNode>
  required?: string[]
  items?: ConfigSchemaNode
  oneOf?: ConfigSchemaNode[]
  anyOf?: ConfigSchemaNode[]
  $defs?: Record<string, ConfigSchemaNode>
  'x-vllm-sr-view'?: {
    view?: string
    path?: string
    kind?: string
    name?: string
    detail?: string
  }
  'x-vllm-sr-surface'?: { display_name?: string; description?: string }
  /** A section summary (the compact view) names its shape and child fields instead of properties. */
  shape?: string
  fields?: ConfigSchemaSummaryField[]
  expanded_href?: string
}

export interface ConfigSchemaSummaryField {
  name: string
  path: string
  type: string
  required?: boolean
  description?: string
  href?: string
}

export interface ConfigSchemaField {
  name: string
  type: string
  required: boolean
  description?: string
  details: string[]
  /** The field's schema, for its nested fields and examples. */
  node?: ConfigSchemaNode
  /** A section summary's child path, which opens as its own section. */
  path?: string
}

export interface ConfigSchemaStructure {
  type: string
  fieldsLabel: string
  description: string
}

interface FilteredConfigSchemaIndex {
  sections: ConfigSchemaIndexSection[]
  surfaces: Record<string, ConfigSchemaSurfaceIndex>
}

export function filterConfigSchemaIndex(
  index: ConfigSchemaIndex,
  query: string,
): FilteredConfigSchemaIndex {
  const normalized = query.trim().toLowerCase()
  if (!normalized) return { sections: index.sections, surfaces: index.surfaces }

  const sections = index.sections.filter((section) =>
    [section.path, section.title, section.description]
      .filter(Boolean)
      .some((value) => value!.toLowerCase().includes(normalized)),
  )
  const surfaces: Record<string, ConfigSchemaSurfaceIndex> = {}
  for (const [kind, catalog] of Object.entries(index.surfaces)) {
    const names = kind.toLowerCase().includes(normalized)
      ? catalog.names
      : catalog.names.filter((name) => name.toLowerCase().includes(normalized))
    if (names.length > 0) surfaces[kind] = { ...catalog, names }
  }
  return { sections, surfaces }
}

function isSummary(document: ConfigSchemaNode): boolean {
  return Array.isArray(document.fields) && !document.properties && !document.$ref
}

export function configSchemaFields(document: ConfigSchemaNode): ConfigSchemaField[] {
  if (isSummary(document)) {
    return (document.fields ?? []).map((field) => ({
      name: field.name,
      type: field.type,
      required: Boolean(field.required),
      description: field.description,
      details: [],
      path: field.path,
    }))
  }
  const root = concreteNode(document, document)
  const fieldRoot = root.type === 'array' && root.items ? concreteNode(document, root.items) : root
  const properties = Object.entries(fieldRoot.properties ?? {})
  if (properties.length === 0) {
    if (fieldRoot.type === 'object' || (!fieldRoot.type && fieldRoot.const === undefined)) return []
    return [fieldFromNode(document, root.type === 'array' ? 'item' : 'value', fieldRoot, true)]
  }

  const required = new Set(fieldRoot.required ?? [])
  return properties.map(([name, node]) => fieldFromNode(document, name, node, required.has(name)))
}

export function configSchemaStructure(document: ConfigSchemaNode): ConfigSchemaStructure {
  if (isSummary(document)) {
    const shape = document.shape ?? 'object'
    return {
      type: shape,
      fieldsLabel: shape.startsWith('array') ? 'Item fields' : 'Fields',
      description: 'Open a field to see its own fields.',
    }
  }
  const root = concreteNode(document, document)
  if (root.type === 'array') {
    return {
      type: schemaType(document, root),
      fieldsLabel: 'Item fields',
      description: 'Each list item uses the fields shown below.',
    }
  }
  if (root.type === 'object' || root.properties) {
    return {
      type: 'object',
      fieldsLabel: 'Fields',
      description: 'This object accepts the fields shown below.',
    }
  }
  return {
    type: schemaType(document, root),
    fieldsLabel: 'Value',
    description: 'This section is a single configuration value.',
  }
}

function fieldFromNode(
  document: ConfigSchemaNode,
  name: string,
  node: ConfigSchemaNode,
  required: boolean,
): ConfigSchemaField {
  const concrete = concreteNode(document, node)
  const details: string[] = []
  if (concrete.const !== undefined) details.push(`fixed: ${formatValue(concrete.const)}`)
  if (concrete.default !== undefined) details.push(`default: ${formatValue(concrete.default)}`)
  if (concrete.enum?.length) details.push(`values: ${concrete.enum.map(formatValue).join(', ')}`)
  if (concrete.minimum !== undefined) details.push(`min: ${concrete.minimum}`)
  if (concrete.maximum !== undefined) details.push(`max: ${concrete.maximum}`)
  return {
    name,
    type: schemaType(document, node),
    required,
    description: concrete.description,
    details,
    node,
  }
}

export function schemaType(document: ConfigSchemaNode, node: ConfigSchemaNode): string {
  const concrete = concreteNode(document, node)
  if (Array.isArray(concrete.type)) return concrete.type.join(' | ')
  if (concrete.type === 'array') {
    return `array<${concrete.items ? schemaType(document, concrete.items) : 'unknown'}>`
  }
  if (concrete.type) return concrete.type
  if (concrete.const !== undefined) return typeof concrete.const
  return 'value'
}

export function focusedSchemaTitle(document: ConfigSchemaNode): string {
  const view = document['x-vllm-sr-view']
  const surface = document['x-vllm-sr-surface']
  return surface?.display_name || view?.path || view?.name || document.title || 'Schema detail'
}

function concreteNode(document: ConfigSchemaNode, node: ConfigSchemaNode): ConfigSchemaNode {
  const resolved = resolveNode(document, node)
  const alternatives = resolved.oneOf ?? resolved.anyOf
  if (!alternatives) return resolved
  const concrete = alternatives.find((alternative) => alternative.type !== 'null')
  return concrete ? resolveNode(document, concrete) : resolved
}

function resolveNode(document: ConfigSchemaNode, node: ConfigSchemaNode): ConfigSchemaNode {
  if (!node.$ref) return node
  const prefix = '#/$defs/'
  if (!node.$ref.startsWith(prefix)) return node
  return document.$defs?.[node.$ref.slice(prefix.length)] ?? node
}

function formatValue(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value)
}

/** The fields one level below a field: an object's properties or a list item's properties. */
export function configSchemaChildFields(
  document: ConfigSchemaNode,
  node: ConfigSchemaNode | undefined,
  depth = 0,
): ConfigSchemaField[] {
  if (!node || depth > 6) return []
  const concrete = concreteNode(document, node)
  const target =
    concrete.type === 'array' && concrete.items ? concreteNode(document, concrete.items) : concrete
  const required = new Set(target.required ?? [])
  return Object.entries(target.properties ?? {}).map(([name, child]) =>
    fieldFromNode(document, name, child, required.has(name)),
  )
}

function sampleValue(document: ConfigSchemaNode, node: ConfigSchemaNode, depth: number): unknown {
  const concrete = concreteNode(document, node)
  if (concrete.default !== undefined) return concrete.default
  if (concrete.const !== undefined) return concrete.const
  if (concrete.enum?.length) return concrete.enum[0]
  const type = Array.isArray(concrete.type)
    ? concrete.type.find((item) => item !== 'null')
    : concrete.type
  if (type === 'array') {
    if (!concrete.items || depth > 3) return []
    const item = sampleValue(document, concrete.items, depth + 1)
    return item === undefined ? [] : [item]
  }
  if (type === 'object' || concrete.properties) {
    if (depth > 3) return {}
    const required = new Set(concrete.required ?? [])
    const entries = Object.entries(concrete.properties ?? {})
    const chosen = entries.filter(([name]) => required.has(name))
    const shown = (chosen.length ? chosen : entries).slice(0, 6)
    return Object.fromEntries(
      shown.map(([name, child]) => [name, sampleValue(document, child, depth + 1)]),
    )
  }
  if (type === 'integer' || type === 'number') return concrete.minimum ?? 0
  if (type === 'boolean') return false
  if (type === 'string') return ''
  return null
}

function yamlKey(key: string): string {
  return /^[A-Za-z_][\w.-]*$/.test(key) ? key : JSON.stringify(key)
}

function yamlScalar(value: unknown): string {
  if (typeof value === 'string')
    return value === '' || /[:#{}[\],&*!|>'"%@`]|^\s|\s$|^(true|false|null|yes|no|\d)/i.test(value)
      ? JSON.stringify(value)
      : value
  return value === null || value === undefined ? 'null' : String(value)
}

function yamlLines(value: unknown, indent: string): string[] {
  if (Array.isArray(value)) {
    if (value.length === 0) return [`${indent}[]`]
    return value.flatMap((item) => {
      if (item && typeof item === 'object' && !Array.isArray(item) && Object.keys(item).length) {
        const [first, ...rest] = yamlLines(item, `${indent}  `)
        return [`${indent}- ${first.trimStart()}`, ...rest]
      }
      return [`${indent}- ${yamlScalar(item)}`]
    })
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    if (entries.length === 0) return [`${indent}{}`]
    return entries.flatMap(([key, child]) => {
      if (
        child &&
        typeof child === 'object' &&
        (Array.isArray(child) ? child.length : Object.keys(child).length)
      )
        return [`${indent}${yamlKey(key)}:`, ...yamlLines(child, `${indent}  `)]
      if (Array.isArray(child)) return [`${indent}${yamlKey(key)}: []`]
      if (child && typeof child === 'object') return [`${indent}${yamlKey(key)}: {}`]
      return [`${indent}${yamlKey(key)}: ${yamlScalar(child)}`]
    })
  }
  return [`${indent}${yamlScalar(value)}`]
}

/**
 * A YAML starting point for the focused schema: required fields first, with
 * defaults, the first allowed value, or an empty value of the right type.
 */
function summaryValue(type: string): unknown {
  if (type.startsWith('array')) return []
  if (type === 'object' || type.startsWith('map')) return {}
  if (type === 'integer' || type === 'number') return 0
  if (type === 'boolean') return false
  return ''
}

export function configSchemaExample(document: ConfigSchemaNode): string {
  const view = document['x-vllm-sr-view']
  const value = isSummary(document)
    ? Object.fromEntries(
        (document.fields ?? []).map((field) => [field.name, summaryValue(field.type)]),
      )
    : sampleValue(document, document, 0)
  if (view?.view === 'section' && view.path) {
    const path = view.path.split('.')
    let nested: unknown = value
    for (let index = path.length - 1; index >= 0; index -= 1) nested = { [path[index]]: nested }
    return yamlLines(nested, '').join('\n')
  }
  const header = view?.kind && view?.name ? [`# ${view.kind}: ${view.name}`] : []
  return [...header, ...yamlLines(value, '')].join('\n')
}
