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
  'x-vllm-sr-view'?: { view?: string; path?: string; kind?: string; name?: string }
  'x-vllm-sr-surface'?: { display_name?: string; description?: string }
}

export interface ConfigSchemaField {
  name: string
  type: string
  required: boolean
  description?: string
  details: string[]
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

export function configSchemaFields(document: ConfigSchemaNode): ConfigSchemaField[] {
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
