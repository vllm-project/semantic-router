import type { LucideIcon } from 'lucide-react'
import {
  ArrowLeftRight,
  Brain,
  BrainCircuit,
  Boxes,
  Container,
  Cpu,
  Database,
  DoorOpen,
  EyeOff,
  Gauge,
  GraduationCap,
  HeartPulse,
  History,
  KeyRound,
  Layers,
  MessagesSquare,
  Network,
  Puzzle,
  Repeat,
  RotateCcw,
  ServerCog,
  ShieldCheck,
  Telescope,
  Timer,
  Wallet,
  Wrench,
  Zap,
} from 'lucide-react'
import type { FieldConfig } from '../components/EditModal'
import { ROUTER_CONFIG_EXTENSION, ROUTER_CONFIG_SCHEMA } from '../generated/routerConfigContract'
import type { FieldSchema } from '../lib/dslSchemaTypes'
import {
  resolveRouterConfigSchema,
  routerConfigFieldAtPath,
  routerConfigFieldsAtPath,
  routerConfigFieldsForRef,
} from '../lib/routerConfigSchema'
import type { Tone } from '../ui'
import { SECTION_META } from './configPageRouterDefaultsCatalog'
import { curatedFieldsForKey, routerLayerMeta } from './configPageRouterDefaultsSupport'
import {
  CURATED_ROUTER_SECTIONS,
  ROUTER_SETTINGS_CARDS,
  ROUTER_SETTINGS_EXCLUDED_PATHS,
  type RouterSettingsCardPolicy,
  type RouterSettingsLayer,
  type RouterSettingsRoot,
  type RouterSystemKey,
} from './configPageRouterSectionCatalog'
import {
  generatedStructuredSchema,
  getRouterStructuredFieldDefinition,
  mergeStructuredSchema,
} from './configPageRouterStructuredFields'
import {
  ROUTER_STRUCTURED_FIELDS,
  type RouterStructuredSchema,
} from './configPageRouterStructuredSchema'

// Builds the router settings form from the generated Router contract. Curated
// modules only add labels, hints and options; the contract decides which
// fields exist, so a field the contract drops disappears from the form.

export interface RouterSettingsField {
  key: string
  /** Path below `global`. */
  path: string[]
  schema: RouterStructuredSchema
  /** Shown as a percentage while the config stores a 0–1 fraction. */
  percent?: boolean
}

export interface RouterSettingsSection {
  key: string
  /** DOM id; `#global-section-<key>` deep links land here. */
  id: string
  layer: RouterSettingsLayer
  title: string
  description: string
  icon: LucideIcon
  tone: Tone
  /** Dotted paths the card edits, for display. */
  paths: string[]
  roots: readonly RouterSettingsRoot[]
  fields: RouterSettingsField[]
  master?: RouterSettingsField
  custom?: RouterSettingsCardPolicy['custom']
}

interface JSONSchemaNode {
  $ref?: string
  type?: string
  title?: string
  description?: string
  properties?: Record<string, JSONSchemaNode>
  items?: JSONSchemaNode
  enum?: unknown[]
  additionalProperties?: boolean | JSONSchemaNode
  anyOf?: JSONSchemaNode[]
  oneOf?: JSONSchemaNode[]
}

type Resolve = (schema: JSONSchemaNode | string) => JSONSchemaNode
const resolve = resolveRouterConfigSchema as unknown as Resolve
const documentRoot = ROUTER_CONFIG_SCHEMA as unknown as JSONSchemaNode

function concrete(node: JSONSchemaNode): JSONSchemaNode {
  const resolved = resolve(node)
  const branches = resolved.oneOf ?? resolved.anyOf
  if (!branches) return resolved
  const branch = branches.find((candidate) => candidate.type !== 'null')
  return branch ? resolve(branch) : resolved
}

/** The raw contract node at a path; `[]` steps into list items and any other key into map entries. */
export function contractNodeAt(path: readonly string[]): JSONSchemaNode | undefined {
  let node: JSONSchemaNode = documentRoot
  for (const segment of path) {
    const current = concrete(node)
    if (segment === '[]') {
      if (!current.items) return undefined
      node = current.items
    } else if (current.properties?.[segment]) {
      node = current.properties[segment]
    } else if (current.additionalProperties && typeof current.additionalProperties === 'object') {
      node = current.additionalProperties
    } else {
      return undefined
    }
  }
  return node
}

const humanize = (key: string) =>
  key
    .split('_')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')

// A mapping of named entries is plain JSON in the shared FieldSchema; the
// contract still knows the entry schema, so the form can edit entries.
function objectMapSchemaAt(
  path: readonly string[],
  base: RouterStructuredSchema,
): RouterStructuredSchema | undefined {
  const node = contractNodeAt(path)
  if (!node) return undefined
  const current = concrete(node)
  const entry = current.additionalProperties
  if (!entry || typeof entry !== 'object' || current.properties) return undefined
  if (!entry.$ref || !concrete(entry).properties) return undefined
  const fields = routerConfigFieldsForRef(entry.$ref)
  if (fields.length === 0) return undefined
  const item = withObjectMaps(
    {
      kind: 'object',
      label: 'Entry',
      defaultValue: {},
      fields: Object.fromEntries(
        fields.map((field) => [field.key, generatedStructuredSchema(field)]),
      ),
    },
    [...path, '*'],
  )
  return { ...base, kind: 'object-map', item, defaultValue: {} }
}

function withObjectMaps(
  schema: RouterStructuredSchema,
  path: readonly string[],
): RouterStructuredSchema {
  if (schema.kind === 'object' && schema.fields) {
    return {
      ...schema,
      fields: Object.fromEntries(
        Object.entries(schema.fields).map(([key, child]) => [
          key,
          withObjectMaps(child, [...path, key]),
        ]),
      ),
    }
  }
  if (schema.kind === 'object-list' && schema.item) {
    return { ...schema, item: withObjectMaps(schema.item, [...path, '[]']) }
  }
  if (schema.kind === 'json') return objectMapSchemaAt(path, schema) ?? schema
  if (schema.kind === 'string-list' && !schema.options) {
    const items = contractNodeAt([...path, '[]'])
    const allowed = items ? concrete(items).enum : undefined
    const options = allowed?.filter((value): value is string => typeof value === 'string')
    if (options?.length) return { ...schema, options }
  }
  return schema
}

// Curated schemas may name fields the contract no longer has; the contract wins.
function pruneToContract(
  schema: RouterStructuredSchema,
  generated: RouterStructuredSchema,
): RouterStructuredSchema {
  if (
    schema.kind === 'object' &&
    schema.fields &&
    generated.kind === 'object' &&
    generated.fields
  ) {
    const allowed = generated.fields
    return {
      ...schema,
      fields: Object.fromEntries(
        Object.entries(schema.fields)
          .filter(([key]) => key in allowed)
          .map(([key, child]) => [key, pruneToContract(child, allowed[key])]),
      ),
    }
  }
  if (
    schema.kind === 'object-list' &&
    schema.item &&
    generated.kind === 'object-list' &&
    generated.item
  ) {
    return { ...schema, item: pruneToContract(schema.item, generated.item) }
  }
  return schema
}

function legacyKeyFor(path: readonly string[]): RouterSystemKey | undefined {
  const dotted = path.join('.')
  return (Object.keys(CURATED_ROUTER_SECTIONS) as RouterSystemKey[]).find(
    (key) => CURATED_ROUTER_SECTIONS[key].path.join('.') === dotted,
  )
}

/** The curated structured definition of a field, which the shared normalizer validates. */
export function curatedStructuredDefinition(path: readonly string[]) {
  const legacyParent = legacyKeyFor(path.slice(0, -1))
  return legacyParent ? ROUTER_STRUCTURED_FIELDS[legacyParent]?.[path[path.length - 1]] : undefined
}

const curatedFieldCache = new Map<RouterSystemKey, FieldConfig[]>()
function curatedFields(key: RouterSystemKey): FieldConfig[] {
  if (!curatedFieldCache.has(key)) curatedFieldCache.set(key, curatedFieldsForKey(key))
  return curatedFieldCache.get(key) ?? []
}

function overlayCuratedField(
  schema: RouterStructuredSchema,
  curated: FieldConfig | undefined,
): RouterStructuredSchema {
  if (!curated || curated.type === 'custom') return schema
  const options = curated.options?.filter((option) => option !== '')
  return {
    ...schema,
    label: curated.label || schema.label,
    description: curated.description ?? schema.description,
    placeholder: curated.placeholder ?? schema.placeholder,
    required: schema.required || curated.required,
    ...(schema.kind === 'string' && options?.length ? { kind: 'select' as const, options } : {}),
    ...(schema.kind === 'select' && options?.length && !schema.options?.length ? { options } : {}),
  }
}

/** The schema of one field below `global`, from the contract plus any curated overlay. */
export function routerSettingsFieldAt(
  path: readonly string[],
  fromField?: FieldSchema,
): RouterSettingsField | undefined {
  const generatedField = fromField ?? routerConfigFieldAtPath(['global', ...path])
  if (!generatedField) return undefined
  const key = path[path.length - 1]
  const parent = path.slice(0, -1)
  const legacyParent = legacyKeyFor(parent)
  const legacySelf = legacyKeyFor(path)
  const generated = generatedStructuredSchema(generatedField)

  let schema = generated
  if (legacyParent && ROUTER_STRUCTURED_FIELDS[legacyParent]?.[key]) {
    const definition = getRouterStructuredFieldDefinition(legacyParent, key)
    schema = pruneToContract(mergeStructuredSchema(generated, definition.schema), generated)
    schema = {
      ...schema,
      label: definition.label || schema.label,
      description: definition.description || schema.description,
    }
  }
  const curated = legacyParent
    ? curatedFields(legacyParent).find((field) => field.name === key)
    : undefined
  schema = overlayCuratedField(schema, curated)
  if (!schema.description && legacySelf)
    schema = { ...schema, description: SECTION_META[legacySelf].description }
  if (!schema.label) schema = { ...schema, label: humanize(key) }

  return {
    key,
    path: [...path],
    schema: withObjectMaps(schema, ['global', ...path]),
    percent: curated?.type === 'percentage' || undefined,
  }
}

function rootFields(root: RouterSettingsRoot): RouterSettingsField[] {
  if (root.mode === 'field') {
    const field = routerSettingsFieldAt(root.path)
    return field ? [field] : []
  }
  const omitted = new Set(root.omit ?? [])
  return routerConfigFieldsAtPath(['global', ...root.path])
    .filter((field) => !omitted.has(field.key))
    .flatMap((field) => {
      const setting = routerSettingsFieldAt([...root.path, field.key], field)
      return setting ? [setting] : []
    })
}

const SECTION_ICONS: Record<string, LucideIcon> = {
  router_core: Cpu,
  fallback: RotateCcw,
  learning: GraduationCap,
  api: Network,
  response_api: MessagesSquare,
  observability: Telescope,
  authz: KeyRound,
  ratelimit: Gauge,
  budgets: Wallet,
  management_api: ServerCog,
  router_replay: History,
  startup_status: HeartPulse,
  masking_identity: EyeOff,
  response_cache: Zap,
  memory: Brain,
  vector_store: Database,
  tool_sessions: Wrench,
  guard_verdicts: ShieldCheck,
  kv_transfer: ArrowLeftRight,
  looper: Repeat,
  system_models: BrainCircuit,
  embedding_models: Boxes,
  admission: DoorOpen,
  signal_limits: Timer,
  deployments: Container,
  plugins: Puzzle,
}

// Copy for cards the legacy catalog had no entry for.
const SECTION_COPY: Record<string, { title: string; description: string }> = {
  fallback: {
    title: 'Fallback & retries',
    description:
      'Retry another backend when one fails, with a circuit breaker that rests a failing backend.',
  },
  tool_sessions: {
    title: 'Tool sessions',
    description:
      'Where the router keeps the state of tool calls that span several requests, and how much of it.',
  },
  guard_verdicts: {
    title: 'Guard verdicts',
    description:
      'Reuse complete request-stage guard checks across Router replicas through an authenticated store.',
  },
  kv_transfer: {
    title: 'KV transfer',
    description:
      'Hand a conversation’s KV cache from one backend to a compatible one instead of recomputing it.',
  },
  signal_limits: {
    title: 'Signal timeouts',
    description:
      'How long a request waits for its signals, and what happens when the decision model is overloaded.',
  },
  deployments: {
    title: 'Model deployments',
    description:
      'Every model the router runs or calls for its own work: decision models, embeddings and rewriters.',
  },
  plugins: {
    title: 'Default plugins',
    description: 'Plugins that run on every request before a decision adds its own.',
  },
}

const LAYER_TONE: Record<RouterSettingsLayer, Tone> = {
  router: 'slate',
  services: 'slate',
  stores: 'slate',
  integrations: 'slate',
  model_catalog: 'green',
  plugins: 'rose',
}

export const ROUTER_SETTINGS_LAYERS: Array<{ key: RouterSettingsLayer; title: string }> = [
  { key: 'router', title: 'Router' },
  { key: 'services', title: 'Services' },
  { key: 'stores', title: 'Stores' },
  { key: 'integrations', title: 'Integrations' },
  { key: 'model_catalog', title: 'Model catalog' },
  { key: 'plugins', title: 'Plugins' },
]

function sectionCopy(key: string, fallbackTitle: string): { title: string; description: string } {
  if (key in SECTION_META) {
    const meta = SECTION_META[key as RouterSystemKey]
    return { title: settingLabel(meta.title), description: meta.description }
  }
  return (
    SECTION_COPY[key] ?? {
      title: settingLabel(fallbackTitle),
      description: routerLayerMeta('').description,
    }
  )
}

function claimedPaths(cards: readonly RouterSettingsCardPolicy[]): string[] {
  return cards.flatMap((card) => card.roots.map((root) => root.path.join('.')))
}

function buildSection(
  card: RouterSettingsCardPolicy,
  fallbackTitle: string,
): RouterSettingsSection {
  const roots = card.roots.filter((root) => contractNodeAt(['global', ...root.path]) !== undefined)
  const masterPath = card.master?.join('.')
  const fields = roots.flatMap(rootFields)
  const master = masterPath
    ? fields.find((field) => field.path.join('.') === masterPath)
    : undefined
  const copy = sectionCopy(card.key, fallbackTitle)
  return {
    key: card.key,
    id: `global-section-${card.key}`,
    layer: card.layer,
    title: copy.title,
    description: copy.description,
    icon: SECTION_ICONS[card.key] ?? Layers,
    tone: LAYER_TONE[card.layer],
    paths: roots.map((root) => `global.${root.path.join('.')}`),
    roots,
    fields:
      master && master.schema.kind === 'boolean'
        ? fields.filter((field) => field !== master)
        : fields,
    master: master && master.schema.kind === 'boolean' ? master : undefined,
    custom: card.custom,
  }
}

let sectionsCache: RouterSettingsSection[] | null = null

/** Every router settings card in display order. */
export function routerSettingsSections(): RouterSettingsSection[] {
  if (sectionsCache) return sectionsCache
  const claimed = claimedPaths(ROUTER_SETTINGS_CARDS)
  const extra: RouterSettingsCardPolicy[] = ROUTER_CONFIG_EXTENSION.global_sections
    .filter((surface) => {
      const dotted = surface.path.join('.')
      return (
        !ROUTER_SETTINGS_EXCLUDED_PATHS.includes(dotted) &&
        !claimed.some(
          (path) =>
            path === dotted || path.startsWith(`${dotted}.`) || dotted.startsWith(`${path}.`),
        )
      )
    })
    .map((surface) => {
      const node = contractNodeAt(['global', ...surface.path])
      const objectLike = node ? Boolean(concrete(node).properties) : false
      return {
        key: surface.path.join('_'),
        layer: (surface.layer as RouterSettingsLayer) ?? 'router',
        roots: [{ path: surface.path, mode: objectLike ? 'children' : 'field' }],
      }
    })
  const titles = new Map(
    ROUTER_CONFIG_EXTENSION.global_sections.map((surface) => [
      surface.path.join('_'),
      surface.display_name,
    ]),
  )
  const cards = [...ROUTER_SETTINGS_CARDS, ...extra]
  const order = ROUTER_SETTINGS_LAYERS.map((layer) => layer.key)
  sectionsCache = cards
    .map((card) => buildSection(card, titles.get(card.key) ?? humanize(card.key)))
    .filter((section) => section.fields.length > 0 || section.master || section.custom)
    .sort((left, right) => order.indexOf(left.layer) - order.indexOf(right.layer))
  return sectionsCache
}

const ACRONYMS: Record<string, string> = {
  api: 'API',
  ca: 'CA',
  cpu: 'CPU',
  db: 'DB',
  gpu: 'GPU',
  grpc: 'gRPC',
  http: 'HTTP',
  https: 'HTTPS',
  id: 'ID',
  ids: 'IDs',
  ip: 'IP',
  json: 'JSON',
  jwt: 'JWT',
  kv: 'KV',
  llm: 'LLM',
  mb: 'MB',
  mcp: 'MCP',
  ms: 'ms',
  otlp: 'OTLP',
  pii: 'PII',
  sso: 'SSO',
  ssl: 'SSL',
  tls: 'TLS',
  ttl: 'TTL',
  uri: 'URI',
  url: 'URL',
  vsr: 'vSR',
  yaml: 'YAML',
}

/** Sentence-case label that keeps acronyms and identifiers such as efConstruction. */
export function settingLabel(label: string): string {
  return label
    .split(/\s+/)
    .filter(Boolean)
    .map((word, index) => {
      const bare = word.toLowerCase().replace(/[^a-z0-9]/g, '')
      if (ACRONYMS[bare]) return word.toLowerCase().replace(bare, ACRONYMS[bare])
      if (word.length === 1) return word
      if (/[a-z][A-Z]/.test(word) || /^[A-Z0-9]{2,}$/.test(word)) return word
      return index === 0
        ? word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
        : word.toLowerCase()
    })
    .join(' ')
}

const UNIT_RULES: Array<{ key: RegExp; unit: string; label: RegExp }> = [
  { key: /_(seconds|sec|secs)$/, unit: 's', label: /\s*(\((seconds?|s)\)|seconds?|secs?)$/i },
  { key: /_ms$/, unit: 'ms', label: /\s*(\((ms|milliseconds?)\)|ms|milliseconds?)$/i },
  { key: /_mb$/, unit: 'MB', label: /\s*(\(mb\)|mb)$/i },
  { key: /_days$/, unit: 'days', label: /\s*days$/i },
  { key: /_minutes$/, unit: 'min', label: /\s*minutes$/i },
  { key: /_bytes$/, unit: 'bytes', label: /\s*bytes$/i },
]

/** Unit suffix and the label without it, read from the key name. */
export function settingPresentation(key: string, label: string): { label: string; unit?: string } {
  const rule = UNIT_RULES.find((candidate) => candidate.key.test(key))
  if (!rule) return { label: settingLabel(label) }
  const stripped = label.replace(rule.label, '').trim()
  const keep = !stripped || /^(max|min|default)$/i.test(stripped)
  return { label: settingLabel(keep ? label : stripped), unit: rule.unit }
}

const SECRET_KEY = /(^|_)(password|secret|api_key|auth_token|token|private_key)$/

export function isSecretSetting(schema: RouterStructuredSchema, key: string): boolean {
  return schema.kind === 'password' || (schema.kind === 'string' && SECRET_KEY.test(key))
}

// Global fields the running stack cannot take in place. The Router judges the
// first; Envoy's generated config carries the others, so a split stack waits
// for `vllm-sr serve` to recreate Envoy.
const RESTART_SENSITIVE: Array<{ path: string; reason: string }> = [
  {
    path: 'services.api.routing_preview.max_concurrency',
    reason: 'The management listener sizes its routing preview pool when it starts.',
  },
  {
    path: 'services.authz.identity',
    reason: 'Envoy’s generated config strips these identity headers.',
  },
  {
    path: 'services.authz.delegated_credential.header',
    reason: 'Envoy’s generated config strips this header.',
  },
]

export function restartReason(path: readonly string[]): string | undefined {
  const dotted = path.join('.')
  return RESTART_SENSITIVE.find(
    (entry) => dotted === entry.path || dotted.startsWith(`${entry.path}.`),
  )?.reason
}

export const ROUTER_CONTRACT_VERSION = ROUTER_CONFIG_EXTENSION.config_version
