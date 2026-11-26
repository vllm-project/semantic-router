import type { Edge, Node } from 'reactflow'
import type { Tone } from '../../ui'
import { signalTypeLabel } from '../configPageSignalCatalog'
import { algorithmMeta } from '../configPageDecisionPresentation'
import type {
  DecisionConfig,
  ParsedTopology,
  RuleCondition,
  RuleNode,
  SignalConfig,
  TestQueryResult,
} from './types'

export type TopoKind = 'lane' | 'entry' | 'signal' | 'projection' | 'decision' | 'more' | 'model'
export type TopoState = 'idle' | 'active' | 'secondary' | 'dimmed'
export type Layer = 'signals' | 'projections' | 'decisions' | 'models'
export type Density = 'compact' | 'comfortable'
export type DecisionKind = 'routing' | 'guardrail' | 'fallback' | 'default'

export interface TopoNodeData {
  kind: TopoKind
  label: string
  sublabel?: string
  /** Short right-aligned fact shown in compact density, such as a priority. */
  meta?: string
  tone: Tone
  signalType?: string
  provider?: string
  count?: number
  state: TopoState
  compact: boolean
  width: number
  height: number
  /** Logical id of the entity behind the node, independent of hidden layers. */
  ref: string
  warning?: string
}

export interface TopoEdgeData {
  tone: Tone
  state: TopoState
}

export interface ProjectionGroup {
  mapping: string
  source: string
  method: string
  outputs: string[]
  upstream: Array<{ type: string; name: string }>
}

export interface TopologyModel {
  entryNames: string[]
  signalTypes: string[]
  signalsByType: Map<string, SignalConfig[]>
  projections: ProjectionGroup[]
  decisions: DecisionConfig[]
  shown: DecisionConfig[]
  hidden: DecisionConfig[]
  models: string[]
  defaultModel?: string
  links: Set<string>
}

export const ENTRY_ID = 'client'
export const MORE_ID = 'decision-more'
export const DEFAULT_ID = 'decision-default'
export const signalNodeId = (type: string) => `signal-group-${type}`
export const projectionNodeId = (mapping: string) => `projection-group-${mapping}`
export const decisionNodeId = (name: string) => `decision-${name}`
export const modelNodeId = (name: string) => `model-${name}`

const VISIBLE_DECISIONS: Record<Density, number> = { compact: 14, comfortable: 10 }
const WIDTHS: Record<Exclude<TopoKind, 'lane'>, number> = {
  entry: 184,
  signal: 192,
  projection: 200,
  decision: 228,
  more: 228,
  model: 204,
}
const LANES = ['entry', 'signal', 'projection', 'decision', 'model'] as const
type Lane = (typeof LANES)[number]
const LANE_LAYER: Record<Lane, Layer | null> = {
  entry: null,
  signal: 'signals',
  projection: 'projections',
  decision: 'decisions',
  model: 'models',
}
const LANE_LABEL: Record<Lane, string> = {
  entry: 'Public model',
  signal: 'Signals',
  projection: 'Projections',
  decision: 'Decisions',
  model: 'Models',
}
const LANE_TONE: Record<Lane, Tone> = {
  entry: 'blue',
  signal: 'gold',
  projection: 'teal',
  decision: 'violet',
  model: 'green',
}
const LANE_GAP = 44

export function ruleConditions(rule: RuleNode): RuleCondition[] {
  if ('conditions' in rule) return rule.conditions.flatMap(ruleConditions)
  return [rule]
}

export function decisionKind(decision: DecisionConfig): DecisionKind {
  if ((decision.plugins ?? []).some((plugin) => plugin.type === 'fast_response')) return 'guardrail'
  if (ruleConditions(decision.rules).length === 0) return 'fallback'
  return 'routing'
}

const KIND_TONE: Record<DecisionKind, Tone> = {
  routing: 'violet',
  guardrail: 'rose',
  fallback: 'slate',
  default: 'slate',
}

/** Whether a decision can match: a fallback has no conditions, an unreachable one names no declared signal. */
export function getDecisionReachability(
  decision: DecisionConfig,
  configuredSignals: ReadonlySet<string>,
): { isFallback: boolean; isUnreachable: boolean; unreachableReason?: string } {
  const leaves = ruleConditions(decision.rules)
  if (leaves.length === 0) return { isFallback: true, isUnreachable: false }
  const configured = leaves.some((condition) =>
    configuredSignals.has(`${condition.type}:${condition.name}`),
  )
  return {
    isFallback: false,
    isUnreachable: !configured,
    ...(configured ? {} : { unreachableReason: 'Referenced signals not configured' }),
  }
}

/** A rule as the DSL writes it, for summaries. */
export function ruleSummary(rule: RuleNode): string {
  if (!('conditions' in rule)) return `${rule.type}("${rule.name}")`
  if (rule.conditions.length === 0) return 'always'
  if (rule.operator === 'NOT') {
    const inner = rule.conditions[0] ? ruleSummary(rule.conditions[0]) : '?'
    return 'conditions' in (rule.conditions[0] ?? {}) ? `NOT (${inner})` : `NOT ${inner}`
  }
  const parts = rule.conditions.map((condition) =>
    'conditions' in condition && condition.operator === 'OR' && rule.operator === 'AND'
      ? `(${ruleSummary(condition)})`
      : ruleSummary(condition),
  )
  return parts.join(` ${rule.operator} `)
}

function barycenter(values: number[], fallback: number) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : fallback
}

/** Entities and logical edges of a routing scope before layers are hidden or laid out. */
export function topologyModel(
  topology: ParsedTopology,
  entryNames: string[],
  options: { density: Density; showAllDecisions: boolean },
): TopologyModel {
  const signals = topology.signals.filter((signal) => signal.type !== 'projection')
  const signalsByType = new Map<string, SignalConfig[]>()
  for (const signal of signals) {
    const list = signalsByType.get(signal.type) ?? []
    list.push(signal)
    signalsByType.set(signal.type, list)
  }

  const groups = new Map<string, ProjectionGroup>()
  for (const signal of topology.signals) {
    if (signal.type !== 'projection') continue
    const config = signal.config as {
      source?: string
      method?: string
      mapping?: string
      upstreamSignals?: Array<{ type: string; name: string }>
    }
    const mapping = config.mapping || signal.name
    const group = groups.get(mapping) ?? {
      mapping,
      source: config.source ?? '',
      method: config.method ?? '',
      outputs: [],
      upstream: config.upstreamSignals ?? [],
    }
    group.outputs.push(signal.name)
    groups.set(mapping, group)
  }
  const projections = [...groups.values()]
  const projectionOf = (output: string) =>
    projections.find((projection) => projection.outputs.includes(output))

  const decisions = [...topology.decisions].sort((a, b) => b.priority - a.priority)
  const limit = options.showAllDecisions ? decisions.length : VISIBLE_DECISIONS[options.density]
  const shown = decisions.slice(0, limit)
  const hidden = decisions.slice(limit)
  const index = new Map(shown.map((decision, position) => [decision.name, position]))
  const moreIndex = shown.length
  const target = (decision: DecisionConfig) =>
    index.has(decision.name) ? decisionNodeId(decision.name) : MORE_ID

  const links = new Set<string>()
  const link = (source: string, targetId: string) => links.add(`${source}>${targetId}`)
  const referencedTypes = new Set<string>()
  for (const decision of decisions) {
    for (const condition of ruleConditions(decision.rules)) {
      if (condition.type === 'projection') {
        const projection = projectionOf(condition.name)
        if (projection) link(projectionNodeId(projection.mapping), target(decision))
        continue
      }
      referencedTypes.add(condition.type)
      link(signalNodeId(condition.type), target(decision))
    }
  }
  for (const projection of projections) {
    for (const type of new Set(projection.upstream.map((input) => input.type))) {
      if (type === 'projection') continue
      referencedTypes.add(type)
      link(signalNodeId(type), projectionNodeId(projection.mapping))
    }
  }
  const signalTypes = [...new Set([...signalsByType.keys(), ...referencedTypes])]
  for (const type of signalTypes) link(ENTRY_ID, signalNodeId(type))

  const modelSet = new Set<string>()
  for (const decision of decisions) {
    for (const ref of decision.modelRefs) {
      if (!ref.model) continue
      modelSet.add(ref.model)
      link(target(decision), modelNodeId(ref.model))
    }
  }
  if (topology.defaultModel) {
    modelSet.add(topology.defaultModel)
    link(DEFAULT_ID, modelNodeId(topology.defaultModel))
    link(ENTRY_ID, DEFAULT_ID)
  }

  const position = (id: string) =>
    id === MORE_ID
      ? moreIndex
      : id === DEFAULT_ID
        ? moreIndex + 1
        : (index.get(id.slice(9)) ?? moreIndex)
  const downstream = (id: string): number[] =>
    [...links]
      .filter((key) => key.startsWith(`${id}>`))
      .map((key) => key.split('>')[1])
      .flatMap((next) =>
        next.startsWith('projection-group-')
          ? [...links]
              .filter((key) => key.startsWith(`${next}>`))
              .map((key) => position(key.split('>')[1]))
          : [position(next)],
      )
  const orderedTypes = [...signalTypes].sort(
    (a, b) =>
      barycenter(downstream(signalNodeId(a)), moreIndex) -
      barycenter(downstream(signalNodeId(b)), moreIndex),
  )
  const orderedProjections = [...projections].sort(
    (a, b) =>
      barycenter(downstream(projectionNodeId(a.mapping)), moreIndex) -
      barycenter(downstream(projectionNodeId(b.mapping)), moreIndex),
  )
  const users = (model: string) =>
    [...links]
      .filter((key) => key.endsWith(`>${modelNodeId(model)}`))
      .map((key) => position(key.split('>')[0]))
  const models = [...modelSet].sort(
    (a, b) => barycenter(users(a), moreIndex) - barycenter(users(b), moreIndex),
  )

  return {
    entryNames,
    signalTypes: orderedTypes,
    signalsByType,
    projections: orderedProjections,
    decisions,
    shown,
    hidden,
    models,
    defaultModel: topology.defaultModel,
    links,
  }
}

/** Remove nodes and connect each source to each target, so hiding a layer keeps paths intact. */
function contract(links: Set<string>, ids: readonly string[]) {
  let current = new Set(links)
  for (const id of ids) {
    const incoming = [...current]
      .filter((key) => key.endsWith(`>${id}`))
      .map((key) => key.split('>')[0])
    const outgoing = [...current]
      .filter((key) => key.startsWith(`${id}>`))
      .map((key) => key.split('>')[1])
    const next = new Set(
      [...current].filter((key) => !key.startsWith(`${id}>`) && !key.endsWith(`>${id}`)),
    )
    for (const source of incoming)
      for (const targetId of outgoing) next.add(`${source}>${targetId}`)
    current = next
  }
  return current
}

interface Highlight {
  active: Set<string>
  secondary: Set<string>
  /** Signal families whose fired signals the matched decision reads directly. */
  decisiveTypes: Set<string>
  /** Per active projection, the families of fired signals it reads. */
  projectionTypes: Map<string, Set<string>>
  moreLabel?: { label: string; sublabel: string; meta: string }
}

function pathHighlight(model: TopologyModel, result: TestQueryResult | null): Highlight | null {
  if (!result) return null
  const active = new Set<string>([ENTRY_ID])
  const secondary = new Set<string>()
  const fired = result.matchedSignals.filter((signal) => signal.type !== 'projection')
  for (const signal of fired) active.add(signalNodeId(signal.type))
  const decisiveTypes = new Set<string>()
  const projectionTypes = new Map<string, Set<string>>()
  let moreLabel: Highlight['moreLabel']
  const decision = model.decisions.find((item) => item.name === result.matchedDecision)
  if (decision) {
    const visible = model.shown.includes(decision)
    active.add(visible ? decisionNodeId(decision.name) : MORE_ID)
    if (!visible) {
      moreLabel = {
        label: decision.name,
        sublabel: `P${decision.priority} · one of ${model.hidden.length} more`,
        meta: `P${decision.priority}`,
      }
    }
    for (const condition of ruleConditions(decision.rules)) {
      if (condition.type === 'projection') {
        const projection = model.projections.find((item) => item.outputs.includes(condition.name))
        if (!projection) continue
        const id = projectionNodeId(projection.mapping)
        active.add(id)
        const fed = fired.filter((signal) =>
          projection.upstream.some(
            (input) => input.type === signal.type && input.name === signal.name,
          ),
        )
        projectionTypes.set(id, new Set(fed.map((signal) => signal.type)))
      } else if (fired.some((signal) => signal.type === condition.type)) {
        decisiveTypes.add(condition.type)
      }
    }
  } else if (result.matchedDecision || result.isFallbackDecision) {
    if (model.defaultModel) active.add(DEFAULT_ID)
  }
  const chosen = result.selectedModel || result.matchedModels[0]
  if (chosen) active.add(modelNodeId(chosen))
  for (const candidate of [...result.matchedModels, ...(result.recommendedModels ?? [])]) {
    if (candidate !== chosen) secondary.add(modelNodeId(candidate))
  }
  return { active, secondary, decisiveTypes, projectionTypes, moreLabel }
}

function edgeState(source: string, targetId: string, highlight: Highlight): TopoState {
  if (!highlight.active.has(source)) return 'dimmed'
  if (highlight.secondary.has(targetId)) return 'secondary'
  if (!highlight.active.has(targetId)) return 'dimmed'
  if (source.startsWith('signal-group-')) {
    const type = source.slice('signal-group-'.length)
    if (targetId.startsWith('projection-group-'))
      return highlight.projectionTypes.get(targetId)?.has(type) ? 'active' : 'dimmed'
    return highlight.decisiveTypes.has(type) ? 'active' : 'dimmed'
  }
  return 'active'
}

export interface TopologyOptions {
  density: Density
  layers: Record<Layer, boolean>
  result: TestQueryResult | null
  focus: string | null
  /** Provider ids by model name, for logos. */
  providers?: Record<string, string>
}

export function buildTopology(model: TopologyModel, options: TopologyOptions) {
  const compact = options.density === 'compact'
  const nodeHeight = compact ? 40 : 52
  const rowHeight = nodeHeight + (compact ? 8 : 12)
  const entryHeight = compact ? 64 : 88
  const configured = new Set(
    [...model.signalsByType.values()].flat().map((signal) => `${signal.type}:${signal.name}`),
  )
  for (const projection of model.projections)
    for (const output of projection.outputs) configured.add(`projection:${output}`)

  type Item = Omit<TopoNodeData, 'state' | 'compact' | 'width' | 'height'>
  const decisionItem = (decision: DecisionConfig): Item => {
    const kind = decisionKind(decision)
    const reach = getDecisionReachability(decision, configured)
    const algorithm = decision.algorithm?.type
      ? (algorithmMeta(decision.algorithm.type)?.label ?? decision.algorithm.type)
      : null
    const target =
      kind === 'guardrail'
        ? 'fast response'
        : decision.modelRefs.length === 1
          ? decision.modelRefs[0].model
          : `${decision.modelRefs.length} models`
    return {
      kind: 'decision',
      label: decision.name,
      sublabel: [`P${decision.priority}`, algorithm, kind === 'fallback' ? 'catch-all' : target]
        .filter(Boolean)
        .join(' · '),
      meta: `P${decision.priority}`,
      tone: KIND_TONE[kind],
      ref: decisionNodeId(decision.name),
      warning: reach.unreachableReason,
    }
  }
  const lanes: Record<Lane, Item[]> = {
    entry: [
      {
        kind: 'entry',
        label: model.entryNames[0] ?? 'vllm-sr/auto',
        sublabel:
          model.entryNames.length > 1
            ? `+${model.entryNames.length - 1} more · ${model.entryNames.slice(1).join(', ')}`
            : 'OpenAI-compatible API',
        count: model.entryNames.length,
        tone: 'blue',
        ref: ENTRY_ID,
      },
    ],
    signal: model.signalTypes.map((type) => {
      const names = (model.signalsByType.get(type) ?? []).map((signal) => signal.name)
      return {
        kind: 'signal' as const,
        label: signalTypeLabel(type),
        sublabel: names.length ? names.slice(0, 3).join(', ') : 'referenced, not declared',
        count: names.length,
        signalType: type,
        tone: 'gold' as const,
        ref: signalNodeId(type),
        warning: names.length ? undefined : 'Referenced signals not configured',
      }
    }),
    projection: model.projections.map((projection) => ({
      kind: 'projection' as const,
      label: projection.mapping,
      sublabel: projection.outputs.join(', '),
      count: projection.outputs.length,
      tone: 'teal' as const,
      ref: projectionNodeId(projection.mapping),
    })),
    decision: [
      ...model.shown.map(decisionItem),
      ...(model.hidden.length
        ? [
            {
              kind: 'more' as const,
              label: `${model.hidden.length} more decisions`,
              sublabel: 'Lower priority · show all',
              count: model.hidden.length,
              tone: 'violet' as const,
              ref: MORE_ID,
            },
          ]
        : []),
      ...(model.defaultModel
        ? [
            {
              kind: 'decision' as const,
              label: 'Default model',
              sublabel: 'When no decision matches',
              tone: 'slate' as const,
              ref: DEFAULT_ID,
            },
          ]
        : []),
    ],
    model: model.models.map((name) => ({
      kind: 'model' as const,
      label: name,
      sublabel: name === model.defaultModel ? 'default model' : undefined,
      provider: options.providers?.[name],
      tone: 'green' as const,
      ref: modelNodeId(name),
    })),
  }

  const visibleLanes = LANES.filter((lane) => {
    const layer = LANE_LAYER[lane]
    return (layer === null || options.layers[layer]) && lanes[lane].length > 0
  })
  const hiddenIds = LANES.filter((lane) => !visibleLanes.includes(lane)).flatMap((lane) =>
    lanes[lane].map((item) => item.ref),
  )
  const links = contract(model.links, hiddenIds)

  const tallest = Math.max(
    entryHeight,
    ...visibleLanes.map((lane) =>
      lane === 'entry' ? entryHeight : lanes[lane].length * rowHeight - (rowHeight - nodeHeight),
    ),
  )
  const nodes: Node<TopoNodeData>[] = []
  let x = 0
  for (const lane of visibleLanes) {
    const items = lanes[lane]
    const width = WIDTHS[items[0].kind as Exclude<TopoKind, 'lane'>]
    const height = lane === 'entry' ? entryHeight : nodeHeight
    const laneHeight =
      lane === 'entry' ? entryHeight : items.length * rowHeight - (rowHeight - nodeHeight)
    const top = (tallest - laneHeight) / 2
    nodes.push({
      id: `lane-${lane}`,
      type: 'lane',
      position: { x, y: -44 },
      draggable: false,
      selectable: false,
      data: {
        kind: 'lane',
        label: LANE_LABEL[lane],
        count:
          lane === 'decision'
            ? model.decisions.length
            : lane === 'signal'
              ? [...model.signalsByType.values()].reduce((total, list) => total + list.length, 0)
              : lane === 'entry'
                ? model.entryNames.length
                : items.length,
        sublabel:
          lane === 'signal'
            ? `${items.length} ${items.length === 1 ? 'family' : 'families'}`
            : lane === 'decision' && model.hidden.length
              ? `top ${model.shown.length} by priority`
              : undefined,
        tone: LANE_TONE[lane],
        state: 'idle',
        compact,
        width,
        height: 28,
        ref: `lane-${lane}`,
      },
    })
    items.forEach((item, position) => {
      nodes.push({
        id: item.ref,
        type: item.kind,
        position: { x, y: top + position * rowHeight },
        draggable: false,
        data: { ...item, state: 'idle', compact, width, height },
      })
    })
    x += width + LANE_GAP
  }

  const ids = new Set(nodes.map((node) => node.id))
  const edges: Edge<TopoEdgeData>[] = [...links]
    .map((key) => key.split('>'))
    .filter(([source, targetId]) => ids.has(source) && ids.has(targetId))
    .map(([source, targetId]) => {
      const tone = nodes.find((node) => node.id === source)!.data.tone
      return {
        id: `${source}>${targetId}`,
        source,
        target: targetId,
        type: 'flow',
        data: { tone: tone === 'slate' ? 'violet' : tone, state: 'idle' },
      }
    })

  const highlight = pathHighlight(model, options.result)
  if (highlight) {
    for (const id of [...highlight.active]) if (!ids.has(id)) highlight.active.delete(id)
    for (const node of nodes) {
      if (node.type === 'lane') continue
      node.data.state = highlight.active.has(node.id)
        ? 'active'
        : highlight.secondary.has(node.id)
          ? 'secondary'
          : 'dimmed'
      if (node.id === MORE_ID && highlight.moreLabel) {
        node.data.label = highlight.moreLabel.label
        node.data.sublabel = highlight.moreLabel.sublabel
        node.data.meta = highlight.moreLabel.meta
      }
    }
    for (const edge of edges) edge.data!.state = edgeState(edge.source, edge.target, highlight)
  } else if (options.focus && ids.has(options.focus)) {
    const focus = options.focus
    const neighbours = new Set([
      focus,
      ...edges.filter((edge) => edge.source === focus).map((edge) => edge.target),
      ...edges.filter((edge) => edge.target === focus).map((edge) => edge.source),
    ])
    for (const node of nodes) {
      if (node.type === 'lane') continue
      node.data.state = node.id === focus ? 'active' : neighbours.has(node.id) ? 'idle' : 'dimmed'
    }
    for (const edge of edges)
      edge.data!.state = edge.source === focus || edge.target === focus ? 'active' : 'dimmed'
  }

  const bounds = { x: 0, y: -44, width: Math.max(0, x - LANE_GAP), height: tallest + 44 }
  const lit = highlight
    ? nodes.filter((node) => node.type !== 'lane' && node.data.state !== 'dimmed')
    : []
  const pathBounds = lit.length
    ? (() => {
        const top = Math.min(...lit.map((node) => node.position.y)) - 28
        const bottom = Math.max(...lit.map((node) => node.position.y + node.data.height)) + 28
        return { x: 0, y: top, width: bounds.width, height: bottom - top }
      })()
    : null
  return { nodes, edges, bounds, pathBounds }
}
