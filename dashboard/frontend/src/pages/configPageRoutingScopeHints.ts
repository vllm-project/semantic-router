import type { FieldSchema } from '../lib/dslSchemas'
import type { SchemaFormHints } from './ConfigPageSchemaFieldsEditorSupport'
import { SIGNAL_CATALOG } from './configPageSignalCatalog'
import type { ConfigData, NormalizedModel } from './configPageSupport'

type Record_ = Record<string, unknown>

function isRecord(value: unknown): value is Record_ {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function records(value: unknown): Record_[] {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && item !== '')
    : []
}

function unique(values: Iterable<string>): string[] {
  return [...new Set(values)].filter(Boolean).sort((a, b) => a.localeCompare(b))
}

/** What one routing profile declares, for suggestions and reference checks. */
export interface RoutingScopeInventory {
  signals: Map<string, Record_[]>
  signalNames: (type: string) => string[]
  projectionOutputs: string[]
  scoreNames: string[]
  models: string[]
  deployments: string[]
  usedLabels: Map<string, string[]>
}

function collectUsedLabels(config: ConfigData | null, into: Map<string, Set<string>>) {
  const add = (type: unknown, name: unknown, label: unknown) => {
    if (typeof type !== 'string' || typeof name !== 'string' || typeof label !== 'string' || !label)
      return
    const key = `${type}:${name}`
    if (!into.has(key)) into.set(key, new Set())
    into.get(key)!.add(label)
  }
  const visit = (node: unknown) => {
    if (Array.isArray(node)) return node.forEach(visit)
    if (!isRecord(node)) return
    add(node.type, node.name, node.label)
    Object.values(node).forEach(visit)
  }
  visit(config?.decisions)
  for (const input of records(config?.projections?.scores).flatMap((score) =>
    records(score.inputs),
  )) {
    if (typeof input.name === 'string' && input.name.includes(':')) {
      const [name, label] = input.name.split(':')
      add(input.type, name, label)
    }
  }
}

export function routingScopeInventory(
  scopedConfig: ConfigData | null,
  models: readonly NormalizedModel[],
  extraDeployments: readonly string[] = [],
): RoutingScopeInventory {
  const signals = new Map<string, Record_[]>()
  for (const entry of SIGNAL_CATALOG) {
    signals.set(
      entry.type,
      records((scopedConfig?.signals as Record_ | undefined)?.[entry.collection]),
    )
  }
  const mappings = records(scopedConfig?.projections?.mappings)
  const projectionOutputs = unique(
    mappings.flatMap((mapping) =>
      records(mapping.outputs).map((output) => String(output.name ?? '')),
    ),
  )
  const scoreNames = unique(
    records(scopedConfig?.projections?.scores).map((score) => String(score.name ?? '')),
  )
  const deployments = unique([
    ...Object.keys((scopedConfig?.global?.model_catalog?.deployments as Record_ | undefined) ?? {}),
    ...extraDeployments,
  ])
  const labels = new Map<string, Set<string>>()
  collectUsedLabels(scopedConfig, labels)
  return {
    signals,
    signalNames: (type) =>
      type === 'projection'
        ? projectionOutputs
        : unique((signals.get(type) ?? []).map((signal) => String(signal.name ?? ''))),
    projectionOutputs,
    scoreNames,
    models: unique(models.map((model) => model.name)),
    deployments,
    usedLabels: new Map([...labels].map(([key, values]) => [key, unique(values)])),
  }
}

/** Labels a signal reports: its question's options, its configured labels, and labels rules already use. */
export function signalLabelsFor(
  inventory: RoutingScopeInventory,
  type: string,
  name: string,
): string[] {
  const signal = (inventory.signals.get(type) ?? []).find((entry) => entry.name === name)
  const question = isRecord(signal?.question) ? signal.question : {}
  return unique([
    ...records(question.choices).map((choice) => String(choice.key ?? '')),
    ...records(question.labels).map((label) => String(label.key ?? '')),
    ...strings(question.levels),
    ...strings(signal?.labels),
    ...(inventory.usedLabels.get(`${type}:${name}`) ?? []),
  ])
}

const MODEL_KEY = /(^|_)models?$|^destination$|^quorum_fallback_target$|^analysis_models$/

/** Suggestions by what a field names: models, deployments, scores, or declared signals. */
export function routingScopeFormHints(
  inventory: RoutingScopeInventory,
  accent?: SchemaFormHints['accent'],
): SchemaFormHints {
  return {
    accent,
    suggest: (field: FieldSchema, path: readonly string[]) => {
      const key = field.key
      if (MODEL_KEY.test(key)) return inventory.models
      if (key === 'deployment') return inventory.deployments
      if (key === 'members') return inventory.signalNames('domain')
      if (key === 'source' && path.includes('mappings')) return inventory.scoreNames
      if (key === 'signal' && path.includes('configuration')) return inventory.signalNames('pii')
      return undefined
    },
    signalNames: (type) => inventory.signalNames(type),
    signalLabels: (type, name) => signalLabelsFor(inventory, type, name),
  }
}
