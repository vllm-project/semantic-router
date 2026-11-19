import type { ASTProgram } from '@/types/dsl'
import type { Tone } from '../../ui'
import type { SchemaFormHints } from '../ConfigPageSchemaFieldsEditorSupport'
import { routingScopeFormHints, type RoutingScopeInventory } from '../configPageRoutingScopeHints'

const unique = (values: Iterable<string>) =>
  [...new Set(values)].filter(Boolean).sort((a, b) => a.localeCompare(b))

/**
 * The routing inventory of a parsed program, in the shape the config pages
 * build from YAML, so the Builder's forms suggest the same names.
 */
export function inventoryFromProgram(
  program: ASTProgram | null,
  catalogModels: readonly string[] = [],
): RoutingScopeInventory {
  const signals = new Map<string, Record<string, unknown>[]>()
  for (const signal of program?.signals ?? []) {
    const list = signals.get(signal.signalType) ?? []
    list.push({ ...signal.fields, name: signal.name })
    signals.set(signal.signalType, list)
  }
  const projectionOutputs = unique(
    (program?.projectionMappings ?? []).flatMap((mapping) =>
      (mapping.outputs ?? []).map((output) => output.name),
    ),
  )
  const routedModels = (program?.routes ?? []).flatMap((route) =>
    route.models.map((model) => model.model),
  )
  return {
    signals,
    signalNames: (type) =>
      type === 'projection'
        ? projectionOutputs
        : unique((signals.get(type) ?? []).map((signal) => String(signal.name ?? ''))),
    projectionOutputs,
    scoreNames: unique((program?.projectionScores ?? []).map((score) => score.name)),
    models: unique([
      ...(program?.models ?? []).map((model) => model.name),
      ...catalogModels,
      ...routedModels,
    ]),
    deployments: [],
    usedLabels: new Map(),
  }
}

export function programFormHints(
  program: ASTProgram | null,
  catalogModels: readonly string[],
  accent: Tone,
): SchemaFormHints {
  return routingScopeFormHints(inventoryFromProgram(program, catalogModels), accent)
}
