import { ruleLeaves } from './configPageDecisionRuleSupport'
import type {
  ConfigData,
  ConfigProjections,
  DecisionConfig,
  ProjectionMapping,
  ProjectionMappingOutput,
  ProjectionPartition,
  ProjectionScore,
  ProjectionScoreInput,
} from './configPageSupport'

export interface ProjectionPartitionFormState {
  name: string
  semantics: string
  members: string[]
  temperature?: number
  default?: string
}

export interface ProjectionScoreFormState {
  name: string
  method: string
  inputs: ProjectionScoreInput[]
}

export interface ProjectionMappingFormState {
  name: string
  source: string
  method: string
  calibration?: ProjectionMapping['calibration']
  outputs: ProjectionMappingOutput[]
}

export type ProjectionKind = 'partition' | 'score' | 'mapping'

export type ProjectionDeleteTarget = { kind: ProjectionKind; name: string }

export const PROJECTION_KIND_LABEL: Record<ProjectionKind, string> = {
  partition: 'Partition',
  score: 'Score',
  mapping: 'Mapping',
}

export const PROJECTION_COLLECTION = {
  partition: 'partitions',
  score: 'scores',
  mapping: 'mappings',
} as const satisfies Record<ProjectionKind, keyof ConfigProjections>

export const PROJECTION_SCHEMA_REF: Record<ProjectionKind, string> = {
  partition: '#/$defs/ProjectionPartition',
  score: '#/$defs/ProjectionScore',
  mapping: '#/$defs/ProjectionMapping',
}

export const EMPTY_PROJECTIONS: ConfigProjections = {
  partitions: [],
  scores: [],
  mappings: [],
}
export const EMPTY_PARTITIONS: ProjectionPartition[] = []
export const EMPTY_SCORES: ProjectionScore[] = []
export const EMPTY_MAPPINGS: ProjectionMapping[] = []

export const projectionInputSource = (input: ProjectionScoreInput): string =>
  `${input.type}:${input.name ?? ''}`

export const cloneProjections = (cfg: ConfigData): ConfigProjections => ({
  partitions: [...(cfg.projections?.partitions || [])],
  scores: [...(cfg.projections?.scores || [])],
  mappings: [...(cfg.projections?.mappings || [])],
})

export const ensureProjectionConfig = (cfg: ConfigData) => {
  if (!cfg.projections) {
    cfg.projections = { partitions: [], scores: [], mappings: [] }
  }
  if (!cfg.projections.partitions) cfg.projections.partitions = []
  if (!cfg.projections.scores) cfg.projections.scores = []
  if (!cfg.projections.mappings) cfg.projections.mappings = []
  return cfg.projections
}

function decisionsReading(
  config: ConfigData | null,
  match: (type: string, name: string) => boolean,
): DecisionConfig[] {
  return (config?.decisions ?? []).filter((decision) =>
    ruleLeaves(decision.rules).some((leaf) =>
      match(String(leaf.node.type ?? ''), String(leaf.node.name ?? '')),
    ),
  )
}

/** Mappings that turn a score into named bands. */
export function scoreMappings(config: ConfigData | null, scoreName: string): ProjectionMapping[] {
  return (config?.projections?.mappings ?? []).filter((mapping) => mapping.source === scoreName)
}

/** Scores that take another projection as an input. */
export function scoresReadingProjection(
  config: ConfigData | null,
  name: string,
): ProjectionScore[] {
  return (config?.projections?.scores ?? []).filter((score) =>
    (score.inputs ?? []).some((input) => input.type === 'projection' && input.name === name),
  )
}

/** Decisions whose rules depend on a projection: partition members, mapping bands, or a score's bands. */
export function projectionDecisionConsumers(
  config: ConfigData | null,
  kind: ProjectionKind,
  entity: ProjectionPartition | ProjectionScore | ProjectionMapping,
): Array<{ decision: DecisionConfig; via: string }> {
  if (kind === 'partition') {
    const members = new Set((entity as ProjectionPartition).members ?? [])
    return decisionsReading(config, (type, name) => type === 'domain' && members.has(name)).map(
      (decision) => ({
        decision,
        via: ruleLeaves(decision.rules)
          .filter((leaf) => leaf.node.type === 'domain' && members.has(String(leaf.node.name)))
          .map((leaf) => `domain("${leaf.node.name}")`)
          .join(', '),
      }),
    )
  }
  const mappings =
    kind === 'mapping' ? [entity as ProjectionMapping] : scoreMappings(config, entity.name)
  const outputs = new Set(
    mappings.flatMap((mapping) => (mapping.outputs ?? []).map((output) => output.name)),
  )
  return decisionsReading(config, (type, name) => type === 'projection' && outputs.has(name)).map(
    (decision) => ({
      decision,
      via: ruleLeaves(decision.rules)
        .filter((leaf) => leaf.node.type === 'projection' && outputs.has(String(leaf.node.name)))
        .map((leaf) => `projection("${leaf.node.name}")`)
        .join(', '),
    }),
  )
}

export interface BandSegment {
  name: string
  from: number
  to: number
  label: string
}

const BOUND_SYMBOL = { gt: '>', gte: '≥', lt: '<', lte: '≤' } as const

export function bandLabel(output: ProjectionMappingOutput): string {
  const lower =
    output.gt !== undefined
      ? `${BOUND_SYMBOL.gt} ${output.gt}`
      : output.gte !== undefined
        ? `${BOUND_SYMBOL.gte} ${output.gte}`
        : ''
  const upper =
    output.lt !== undefined
      ? `${BOUND_SYMBOL.lt} ${output.lt}`
      : output.lte !== undefined
        ? `${BOUND_SYMBOL.lte} ${output.lte}`
        : ''
  return [lower, upper].filter(Boolean).join(' and ') || 'no bounds'
}

/** Where each band sits on one axis, for strips and previews. */
export function bandGeometry(outputs: readonly ProjectionMappingOutput[]): {
  low: number
  high: number
  bands: BandSegment[]
} {
  const bounds = outputs.flatMap((output) =>
    [output.gt, output.gte, output.lt, output.lte].filter(
      (value): value is number => typeof value === 'number' && Number.isFinite(value),
    ),
  )
  const minBound = bounds.length ? Math.min(...bounds) : 0
  const maxBound = bounds.length ? Math.max(...bounds) : 1
  const low = Math.min(0, minBound - (maxBound - minBound || 1) * 0.15)
  const high = Math.max(1, maxBound + (maxBound - minBound || 1) * 0.15)
  return {
    low,
    high,
    bands: outputs.map((output) => ({
      name: output.name,
      from: output.gt ?? output.gte ?? low,
      to: output.lt ?? output.lte ?? high,
      label: bandLabel(output),
    })),
  }
}
