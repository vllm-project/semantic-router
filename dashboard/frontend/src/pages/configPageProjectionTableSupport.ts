import type {
  ConfigData,
  ConfigProjections,
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

export type ProjectionDeleteTarget =
  | { kind: 'partition'; name: string }
  | { kind: 'score'; name: string }
  | { kind: 'mapping'; name: string }

export const EMPTY_PROJECTIONS: ConfigProjections = {
  partitions: [],
  scores: [],
  mappings: [],
}
export const EMPTY_PARTITIONS: ProjectionPartition[] = []
export const EMPTY_SCORES: ProjectionScore[] = []
export const EMPTY_MAPPINGS: ProjectionMapping[] = []

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
