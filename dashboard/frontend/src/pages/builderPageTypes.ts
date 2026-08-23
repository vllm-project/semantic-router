import type {
  ASTProjectionPartitionDecl,
  ASTProjectionMappingDecl,
  ASTProjectionScoreDecl,
  ASTPluginDecl,
  ASTRouteDecl,
  ASTSignalDecl,
} from '@/types/dsl'

export type EntityKind =
  | 'signal'
  | 'projection-partition'
  | 'projection-score'
  | 'projection-mapping'
  | 'route'
  | 'plugin'

export interface Selection {
  kind: EntityKind
  name: string
}

export interface SectionState {
  signals: boolean
  projectionPartitions: boolean
  projectionScores: boolean
  projectionMappings: boolean
  routes: boolean
  plugins: boolean
}

export type BuilderSelectedEntity =
  | ASTSignalDecl
  | ASTProjectionPartitionDecl
  | ASTProjectionScoreDecl
  | ASTProjectionMappingDecl
  | ASTRouteDecl
  | ASTPluginDecl
  | null

export interface AvailableSignal {
  signalType: string
  name: string
}

export interface AvailablePlugin {
  name: string
  pluginType: string
}
