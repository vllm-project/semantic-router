import type {
  ASTModelDecl,
  ASTProjectionPartitionDecl,
  ASTProjectionMappingDecl,
  ASTProjectionScoreDecl,
  ASTPluginDecl,
  ASTRouteDecl,
  ASTSignalDecl,
} from "@/types/dsl";

export type EntityKind =
  | "model"
  | "signal"
  | "projection-partition"
  | "projection-score"
  | "projection-mapping"
  | "route"
  | "plugin";

export interface Selection {
  kind: EntityKind;
  name: string;
}

export interface SectionState {
  models: boolean;
  signals: boolean;
  projectionPartitions: boolean;
  projectionScores: boolean;
  projectionMappings: boolean;
  routes: boolean;
  plugins: boolean;
}

export type BuilderSelectedEntity =
  | ASTModelDecl
  | ASTSignalDecl
  | ASTProjectionPartitionDecl
  | ASTProjectionScoreDecl
  | ASTProjectionMappingDecl
  | ASTRouteDecl
  | ASTPluginDecl
  | null;

export interface AvailableSignal {
  signalType: string;
  name: string;
}

export interface AvailablePlugin {
  name: string;
  pluginType: string;
}
