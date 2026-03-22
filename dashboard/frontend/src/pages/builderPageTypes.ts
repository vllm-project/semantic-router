import type {
  ASTModelDecl,
  ASTProjectionMappingDecl,
  ASTProjectionScoreDecl,
  ASTPluginDecl,
  ASTRouteDecl,
  ASTSignalDecl,
  ASTSignalGroupDecl,
} from "@/types/dsl";

export type EntityKind =
  | "model"
  | "signal"
  | "signal-group"
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
  signalGroups: boolean;
  projectionScores: boolean;
  projectionMappings: boolean;
  routes: boolean;
  plugins: boolean;
}

export type BuilderSelectedEntity =
  | ASTModelDecl
  | ASTSignalDecl
  | ASTSignalGroupDecl
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
