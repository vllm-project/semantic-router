import type {
  ASTModelDecl,
  ASTPluginDecl,
  ASTRouteDecl,
  ASTSignalDecl,
} from "@/types/dsl";

export type EntityKind = "model" | "signal" | "route" | "plugin";

export interface Selection {
  kind: EntityKind;
  name: string;
}

export interface SectionState {
  models: boolean;
  signals: boolean;
  routes: boolean;
  plugins: boolean;
}

export type BuilderSelectedEntity =
  | ASTModelDecl
  | ASTSignalDecl
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
