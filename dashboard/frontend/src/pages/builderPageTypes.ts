import type {
  ASTBackendDecl,
  ASTPluginDecl,
  ASTRouteDecl,
  ASTSignalDecl,
} from "@/types/dsl";

export type EntityKind = "signal" | "route" | "plugin" | "backend" | "global";

export interface Selection {
  kind: EntityKind;
  name: string;
}

export interface SectionState {
  signals: boolean;
  routes: boolean;
  plugins: boolean;
  backends: boolean;
  global: boolean;
}

export type BuilderSelectedEntity =
  | ASTSignalDecl
  | ASTRouteDecl
  | ASTPluginDecl
  | ASTBackendDecl
  | { fields: Record<string, unknown> }
  | null;

export interface AvailableSignal {
  signalType: string;
  name: string;
}

export interface AvailablePlugin {
  name: string;
  pluginType: string;
}
