import type { RouteInput } from '@/lib/dslMutations'
import type { ASTProgram, Diagnostic, EditorMode, SymbolTable, DSLFieldObject } from '@/types/dsl'

interface DSLState {
  // --- Editor content ---
  dslSource: string
  /** Canonical compiler output used for inspection outside Recipe Builder. */
  yamlOutput: string
  crdOutput: string
  diagnostics: Diagnostic[]
  symbols: SymbolTable | null
  /** Parsed AST from last successful parse (for Visual Builder) */
  ast: ASTProgram | null

  // --- Runtime ---
  wasmReady: boolean
  wasmError: string | null
  loading: boolean
  compileError: string | null

  // --- UI ---
  mode: EditorMode
  dirty: boolean
  lastCompileAt: number | null
}

interface DSLActions {
  /** Initialize WASM runtime. Call once at app startup. */
  initWasm(): Promise<void>

  /** Update DSL source (e.g., on editor keystroke). Triggers debounced validation. */
  setDslSource(source: string): void

  /** Run full compile: DSL → YAML + CRD + diagnostics. */
  compile(): void

  /** Validate only (faster than compile, for real-time feedback). */
  validate(): void

  /** Parse DSL → AST + diagnostics + symbols (for Visual Builder). */
  parseAST(): void

  /** Decompile YAML → DSL-owned models, routing, entrypoints, and recipes. */
  decompile(yaml: string): string | null

  /** Format the current DSL source. */
  format(): void

  /** Switch editor mode. */
  setMode(mode: EditorMode): void

  /** Reset editor state to initial values. */
  reset(): void

  /** Load a DSL source as the current editor document. */
  loadDsl(source: string): void

  /** Decompile a pasted document into the standalone editor. */
  importYaml(yaml: string): void

  /** Update a signal's fields in DSL source text, then re-parse AST. */
  mutateSignal(signalType: string, name: string, fields: DSLFieldObject): void

  /** Add a new signal to DSL source text, then re-parse AST. */
  addSignal(signalType: string, name: string, fields: DSLFieldObject): void

  /** Delete a signal from DSL source text, then re-parse AST. */
  deleteSignal(signalType: string, name: string): void

  /** Update a projection partition declaration's fields, then re-parse AST. */
  mutateProjectionPartition(name: string, fields: DSLFieldObject): void

  /** Add a new projection partition declaration, then re-parse AST. */
  addProjectionPartition(name: string, fields: DSLFieldObject): void

  /** Delete a projection partition declaration, then re-parse AST. */
  deleteProjectionPartition(name: string): void

  /** Update a projection score declaration, then re-parse AST. */
  mutateProjectionScore(name: string, fields: DSLFieldObject): void

  /** Add a new projection score declaration, then re-parse AST. */
  addProjectionScore(name: string, fields: DSLFieldObject): void

  /** Delete a projection score declaration, then re-parse AST. */
  deleteProjectionScore(name: string): void

  /** Update a projection mapping declaration, then re-parse AST. */
  mutateProjectionMapping(name: string, fields: DSLFieldObject): void

  /** Add a new projection mapping declaration, then re-parse AST. */
  addProjectionMapping(name: string, fields: DSLFieldObject): void

  /** Delete a projection mapping declaration, then re-parse AST. */
  deleteProjectionMapping(name: string): void

  /** Update a plugin declaration's fields, then re-parse AST. */
  mutatePlugin(name: string, pluginType: string, fields: DSLFieldObject): void

  /** Add a new plugin declaration, then re-parse AST. */
  addPlugin(name: string, pluginType: string, fields: DSLFieldObject): void

  /** Delete a plugin declaration, then re-parse AST. */
  deletePlugin(name: string, pluginType: string): void

  /** Delete a route declaration, then re-parse AST. */
  deleteRoute(name: string): void

  /** Update a route declaration, then re-parse AST. */
  mutateRoute(name: string, input: RouteInput): void

  /** Add a new route, then re-parse AST. */
  addRoute(name: string, input: RouteInput): void
}

type DSLStore = DSLState & DSLActions

export type { DSLActions, DSLState, DSLStore }
