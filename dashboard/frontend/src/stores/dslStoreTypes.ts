import type { RouteInput } from '@/lib/dslMutations'
import type {
  ASTProgram,
  BuilderNLGenerateRequest,
  BuilderNLProgressEvent,
  BuilderNLStagedDraft,
  ConfigVersion,
  DeployResult,
  DeployStep,
  Diagnostic,
  EditorMode,
  SymbolTable,
  DSLFieldObject,
} from '@/types/dsl'

interface DSLState {
  // --- Editor content ---
  dslSource: string
  yamlOutput: string
  crdOutput: string
  diagnostics: Diagnostic[]
  symbols: SymbolTable | null
  /** Parsed AST from last successful parse (for Visual Builder) */
  ast: ASTProgram | null
  baseConfigYaml: string

  // --- Runtime ---
  wasmReady: boolean
  wasmError: string | null
  loading: boolean
  compileError: string | null

  // --- UI ---
  mode: EditorMode
  dirty: boolean
  lastCompileAt: number | null

  // --- Deploy ---
  deploying: boolean
  deployStep: DeployStep | null
  deployResult: DeployResult | null
  showDeployConfirm: boolean
  configVersions: ConfigVersion[]

  // --- Deploy Preview (diff) ---
  deployPreviewCurrent: string
  deployPreviewMerged: string
  deployPreviewLoading: boolean
  deployPreviewError: string | null

  // --- Natural Language Builder ---
  nlGenerating: boolean
  nlGenerateError: string | null
  nlStagedDraft: BuilderNLStagedDraft | null
  nlProgressEvents: BuilderNLProgressEvent[]
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

  /** Decompile YAML → routing-only DSL (for import from existing config). */
  decompile(yaml: string): string | null

  /** Format the current DSL source. */
  format(): void

  /** Switch editor mode. */
  setMode(mode: EditorMode): void

  /** Reset editor state to initial values. */
  reset(): void

  /** Load DSL source without preserving an imported full-config deploy base. */
  loadDsl(source: string): void

  /** Load YAML and decompile only its routing section to DSL. */
  importYaml(yaml: string): void

  /** Fetch current router config YAML and decompile only its routing section to DSL. */
  loadFromRouter(): Promise<void>

  /** Update a model's fields in DSL source text, then re-parse AST. */
  mutateModel(name: string, fields: DSLFieldObject): void

  /** Add a new model to DSL source text, then re-parse AST. */
  addModel(name: string, fields: DSLFieldObject): void

  /** Delete a model from DSL source text, then re-parse AST. */
  deleteModel(name: string): void

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

  /** Show deploy confirmation dialog. Compiles first if needed. Fetches preview diff. */
  requestDeploy(): void

  /** Execute the deploy (called after user confirms). */
  executeDeploy(): Promise<void>

  /** Cancel/dismiss deploy dialog. */
  dismissDeploy(): void

  /** Rollback to a specific version. */
  rollback(version: string): Promise<void>

  /** Fetch available config versions. */
  fetchVersions(): Promise<void>

  /** Generate DSL from a natural-language request and optional custom model connection. */
  generateFromNaturalLanguage(input: BuilderNLGenerateRequest): Promise<void>

  /** Apply the staged NL draft into the live Builder editor while preserving the current deploy base YAML. */
  applyNaturalLanguageDraft(): void

  /** Clear the staged NL draft and any related review state. */
  discardNaturalLanguageDraft(): void
}

type DSLStore = DSLState & DSLActions

export type { DSLActions, DSLState, DSLStore }
