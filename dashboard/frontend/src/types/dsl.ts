/**
 * TypeScript types for DSL WASM bridge.
 * These match the JSON structures returned by the Go WASM functions:
 *   signalCompile, signalValidate, signalDecompile, signalFormat, signalParseAST
 */

// ---------- Diagnostics ----------

export type DiagLevel = 'error' | 'warning' | 'constraint'

export interface QuickFix {
  description: string
  newText: string
}

export interface Diagnostic {
  level: DiagLevel
  message: string
  line: number
  column: number
  fixes?: QuickFix[]
}

// ---------- Symbol Table (for context-aware completion) ----------

export interface SymbolInfo {
  name: string
  type: string
}

export interface SymbolTable {
  signals: SymbolInfo[]
  models: string[]
  plugins: string[]
  routes: string[]
}

// ---------- AST Types (for Visual Builder) ----------

export interface ASTPosition {
  Line: number
  Column: number
}

export type DSLFieldScalar = string | number | boolean | null
export type DSLFieldValue = DSLFieldScalar | DSLFieldObject | DSLFieldValue[]
export interface DSLFieldObject {
  [key: string]: DSLFieldValue | undefined
}

/** Boolean expression node — discriminated union via "type" field */
export type BoolExprNode =
  | { type: 'and'; left: BoolExprNode; right: BoolExprNode; pos: ASTPosition }
  | { type: 'or'; left: BoolExprNode; right: BoolExprNode; pos: ASTPosition }
  | { type: 'not'; expr: BoolExprNode; pos: ASTPosition }
  | { type: 'signal_ref'; signalType: string; signalName: string; pos: ASTPosition }

export interface ASTSignalDecl {
  signalType: string
  name: string
  fields: DSLFieldObject
  pos: ASTPosition
}

export interface ASTProjectionPartitionDecl {
  name: string
  semantics?: string
  temperature?: number
  members: string[]
  default?: string
  pos: ASTPosition
}

export interface ASTProjectionScoreInput {
  signalType: string
  signalName: string
  weight: number
  valueSource?: string
  match?: number
  miss?: number
}

export interface ASTProjectionScoreDecl {
  name: string
  method?: string
  inputs?: ASTProjectionScoreInput[]
  pos: ASTPosition
}

export interface ASTProjectionMappingCalibration {
  method?: string
  slope?: number
}

export interface ASTProjectionMappingOutput {
  name: string
  lt?: number
  lte?: number
  gt?: number
  gte?: number
}

export interface ASTProjectionMappingDecl {
  name: string
  source?: string
  method?: string
  calibration?: ASTProjectionMappingCalibration
  outputs?: ASTProjectionMappingOutput[]
  pos: ASTPosition
}

export interface ASTModelRef {
  model: string
  reasoning?: boolean
  reasoningDescription?: string
  effort?: string
  lora?: string
  paramSize?: string
  weight?: number
  reasoningFamily?: string
  pos: ASTPosition
}

export interface ASTAlgoSpec {
  algoType: string
  fields: DSLFieldObject
  pos: ASTPosition
}

export interface ASTPluginRef {
  name: string
  fields?: DSLFieldObject
  pos: ASTPosition
}

export interface ASTRouteDecl {
  name: string
  description?: string
  priority: number
  tier?: number
  when: BoolExprNode | null
  models: ASTModelRef[]
  algorithm?: ASTAlgoSpec
  plugins: ASTPluginRef[]
  pos: ASTPosition
}

export interface ASTModelDecl {
  name: string
  fields: DSLFieldObject
  pos: ASTPosition
}

export interface ASTPluginDecl {
  name: string
  pluginType: string
  fields: DSLFieldObject
  pos: ASTPosition
}

export interface ASTTestEntryDecl {
  query: string
  routeName: string
  pos: ASTPosition
}

export interface ASTTestBlockDecl {
  name: string
  entries: ASTTestEntryDecl[]
  pos: ASTPosition
}

export interface ASTEntrypointDecl {
  name: string
  aliases?: string[]
  recipe?: string
  assignments?: Record<string, ASTEntrypointAssignmentSetDecl>
  rules?: ASTEntrypointRuleDecl[]
  pos: ASTPosition
}

export interface ASTEntrypointRuleDecl {
  name: string
  matches?: ASTEntrypointMatchDecl[]
  recipe: string
  assignments: Record<string, ASTEntrypointAssignmentSetDecl>
  pos: ASTPosition
}

export interface ASTEntrypointMatchDecl {
  claim?: { name: string; exact: DSLFieldValue }
  path?: { exact?: string; prefix?: string }
}

export interface ASTEntrypointAssignmentSetDecl {
  models: ASTEntrypointAssignmentDecl[]
  fallback?: { strategy: string; on: string[] }
}

export interface ASTEntrypointAssignmentDecl {
  model: string
  priority: number
  weight?: string
  loraName?: string
  reasoning?: { enabled: boolean; effort?: string; description?: string }
  pos: ASTPosition
}

export interface ASTRecipeDecl {
  name: string
  description?: string
  program: ASTProgram
  pos: ASTPosition
}

export interface ASTProgram {
  strategy?: string
  entrypoints?: ASTEntrypointDecl[]
  recipes?: ASTRecipeDecl[]
  signals: ASTSignalDecl[]
  projectionPartitions?: ASTProjectionPartitionDecl[]
  projectionScores?: ASTProjectionScoreDecl[]
  projectionMappings?: ASTProjectionMappingDecl[]
  routes: ASTRouteDecl[]
  models?: ASTModelDecl[]
  plugins: ASTPluginDecl[]
  testBlocks?: ASTTestBlockDecl[]
}

// ---------- WASM Result Types ----------

export interface CompileResult {
  yaml: string
  crd?: string
  recipeDocuments?: Array<{
    name?: string
    description?: string
    document: Record<string, unknown>
  }>
  diagnostics: Diagnostic[]
  ast?: ASTProgram
  error?: string
}

export interface ValidateResult {
  diagnostics: Diagnostic[]
  errorCount: number
  symbols?: SymbolTable
  error?: string
}

export interface ParseASTResult {
  ast?: ASTProgram
  diagnostics: Diagnostic[]
  symbols?: SymbolTable
  errorCount: number
  error?: string
}

export interface DecompileResult {
  dsl: string
  error?: string
}

export interface FormatResult {
  dsl: string
  error?: string
}

// ---------- Editor State ----------

export type EditorMode = 'dsl' | 'visual'

export interface EditorState {
  /** Current DSL source text in the editor */
  dslSource: string
  /** Compiled YAML output */
  yamlOutput: string
  /** Compiled CRD output */
  crdOutput: string
  /** Current diagnostics from validation */
  diagnostics: Diagnostic[]
  /** Whether WASM runtime is loaded and ready */
  wasmReady: boolean
  /** Loading state for async operations */
  loading: boolean
  /** Current active editor mode */
  mode: EditorMode
  /** Whether there are unsaved changes */
  dirty: boolean
  /** Last successful compile timestamp */
  lastCompileAt: number | null
}

// ---------- WASM Bridge Interface ----------

export interface WasmBridge {
  /** Whether the WASM module is loaded */
  ready: boolean
  /** Load and initialize the WASM module */
  init(): Promise<void>
  /** Compile DSL → YAML + CRD + AST + diagnostics */
  compile(dsl: string): CompileResult
  /** Validate DSL (fast, no compile) */
  validate(dsl: string): ValidateResult
  /** Parse DSL → AST + diagnostics + symbols (no compile, for Visual Builder) */
  parseAST(dsl: string): ParseASTResult
  /** Decompile YAML → DSL */
  decompile(yaml: string): DecompileResult
  /** Format DSL source */
  format(dsl: string): FormatResult
}
