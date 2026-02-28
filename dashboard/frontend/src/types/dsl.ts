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
  backends: SymbolInfo[]
  routes: string[]
}

// ---------- AST Types (for Visual Builder) ----------

export interface ASTPosition {
  Line: number
  Column: number
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
  fields: Record<string, unknown>
  pos: ASTPosition
}

export interface ASTModelRef {
  model: string
  reasoning?: boolean
  effort?: string
  lora?: string
  paramSize?: string
  weight?: number
  reasoningFamily?: string
  pos: ASTPosition
}

export interface ASTAlgoSpec {
  algoType: string
  fields: Record<string, unknown>
  pos: ASTPosition
}

export interface ASTPluginRef {
  name: string
  fields?: Record<string, unknown>
  pos: ASTPosition
}

export interface ASTRouteDecl {
  name: string
  description?: string
  priority: number
  when: BoolExprNode | null
  models: ASTModelRef[]
  algorithm?: ASTAlgoSpec
  plugins: ASTPluginRef[]
  pos: ASTPosition
}

export interface ASTPluginDecl {
  name: string
  pluginType: string
  fields: Record<string, unknown>
  pos: ASTPosition
}

export interface ASTBackendDecl {
  backendType: string
  name: string
  fields: Record<string, unknown>
  pos: ASTPosition
}

export interface ASTGlobalDecl {
  fields: Record<string, unknown>
  pos: ASTPosition
}

export interface ASTProgram {
  signals: ASTSignalDecl[]
  routes: ASTRouteDecl[]
  plugins: ASTPluginDecl[]
  backends: ASTBackendDecl[]
  global?: ASTGlobalDecl
}

// ---------- WASM Result Types ----------

export interface CompileResult {
  yaml: string
  crd?: string
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

export type EditorMode = 'dsl' | 'visual' | 'nl'

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
