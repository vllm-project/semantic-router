/**
 * Zustand store for DSL editor state management.
 *
 * Manages:
 * - DSL source text, YAML/CRD output, diagnostics
 * - WASM lifecycle (init, ready state)
 * - Editor mode switching (DSL / Visual / NL)
 * - Debounced validation on keystroke
 * - Full compile on demand
 * - Decompile (YAML → DSL) for import workflows
 * - Format (canonical pretty-print)
 */

import { create } from 'zustand'
import { wasmBridge } from '@/lib/wasm'
import {
  updateSignal,
  addSignal as addSignalMut,
  deleteSignal as deleteSignalMut,
  updatePlugin,
  addPlugin as addPluginMut,
  deletePlugin as deletePluginMut,
  updateBackend,
  addBackend as addBackendMut,
  deleteBackend as deleteBackendMut,
  deleteRoute as deleteRouteMut,
  updateRoute as updateRouteMut,
  addRoute as addRouteMut,
  updateGlobal as updateGlobalMut,
} from '@/lib/dslMutations'
import type { RouteInput } from '@/lib/dslMutations'
import type {
  Diagnostic,
  EditorMode,
  CompileResult,
  ValidateResult,
  SymbolTable,
  ASTProgram,
} from '@/types/dsl'

// ---------- Store State ----------

interface DSLState {
  // --- Editor content ---
  dslSource: string
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

// ---------- Store Actions ----------

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

  /** Decompile YAML → DSL (for import from existing config). */
  decompile(yaml: string): string | null

  /** Format the current DSL source. */
  format(): void

  /** Switch editor mode. */
  setMode(mode: EditorMode): void

  /** Reset editor state to initial values. */
  reset(): void

  /** Load DSL source from external input (e.g., file import). */
  loadDsl(source: string): void

  /** Load YAML and decompile to DSL. */
  importYaml(yaml: string): void

  // --- Visual Builder mutations (Phase 2) ---

  /** Update a signal's fields in DSL source text, then re-parse AST. */
  mutateSignal(signalType: string, name: string, fields: Record<string, unknown>): void

  /** Add a new signal to DSL source text, then re-parse AST. */
  addSignal(signalType: string, name: string, fields: Record<string, unknown>): void

  /** Delete a signal from DSL source text, then re-parse AST. */
  deleteSignal(signalType: string, name: string): void

  /** Update a plugin declaration's fields, then re-parse AST. */
  mutatePlugin(name: string, pluginType: string, fields: Record<string, unknown>): void

  /** Add a new plugin declaration, then re-parse AST. */
  addPlugin(name: string, pluginType: string, fields: Record<string, unknown>): void

  /** Delete a plugin declaration, then re-parse AST. */
  deletePlugin(name: string, pluginType: string): void

  /** Update a backend declaration's fields, then re-parse AST. */
  mutateBackend(backendType: string, name: string, fields: Record<string, unknown>): void

  /** Add a new backend declaration, then re-parse AST. */
  addBackend(backendType: string, name: string, fields: Record<string, unknown>): void

  /** Delete a backend declaration, then re-parse AST. */
  deleteBackend(backendType: string, name: string): void

  /** Delete a route declaration, then re-parse AST. */
  deleteRoute(name: string): void

  /** Update a route declaration, then re-parse AST. */
  mutateRoute(name: string, input: RouteInput): void

  /** Add a new route, then re-parse AST. */
  addRoute(name: string, input: RouteInput): void

  /** Update the GLOBAL block's fields, then re-parse AST. */
  mutateGlobal(fields: Record<string, unknown>): void
}

export type DSLStore = DSLState & DSLActions

// ---------- Debounce helper ----------

let validateTimer: ReturnType<typeof setTimeout> | null = null
const VALIDATE_DEBOUNCE_MS = 300

// ---------- Initial state ----------

const initialState: DSLState = {
  dslSource: '',
  yamlOutput: '',
  crdOutput: '',
  diagnostics: [],
  symbols: null,
  ast: null,
  wasmReady: false,
  wasmError: null,
  loading: false,
  compileError: null,
  mode: 'dsl',
  dirty: false,
  lastCompileAt: null,
}

// ---------- Store ----------

export const useDSLStore = create<DSLStore>((set, get) => ({
  ...initialState,

  async initWasm() {
    if (get().wasmReady) return
    set({ loading: true, wasmError: null })
    try {
      await wasmBridge.init()
      set({ wasmReady: true, loading: false })
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({ wasmError: msg, loading: false })
      console.error('[DSLStore] WASM init failed:', msg)
    }
  },

  setDslSource(source: string) {
    set({ dslSource: source, dirty: true })

    // Debounced auto-validation
    if (validateTimer) clearTimeout(validateTimer)
    validateTimer = setTimeout(() => {
      const state = get()
      if (state.wasmReady && state.dslSource) {
        state.validate()
      }
    }, VALIDATE_DEBOUNCE_MS)
  },

  compile() {
    const { dslSource, wasmReady } = get()
    if (!wasmReady) return
    if (!dslSource.trim()) {
      set({ yamlOutput: '', crdOutput: '', diagnostics: [], compileError: null, dirty: false })
      return
    }

    set({ loading: true })
    try {
      const result: CompileResult = wasmBridge.compile(dslSource)
      set({
        yamlOutput: result.yaml || '',
        crdOutput: result.crd || '',
        diagnostics: result.diagnostics || [],
        ast: result.ast || null,
        compileError: result.error || null,
        dirty: false,
        lastCompileAt: Date.now(),
        loading: false,
      })
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      set({ compileError: msg, loading: false })
    }
  },

  validate() {
    const { dslSource, wasmReady } = get()
    if (!wasmReady) return
    if (!dslSource.trim()) {
      set({ diagnostics: [], compileError: null })
      return
    }

    try {
      const result: ValidateResult = wasmBridge.validate(dslSource)
      set({
        diagnostics: result.diagnostics || [],
        symbols: result.symbols || null,
        compileError: result.error || null,
      })
    } catch (err) {
      console.error('[DSLStore] validate error:', err)
    }
  },

  parseAST() {
    const { dslSource, wasmReady } = get()
    if (!wasmReady) return
    if (!dslSource.trim()) {
      set({ ast: null, diagnostics: [], symbols: null, compileError: null })
      return
    }

    try {
      const result = wasmBridge.parseAST(dslSource)
      set({
        ast: result.ast || null,
        diagnostics: result.diagnostics || [],
        symbols: result.symbols || null,
        compileError: result.error || null,
      })
    } catch (err) {
      console.error('[DSLStore] parseAST error:', err)
    }
  },

  decompile(yaml: string): string | null {
    const { wasmReady } = get()
    if (!wasmReady) return null

    const result = wasmBridge.decompile(yaml)
    if (result.error) {
      console.error('[DSLStore] decompile error:', result.error)
      return null
    }
    return result.dsl
  },

  format() {
    const { dslSource, wasmReady } = get()
    if (!wasmReady || !dslSource.trim()) return

    try {
      const result = wasmBridge.format(dslSource)
      if (result.error) {
        console.error('[DSLStore] format error:', result.error)
        return
      }
      set({ dslSource: result.dsl, dirty: true })
    } catch (err) {
      console.error('[DSLStore] format error:', err)
    }
  },

  setMode(mode: EditorMode) {
    set({ mode })
  },

  reset() {
    if (validateTimer) clearTimeout(validateTimer)
    set({ ...initialState, wasmReady: get().wasmReady })
  },

  loadDsl(source: string) {
    set({ dslSource: source, dirty: false, diagnostics: [], compileError: null })
    // Trigger validation after load
    const state = get()
    if (state.wasmReady && source.trim()) {
      state.validate()
    }
  },

  importYaml(yaml: string) {
    const dsl = get().decompile(yaml)
    if (dsl) {
      get().loadDsl(dsl)
    }
  },

  // --- Visual Builder mutations (Phase 2) ---

  mutateSignal(signalType: string, name: string, fields: Record<string, unknown>) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateSignal(dslSource, signalType, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addSignal(signalType: string, name: string, fields: Record<string, unknown>) {
    const { dslSource, wasmReady } = get()
    const newSrc = addSignalMut(dslSource, signalType, name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deleteSignal(signalType: string, name: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deleteSignalMut(dslSource, signalType, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutatePlugin(name: string, pluginType: string, fields: Record<string, unknown>) {
    const { dslSource, wasmReady } = get()
    const newSrc = updatePlugin(dslSource, name, pluginType, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addPlugin(name: string, pluginType: string, fields: Record<string, unknown>) {
    const { dslSource, wasmReady } = get()
    const newSrc = addPluginMut(dslSource, name, pluginType, fields)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deletePlugin(name: string, pluginType: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deletePluginMut(dslSource, name, pluginType)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutateBackend(backendType: string, name: string, fields: Record<string, unknown>) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateBackend(dslSource, backendType, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addBackend(backendType: string, name: string, fields: Record<string, unknown>) {
    const { dslSource, wasmReady } = get()
    const newSrc = addBackendMut(dslSource, backendType, name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deleteBackend(backendType: string, name: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deleteBackendMut(dslSource, backendType, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deleteRoute(name: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deleteRouteMut(dslSource, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutateRoute(name: string, input: RouteInput) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateRouteMut(dslSource, name, input)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addRoute(name: string, input: RouteInput) {
    const { dslSource, wasmReady } = get()
    const newSrc = addRouteMut(dslSource, name, input)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutateGlobal(fields: Record<string, unknown>) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateGlobalMut(dslSource, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },
}))

// Eagerly start WASM init on store creation (module-level side-effect).
// This overlaps with network fetch of JS/CSS bundles for faster perceived load.
useDSLStore.getState().initWasm()
