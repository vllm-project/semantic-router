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
  DeployStep,
  DeployResult,
  ConfigVersion,
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

  /** Fetch current router config YAML and decompile to DSL. */
  loadFromRouter(): Promise<void>

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

  // --- Deploy actions ---

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
  deploying: false,
  deployStep: null,
  deployResult: null,
  showDeployConfirm: false,
  configVersions: [],
  deployPreviewCurrent: '',
  deployPreviewMerged: '',
  deployPreviewLoading: false,
  deployPreviewError: null,
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

    console.log('[dslStore.compile] Compiling DSL: source size=%d', dslSource.length)
    // Check if DSL source contains test_route
    const routeNames = dslSource.match(/ROUTE\s+(\w+)/g)
    console.log('[dslStore.compile] ROUTE declarations in DSL source:', routeNames)
    set({ loading: true })
    try {
      const result: CompileResult = wasmBridge.compile(dslSource)

      // Log compile result summary
      console.log('[dslStore.compile] Compile result: yaml size=%d, crd size=%d, diagnostics=%d, error=%s',
        result.yaml?.length ?? 0, result.crd?.length ?? 0,
        result.diagnostics?.length ?? 0, result.error ?? 'none')
      if (result.diagnostics?.length) {
        console.log('[dslStore.compile] Diagnostics:', result.diagnostics)
      }

      // Quick count of decisions in YAML output
      if (result.yaml) {
        const decMatch = result.yaml.match(/^\s*- name:/gm)
        console.log('[dslStore.compile] YAML "- name:" lines count=%d', decMatch?.length ?? 0)
      }

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
      console.error('[dslStore.compile] Compile threw error:', msg)
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

  async loadFromRouter() {
    const { wasmReady } = get()
    if (!wasmReady) throw new Error('WASM not ready')

    const resp = await fetch('/api/router/config/yaml')
    if (!resp.ok) {
      throw new Error(`Failed to fetch config: HTTP ${resp.status}`)
    }
    const yaml = await resp.text()
    if (!yaml.trim()) {
      throw new Error('Router config is empty')
    }
    get().importYaml(yaml)
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

  // --- Deploy actions ---

  requestDeploy() {
    const { yamlOutput, dslSource, wasmReady, dirty } = get()
    if (!wasmReady || !dslSource.trim()) return

    // Re-compile if DSL was modified since last compile, or never compiled
    if (!yamlOutput || dirty) {
      get().compile()
    }

    // Check for compile errors
    const { diagnostics: diags, yamlOutput: yaml } = get()
    const hasErrors = diags.some(d => d.level === 'error')
    if (hasErrors || !yaml) {
      set({
        deployResult: {
          status: 'error',
          message: 'Cannot deploy: DSL has compilation errors. Fix errors and compile first.',
        },
        showDeployConfirm: false,
      })
      return
    }

    // Show modal and fetch preview diff
    set({
      showDeployConfirm: true,
      deployResult: null,
      deployPreviewCurrent: '',
      deployPreviewMerged: '',
      deployPreviewLoading: true,
      deployPreviewError: null,
    })

    // Fetch preview asynchronously
    fetch('/api/router/config/deploy/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ yaml }),
    })
      .then(async (resp) => {
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}))
          throw new Error(data.message || data.error || 'Failed to fetch preview')
        }
        return resp.json()
      })
      .then((data: { current: string; preview: string }) => {
        set({
          deployPreviewCurrent: data.current,
          deployPreviewMerged: data.preview,
          deployPreviewLoading: false,
        })
      })
      .catch((err) => {
        set({
          deployPreviewLoading: false,
          deployPreviewError: err instanceof Error ? err.message : String(err),
        })
      })
  },

  async executeDeploy() {
    const { yamlOutput, dslSource } = get()
    if (!yamlOutput) return

    console.log('[dslStore.executeDeploy] Sending deploy: YAML size=%d, DSL size=%d', yamlOutput.length, dslSource.length)

    set({ deploying: true, deployStep: 'validating', showDeployConfirm: false, deployResult: null })

    try {
      // Step: validating → backing_up → writing → reloading → done
      set({ deployStep: 'backing_up' })
      await new Promise(r => setTimeout(r, 200)) // Small delay for UX

      set({ deployStep: 'writing' })
      const resp = await fetch('/api/router/config/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ yaml: yamlOutput, dsl: dslSource }),
      })

      const data = await resp.json()

      if (!resp.ok) {
        set({
          deploying: false,
          deployStep: 'error',
          deployResult: {
            status: 'error',
            message: data.message || data.error || 'Deploy failed',
          },
        })
        return
      }

      // Wait for router reload (poll status)
      set({ deployStep: 'reloading' })
      let healthy = false
      for (let i = 0; i < 10; i++) {
        await new Promise(r => setTimeout(r, 500))
        try {
          const statusResp = await fetch('/api/status')
          if (statusResp.ok) {
            healthy = true
            break
          }
        } catch {
          // continue polling
        }
      }

      set({
        deploying: false,
        deployStep: 'done',
        deployResult: {
          status: 'success',
          version: data.version,
          message: healthy
            ? `Deployed v${data.version} — Router reloaded successfully.`
            : `Deployed v${data.version} — Router reload status unknown (check logs).`,
        },
        dirty: false,
      })

      // Refresh versions list
      get().fetchVersions()

      // Notify other components (e.g. DashboardPage) to refresh config
      window.dispatchEvent(new CustomEvent('config-deployed'))
    } catch (err) {
      set({
        deploying: false,
        deployStep: 'error',
        deployResult: {
          status: 'error',
          message: `Deploy failed: ${err instanceof Error ? err.message : String(err)}`,
        },
      })
    }
  },

  dismissDeploy() {
    set({
      showDeployConfirm: false,
      deployResult: null,
      deployStep: null,
      deployPreviewCurrent: '',
      deployPreviewMerged: '',
      deployPreviewLoading: false,
      deployPreviewError: null,
    })
  },

  async rollback(version: string) {
    set({ deploying: true, deployStep: 'writing', deployResult: null })

    try {
      const resp = await fetch('/api/router/config/rollback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ version }),
      })

      const data = await resp.json()

      if (!resp.ok) {
        set({
          deploying: false,
          deployStep: 'error',
          deployResult: {
            status: 'error',
            message: data.message || 'Rollback failed',
          },
        })
        return
      }

      set({ deployStep: 'reloading' })
      await new Promise(r => setTimeout(r, 2000))

      set({
        deploying: false,
        deployStep: 'done',
        deployResult: {
          status: 'success',
          version: data.version,
          message: `Rolled back to v${data.version}. Router will reload automatically.`,
        },
      })

      get().fetchVersions()
    } catch (err) {
      set({
        deploying: false,
        deployStep: 'error',
        deployResult: {
          status: 'error',
          message: `Rollback failed: ${err instanceof Error ? err.message : String(err)}`,
        },
      })
    }
  },

  async fetchVersions() {
    try {
      const resp = await fetch('/api/router/config/versions')
      if (resp.ok) {
        const versions = await resp.json()
        set({ configVersions: versions || [] })
      }
    } catch {
      // silently fail
    }
  },
}))

// Eagerly start WASM init on store creation (module-level side-effect).
// This overlaps with network fetch of JS/CSS bundles for faster perceived load.
useDSLStore.getState().initWasm()
