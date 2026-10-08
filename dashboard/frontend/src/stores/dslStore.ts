/**
 * Zustand store for DSL editor state management.
 *
 * Manages:
 * - DSL source text, YAML/CRD output, diagnostics
 * - Compiler availability (init, ready state)
 * - Editor mode switching (DSL / Visual)
 * - Debounced validation on keystroke
 * - Full compile on demand
 * - Decompile router YAML → DSL-owned models, routing, entrypoints, and recipes
 * - Format (canonical pretty-print)
 */

import { create } from 'zustand'
import { dslCompiler } from '@/lib/dslCompiler'
import {
  updateModel,
  addModel as addModelMut,
  deleteModel as deleteModelMut,
  updateSignal,
  addSignal as addSignalMut,
  deleteSignal as deleteSignalMut,
  updateProjectionPartition as updateProjectionPartitionMut,
  addProjectionPartition as addProjectionPartitionMut,
  deleteProjectionPartition as deleteProjectionPartitionMut,
  updateProjection as updateProjectionMut,
  addProjection as addProjectionMut,
  deleteProjection as deleteProjectionMut,
  updatePlugin,
  addPlugin as addPluginMut,
  deletePlugin as deletePluginMut,
  deleteRoute as deleteRouteMut,
  updateRoute as updateRouteMut,
  addRoute as addRouteMut,
} from '@/lib/dslMutations'
import type { RouteInput } from '@/lib/dslMutations'
import type { EditorMode, CompileResult, ValidateResult, DSLFieldObject } from '@/types/dsl'
import type { DSLStore } from './dslStoreTypes'
import { initialDSLState, type DeployStatusResponse } from './dslStoreSupport'
import { renderCanonicalYaml } from './dslStoreYamlSupport'

// ---------- Debounce helper ----------

let validateTimer: ReturnType<typeof setTimeout> | null = null
const VALIDATE_DEBOUNCE_MS = 300
let renderedYamlRequestId = 0
let sourceRevision = 0
let compileRequestId = 0
let analysisRequestId = 0
let importRequestId = 0
let initRequestId = 0
let editorRequests = new AbortController()

// ---------- Store ----------

export const useDSLStore = create<DSLStore>((set, get) => ({
  ...initialDSLState,

  async initCompiler() {
    if (get().compilerReady) return
    const requestId = ++initRequestId
    set({ loading: true, compilerError: null })
    try {
      await dslCompiler.init()
      if (requestId !== initRequestId) return
      set({ compilerReady: true, loading: false })
    } catch (err) {
      if (requestId !== initRequestId) return
      const msg = err instanceof Error ? err.message : String(err)
      set({ compilerError: msg, loading: false })
      console.error('[DSLStore] Compiler init failed:', msg)
    }
  },

  pauseEditorWork() {
    if (validateTimer) clearTimeout(validateTimer)
    sourceRevision++
    compileRequestId++
    analysisRequestId++
    importRequestId++
    renderedYamlRequestId++
    initRequestId++
    editorRequests.abort()
    editorRequests = new AbortController()
    dslCompiler.cancelPending()
    // Preserve the draft and outputs across navigation; only cancel reads and
    // compiler work. An already submitted deployment keeps its own lifecycle.
    set({ loading: false })
  },

  setDslSource(source: string) {
    set({
      dslSource: source,
      dirty: true,
      diagnostics: [],
      compileError: null,
    })

    // Debounced auto-validation
    if (validateTimer) clearTimeout(validateTimer)
    validateTimer = setTimeout(() => {
      const state = get()
      if (state.compilerReady && state.dslSource) {
        void (state.mode === 'visual' ? state.parseAST() : state.validate())
      }
    }, VALIDATE_DEBOUNCE_MS)
  },

  async compile() {
    const { dslSource, compilerReady, baseConfigYaml } = get()
    if (!compilerReady) return
    if (!dslSource.trim()) {
      set({
        renderedYamlOutput: '',
        yamlOutput: '',
        crdOutput: '',
        diagnostics: [],
        compileError: null,
        dirty: false,
      })
      return
    }

    const requestId = ++compileRequestId
    const revision = sourceRevision
    set({ loading: true })
    try {
      const result: CompileResult = await dslCompiler.compile(dslSource)

      if (requestId !== compileRequestId || revision !== sourceRevision) return
      const compiledYaml = result.yaml || ''
      set({
        renderedYamlOutput: compiledYaml,
        yamlOutput: compiledYaml,
        crdOutput: result.crd || '',
        diagnostics: result.diagnostics || [],
        ast: result.ast || null,
        compileError: result.error || null,
        dirty: false,
        lastCompileAt: Date.now(),
        loading: false,
      })
      if (compiledYaml) {
        const requestId = ++renderedYamlRequestId
        void renderCanonicalYaml(compiledYaml, dslSource, baseConfigYaml, editorRequests.signal)
          .then((renderedYamlOutput) => {
            if (requestId !== renderedYamlRequestId || get().yamlOutput !== compiledYaml) return
            set({ renderedYamlOutput })
          })
          .catch((error) => {
            if (requestId !== renderedYamlRequestId) return
            console.warn('[dslStore.compile] Full YAML preview unavailable:', error)
          })
      }
    } catch (err) {
      if (requestId !== compileRequestId || revision !== sourceRevision) return
      const msg = err instanceof Error ? err.message : String(err)
      console.error('[dslStore.compile] Compile threw error:', msg)
      set({
        compileError: msg,
        diagnostics: [],
        symbols: null,
        ast: null,
        renderedYamlOutput: '',
        yamlOutput: '',
        crdOutput: '',
        loading: false,
      })
    } finally {
      if (requestId === compileRequestId) set({ loading: false })
    }
  },

  async validate() {
    const { dslSource, compilerReady } = get()
    if (!compilerReady) return
    if (!dslSource.trim()) {
      set({ diagnostics: [], compileError: null })
      return
    }

    const revision = sourceRevision
    const requestId = ++analysisRequestId
    try {
      const result: ValidateResult = await dslCompiler.validate(dslSource)
      if (requestId !== analysisRequestId || revision !== sourceRevision) return
      set({
        diagnostics: result.diagnostics || [],
        symbols: result.symbols || null,
        compileError: result.error || null,
      })
    } catch (err) {
      if (requestId !== analysisRequestId || revision !== sourceRevision) return
      console.error('[DSLStore] validate error:', err)
      set({
        diagnostics: [],
        symbols: null,
        compileError: err instanceof Error ? err.message : String(err),
      })
    }
  },

  async parseAST() {
    const { dslSource, compilerReady } = get()
    if (!compilerReady) return
    if (!dslSource.trim()) {
      set({ ast: null, diagnostics: [], symbols: null, compileError: null })
      return
    }

    const revision = sourceRevision
    const requestId = ++analysisRequestId
    try {
      const result = await dslCompiler.parseAST(dslSource)
      if (requestId !== analysisRequestId || revision !== sourceRevision) return
      set({
        ast: result.ast || null,
        diagnostics: result.diagnostics || [],
        symbols: result.symbols || null,
        compileError: result.error || null,
      })
    } catch (err) {
      if (requestId !== analysisRequestId || revision !== sourceRevision) return
      console.error('[DSLStore] parseAST error:', err)
      set({
        ast: null,
        diagnostics: [],
        symbols: null,
        compileError: err instanceof Error ? err.message : String(err),
      })
    }
  },

  async decompile(yaml: string): Promise<string> {
    const { compilerReady } = get()
    if (!compilerReady) throw new Error('Compiler not ready')

    const result = await dslCompiler.decompile(yaml)
    if (result.error) {
      throw new Error(result.error)
    }
    return result.dsl
  },

  async format() {
    const { dslSource, compilerReady } = get()
    if (!compilerReady || !dslSource.trim()) return

    const revision = sourceRevision
    try {
      const result = await dslCompiler.format(dslSource)
      if (revision !== sourceRevision) return
      if (result.error) {
        set({ compileError: result.error, diagnostics: [] })
        return
      }
      set({
        dslSource: result.dsl,
        dirty: true,
        compileError: null,
      })
      get().validate()
    } catch (err) {
      if (revision !== sourceRevision) return
      set({
        compileError: err instanceof Error ? err.message : String(err),
        diagnostics: [],
      })
    }
  },

  setMode(mode: EditorMode) {
    set({ mode })
  },

  reset() {
    if (validateTimer) clearTimeout(validateTimer)
    sourceRevision++
    importRequestId++
    set({ ...initialDSLState, compilerReady: get().compilerReady })
  },

  loadDsl(source: string) {
    set({
      dslSource: source,
      dirty: false,
      savedSource: source,
      diagnostics: [],
      compileError: null,
      baseConfigYaml: '',
      renderedYamlOutput: '',
    })
    // Trigger validation after load
    const state = get()
    if (state.compilerReady && source.trim()) {
      void (state.mode === 'visual' ? state.parseAST() : state.validate())
    }
  },

  async importYaml(yaml: string) {
    const revision = sourceRevision
    const requestId = ++importRequestId
    const dsl = await get().decompile(yaml)
    if (requestId !== importRequestId || revision !== sourceRevision) {
      throw new Error(
        'The source changed during import. Import again to replace the current draft.',
      )
    }
    if (!dsl) {
      throw new Error('Failed to decompile YAML')
    }
    set({
      dslSource: dsl,
      dirty: false,
      savedSource: dsl,
      diagnostics: [],
      compileError: null,
      baseConfigYaml: yaml,
      renderedYamlOutput: yaml,
    })
    const state = get()
    if (state.compilerReady && dsl.trim()) {
      void (state.mode === 'visual' ? state.parseAST() : state.validate())
    }
  },

  async loadFromRouter() {
    const { compilerReady } = get()
    if (!compilerReady) throw new Error('Compiler not ready')

    const revision = sourceRevision
    const resp = await fetch('/api/router/config/yaml', { signal: editorRequests.signal })
    if (!resp.ok) {
      throw new Error(`Failed to fetch config: HTTP ${resp.status}`)
    }
    const yaml = await resp.text()
    if (!yaml.trim()) {
      throw new Error('Router config is empty')
    }
    if (revision !== sourceRevision)
      throw new Error('The draft changed while loading. Load again to replace it.')
    await get().importYaml(yaml)
  },

  // --- Visual Builder mutations (Phase 2) ---

  mutateModel(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = updateModel(dslSource, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  addModel(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = addModelMut(dslSource, name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  deleteModel(name: string) {
    const { dslSource, compilerReady } = get()
    const newSrc = deleteModelMut(dslSource, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  mutateSignal(signalType: string, name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = updateSignal(dslSource, signalType, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  addSignal(signalType: string, name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = addSignalMut(dslSource, signalType, name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  deleteSignal(signalType: string, name: string) {
    const { dslSource, compilerReady } = get()
    const newSrc = deleteSignalMut(dslSource, signalType, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  mutateProjectionPartition(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = updateProjectionPartitionMut(dslSource, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  addProjectionPartition(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = addProjectionPartitionMut(dslSource, name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  deleteProjectionPartition(name: string) {
    const { dslSource, compilerReady } = get()
    const newSrc = deleteProjectionPartitionMut(dslSource, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  mutateProjectionScore(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = updateProjectionMut(dslSource, 'score', name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  addProjectionScore(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = addProjectionMut(dslSource, 'score', name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  deleteProjectionScore(name: string) {
    const { dslSource, compilerReady } = get()
    const newSrc = deleteProjectionMut(dslSource, 'score', name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  mutateProjectionMapping(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = updateProjectionMut(dslSource, 'mapping', name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  addProjectionMapping(name: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = addProjectionMut(dslSource, 'mapping', name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  deleteProjectionMapping(name: string) {
    const { dslSource, compilerReady } = get()
    const newSrc = deleteProjectionMut(dslSource, 'mapping', name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  mutatePlugin(name: string, pluginType: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = updatePlugin(dslSource, name, pluginType, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  addPlugin(name: string, pluginType: string, fields: DSLFieldObject) {
    const { dslSource, compilerReady } = get()
    const newSrc = addPluginMut(dslSource, name, pluginType, fields)
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  deletePlugin(name: string, pluginType: string) {
    const { dslSource, compilerReady } = get()
    const newSrc = deletePluginMut(dslSource, name, pluginType)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  deleteRoute(name: string) {
    const { dslSource, compilerReady } = get()
    const newSrc = deleteRouteMut(dslSource, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  mutateRoute(name: string, input: RouteInput) {
    const { dslSource, compilerReady } = get()
    const newSrc = updateRouteMut(dslSource, name, input)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  addRoute(name: string, input: RouteInput) {
    const { dslSource, compilerReady } = get()
    const newSrc = addRouteMut(dslSource, name, input)
    set({ dslSource: newSrc, dirty: true })
    if (compilerReady) get().parseAST()
  },

  // --- Deploy actions ---

  async requestDeploy() {
    const { yamlOutput, dslSource, compilerReady, dirty, baseConfigYaml } = get()
    if (!compilerReady || !dslSource.trim()) return

    const revision = sourceRevision
    // Re-compile if DSL was modified since last compile, or never compiled
    if (!yamlOutput || dirty) {
      await get().compile()
    }

    if (revision !== sourceRevision) return

    // Check for compile errors
    const { diagnostics: diags, yamlOutput: yaml, compileError } = get()
    const hasErrors = diags.some((d) => d.level === 'error')
    if (compileError || hasErrors || !yaml) {
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
      body: JSON.stringify({ yaml, dsl: dslSource, baseYaml: baseConfigYaml, mode: 'replace' }),
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
    const { yamlOutput, dslSource, baseConfigYaml } = get()
    if (!yamlOutput) return

    console.log(
      '[dslStore.executeDeploy] Sending deploy: YAML size=%d, DSL size=%d',
      yamlOutput.length,
      dslSource.length,
    )

    set({ deploying: true, deployStep: 'validating', showDeployConfirm: false, deployResult: null })

    try {
      // Step: validating → backing_up → writing → reloading → done
      set({ deployStep: 'backing_up' })
      await new Promise((r) => setTimeout(r, 200)) // Small delay for UX

      set({ deployStep: 'writing' })
      const resp = await fetch('/api/router/config/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          yaml: yamlOutput,
          dsl: dslSource,
          baseYaml: baseConfigYaml,
          mode: 'replace',
        }),
      })

      const responseText = await resp.text()
      let data: { status?: string; version?: string; message?: string; error?: string } = {}
      try {
        data = responseText ? (JSON.parse(responseText) as typeof data) : {}
      } catch {
        data = responseText ? { message: responseText } : {}
      }

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

      if (data.status === 'persisted' || data.status === 'restart_required') {
        set({
          deploying: false,
          deployStep: 'done',
          deployResult: {
            status: 'success',
            version: data.version,
            message:
              data.message || 'Configuration saved. Roll out Router and Envoy to activate it.',
          },
          savedSource: dslSource,
        })
        get().fetchVersions()
        return
      }

      // Wait for runtime reload (poll actual health status)
      set({ deployStep: 'reloading' })
      let healthy = false
      // A standalone stack reports no Envoy service: the Router serves the listeners.
      let reloaded = 'Router and Envoy'
      for (let i = 0; i < 10; i++) {
        await new Promise((r) => setTimeout(r, 500))
        try {
          const statusResp = await fetch('/api/status')
          if (!statusResp.ok) continue

          const statusData = (await statusResp.json()) as DeployStatusResponse
          const routerHealthy =
            statusData.services?.find((service) => service.name === 'Router')?.healthy === true
          const envoyService = statusData.services?.find((service) => service.name === 'Envoy')
          const envoyHealthy = envoyService ? envoyService.healthy === true : true
          reloaded = envoyService ? 'Router and Envoy' : 'Router'

          if (statusData.overall === 'healthy' && routerHealthy && envoyHealthy) {
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
            ? `Deployed v${data.version} — ${reloaded} reloaded successfully.`
            : `Deployed v${data.version} — Runtime reload status unknown (check logs).`,
        },
        savedSource: dslSource,
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

      if (data.status === 'persisted' || data.status === 'restart_required') {
        set({
          deploying: false,
          deployStep: 'done',
          deployResult: {
            status: 'success',
            version: data.version,
            message: data.message || 'Rollback saved. Roll out Router and Envoy to activate it.',
          },
        })
        get().fetchVersions()
        return
      }

      set({ deployStep: 'reloading' })
      await new Promise((r) => setTimeout(r, 2000))

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

/** The unsaved signal, derived so no mutation can leave it stale: the source
 * differs from the snapshot taken at the last load, import, reset, or
 * successful deploy. */
export const selectHasUnsavedChanges = (state: DSLStore) => state.dslSource !== state.savedSource

// The edits live in this store, so the unload guard belongs to the store's
// lifetime rather than to any page: in-app navigation unmounts the editor and
// its listener, which would drop pending edits to a reload on the next route.
// One listener derives the signal when the event fires, so no mutation can
// leave it stale. Vitest runs in Node, where no window exists.
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', (event: BeforeUnloadEvent) => {
    if (!selectHasUnsavedChanges(useDSLStore.getState())) return
    event.preventDefault()
    event.returnValue = ''
  })
}

// Any source edit (including scoped visual mutations) invalidates in-flight
// analysis. Comparing text alone would miss an A → B → A edit sequence.
useDSLStore.subscribe((state, previous) => {
  if (state.dslSource !== previous.dslSource || state.baseConfigYaml !== previous.baseConfigYaml) {
    sourceRevision++
    renderedYamlRequestId++
  }
})
