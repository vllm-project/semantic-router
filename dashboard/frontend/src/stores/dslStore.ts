/**
 * Zustand store for DSL editor state management.
 *
 * Manages:
 * - DSL source text, YAML/CRD output, diagnostics
 * - WASM lifecycle (init, ready state)
 * - Editor mode switching (DSL / Visual / NL)
 * - Debounced validation on keystroke
 * - Full compile on demand
 * - Decompile router YAML → routing-only DSL for import workflows
 * - Format (canonical pretty-print)
 */

import { create } from 'zustand'
import { wasmBridge } from '@/lib/wasm'
import { generateBuilderNLDraftStreaming } from '@/utils/builderNLApi'
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
import type {
  BuilderNLGenerateRequest,
  BuilderNLProgressEvent,
  BuilderNLReview,
  BuilderNLStagedDraft,
  BuilderNLValidation,
  EditorMode,
  CompileResult,
  ValidateResult,
  DSLFieldObject,
} from '@/types/dsl'
import type { DSLState, DSLStore } from './dslStoreTypes'

interface DeployStatusService {
  name?: string
  healthy?: boolean
}

interface DeployStatusResponse {
  overall?: string
  services?: DeployStatusService[]
}

// ---------- Debounce helper ----------

let validateTimer: ReturnType<typeof setTimeout> | null = null
const VALIDATE_DEBOUNCE_MS = 300

function normalizeBuilderNLReview(review: BuilderNLReview | undefined): BuilderNLReview {
  return {
    ready: review?.ready ?? false,
    summary: review?.summary ?? '',
    warnings: Array.isArray(review?.warnings) ? review.warnings : [],
    checks: Array.isArray(review?.checks) ? review.checks : [],
  }
}

function normalizeBuilderNLValidation(
  validation: BuilderNLValidation | undefined,
): BuilderNLValidation {
  return {
    ready: validation?.ready ?? false,
    diagnostics: Array.isArray(validation?.diagnostics) ? validation.diagnostics : [],
    errorCount: typeof validation?.errorCount === 'number' ? validation.errorCount : 0,
    compileError: validation?.compileError || undefined,
  }
}

// ---------- Initial state ----------

const initialState: DSLState = {
  dslSource: '',
  yamlOutput: '',
  crdOutput: '',
  diagnostics: [],
  symbols: null,
  ast: null,
  baseConfigYaml: '',
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
  nlGenerating: false,
  nlGenerateError: null,
  nlStagedDraft: null,
  nlProgressEvents: [],
}

function appendBuilderNLProgress(
  set: (partial: Partial<DSLStore> | ((state: DSLStore) => Partial<DSLStore>)) => void,
  event: BuilderNLProgressEvent,
) {
  console.log(`[builder-nl][${event.phase}] ${event.message}`)
  set((state) => ({
    nlProgressEvents: [...state.nlProgressEvents.slice(-79), event],
  }))
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
    set({
      dslSource: source,
      dirty: false,
      diagnostics: [],
      compileError: null,
      baseConfigYaml: '',
    })
    // Trigger validation after load
    const state = get()
    if (state.wasmReady && source.trim()) {
      state.validate()
    }
  },

  importYaml(yaml: string) {
    const dsl = get().decompile(yaml)
    if (!dsl) {
      throw new Error('Failed to decompile YAML')
    }
    set({
      dslSource: dsl,
      dirty: false,
      diagnostics: [],
      compileError: null,
      baseConfigYaml: yaml,
    })
    const state = get()
    if (state.wasmReady && dsl.trim()) {
      state.validate()
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

  mutateModel(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateModel(dslSource, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addModel(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = addModelMut(dslSource, name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deleteModel(name: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deleteModelMut(dslSource, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutateSignal(signalType: string, name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateSignal(dslSource, signalType, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addSignal(signalType: string, name: string, fields: DSLFieldObject) {
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

  mutateProjectionPartition(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateProjectionPartitionMut(dslSource, name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addProjectionPartition(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = addProjectionPartitionMut(dslSource, name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deleteProjectionPartition(name: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deleteProjectionPartitionMut(dslSource, name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutateProjectionScore(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateProjectionMut(dslSource, 'score', name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addProjectionScore(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = addProjectionMut(dslSource, 'score', name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deleteProjectionScore(name: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deleteProjectionMut(dslSource, 'score', name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutateProjectionMapping(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = updateProjectionMut(dslSource, 'mapping', name, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addProjectionMapping(name: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = addProjectionMut(dslSource, 'mapping', name, fields)
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  deleteProjectionMapping(name: string) {
    const { dslSource, wasmReady } = get()
    const newSrc = deleteProjectionMut(dslSource, 'mapping', name)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  mutatePlugin(name: string, pluginType: string, fields: DSLFieldObject) {
    const { dslSource, wasmReady } = get()
    const newSrc = updatePlugin(dslSource, name, pluginType, fields)
    if (newSrc === dslSource) return
    set({ dslSource: newSrc, dirty: true })
    if (wasmReady) get().parseAST()
  },

  addPlugin(name: string, pluginType: string, fields: DSLFieldObject) {
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

  // --- Deploy actions ---

  requestDeploy() {
    const { yamlOutput, dslSource, wasmReady, dirty, baseConfigYaml } = get()
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
      body: JSON.stringify({ yaml, baseYaml: baseConfigYaml }),
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
        body: JSON.stringify({ yaml: yamlOutput, dsl: dslSource, baseYaml: baseConfigYaml }),
      })

      const responseText = await resp.text()
      let data: { version?: string; message?: string; error?: string } = {}
      try {
        data = responseText ? JSON.parse(responseText) as typeof data : {}
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

      // Wait for runtime reload (poll actual health status)
      set({ deployStep: 'reloading' })
      let healthy = false
      for (let i = 0; i < 10; i++) {
        await new Promise(r => setTimeout(r, 500))
        try {
          const statusResp = await fetch('/api/status')
          if (!statusResp.ok) continue

          const statusData = await statusResp.json() as DeployStatusResponse
          const routerHealthy = statusData.services?.find(service => service.name === 'Router')?.healthy === true
          const envoyService = statusData.services?.find(service => service.name === 'Envoy')
          const envoyHealthy = envoyService ? envoyService.healthy === true : true

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
            ? `Deployed v${data.version} — Router and Envoy reloaded successfully.`
            : `Deployed v${data.version} — Runtime reload status unknown (check logs).`,
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

  async generateFromNaturalLanguage(input: BuilderNLGenerateRequest) {
    const prompt = input.prompt.trim()
    if (!prompt) {
      set({ nlGenerateError: 'Describe the routing behavior you want to build.' })
      return
    }

    set({
      nlGenerating: true,
      nlGenerateError: null,
      nlStagedDraft: null,
      nlProgressEvents: [{
        phase: 'request',
        level: 'info',
        message: 'Sending Builder NL request to the streaming backend.',
        timestamp: Date.now(),
      }],
    })

    try {
      const data = await generateBuilderNLDraftStreaming({
        ...input,
        prompt,
        currentDsl: input.currentDsl?.trim() || '',
      }, (event) => appendBuilderNLProgress(set, event))
      const stagedDraft: BuilderNLStagedDraft = {
        prompt,
        dsl: data.dsl,
        baseYaml: data.baseYaml || get().baseConfigYaml,
        summary: data.summary || '',
        suggestedTestQuery: data.suggestedTestQuery || '',
        review: normalizeBuilderNLReview(data.review),
        validation: normalizeBuilderNLValidation(data.validation),
      }
      set({
        nlGenerating: false,
        nlGenerateError: null,
        nlStagedDraft: stagedDraft,
      })
    } catch (err) {
      appendBuilderNLProgress(set, {
        phase: 'error',
        level: 'error',
        message: err instanceof Error ? err.message : String(err),
        timestamp: Date.now(),
      })
      set({
        nlGenerating: false,
        nlGenerateError: err instanceof Error ? err.message : String(err),
        nlStagedDraft: null,
      })
    }
  },

  applyNaturalLanguageDraft() {
    const stagedDraft = get().nlStagedDraft
    if (!stagedDraft) return

    set({
      dslSource: stagedDraft.dsl,
      baseConfigYaml: stagedDraft.baseYaml,
      dirty: false,
      diagnostics: [],
      compileError: null,
      ast: null,
      symbols: null,
      yamlOutput: '',
      crdOutput: '',
      mode: 'dsl',
      nlGenerateError: null,
      nlStagedDraft: null,
    })

    const state = get()
    if (state.wasmReady && stagedDraft.dsl.trim()) {
      state.compile()
    }
  },

  discardNaturalLanguageDraft() {
    set({
      nlGenerateError: null,
      nlStagedDraft: null,
    })
  },
}))

// Eagerly start WASM init on store creation (module-level side-effect).
// This overlaps with network fetch of JS/CSS bundles for faster perceived load.
useDSLStore.getState().initWasm()
