/**
 * Zustand store for DSL editor state management.
 *
 * Manages:
 * - DSL source text, YAML/CRD output, diagnostics
 * - WASM lifecycle (init, ready state)
 * - Editor mode switching (DSL / Visual)
 * - Debounced validation on keystroke
 * - Full compile on demand
 * - Decompile router YAML → DSL-owned models, routing, entrypoints, and recipes
 * - Format (canonical pretty-print)
 */

import { create } from 'zustand'
import { wasmBridge } from '@/lib/wasm'
import {
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
import { initialDSLState } from './dslStoreSupport'

// ---------- Debounce helper ----------

let validateTimer: ReturnType<typeof setTimeout> | null = null
const VALIDATE_DEBOUNCE_MS = 300

// ---------- Store ----------

export const useDSLStore = create<DSLStore>((set, get) => ({
  ...initialDSLState,

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
      set({
        yamlOutput: '',
        crdOutput: '',
        diagnostics: [],
        compileError: null,
      })
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
      console.log(
        '[dslStore.compile] Compile result: yaml size=%d, crd size=%d, diagnostics=%d, error=%s',
        result.yaml?.length ?? 0,
        result.crd?.length ?? 0,
        result.diagnostics?.length ?? 0,
        result.error ?? 'none',
      )
      if (result.diagnostics?.length) {
        console.log('[dslStore.compile] Diagnostics:', result.diagnostics)
      }

      // Quick count of decisions in YAML output
      if (result.yaml) {
        const decMatch = result.yaml.match(/^\s*- name:/gm)
        console.log('[dslStore.compile] YAML "- name:" lines count=%d', decMatch?.length ?? 0)
      }

      const compiledYaml = result.yaml || ''
      set({
        yamlOutput: compiledYaml,
        crdOutput: result.crd || '',
        diagnostics: result.diagnostics || [],
        ast: result.ast || null,
        compileError: result.error || null,
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
    set({ ...initialDSLState, wasmReady: get().wasmReady })
  },

  loadDsl(source: string) {
    set({
      dslSource: source,
      dirty: false,
      diagnostics: [],
      compileError: null,
    })
    // Trigger validation after load
    const state = get()
    if (state.wasmReady && source.trim()) {
      state.validate()
    }
  },

  importYaml(yaml: string) {
    const dsl = get().decompile(yaml)
    if (!dsl) throw new Error('Failed to decompile document')
    get().loadDsl(dsl)
  },

  // --- Visual Builder mutations (Phase 2) ---

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
}))

// Eagerly start WASM init on store creation (module-level side-effect).
// This overlaps with network fetch of JS/CSS bundles for faster perceived load.
useDSLStore.getState().initWasm()
