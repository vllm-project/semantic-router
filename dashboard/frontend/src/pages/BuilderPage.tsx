import React, { useEffect, useCallback, useState, useMemo, useRef, useLayoutEffect } from 'react'
import { createPortal } from 'react-dom'
import { DiffEditor } from '@monaco-editor/react'
import { useDSLStore } from '@/stores/dslStore'
import type { EditorMode, Diagnostic, ASTSignalDecl, ASTRouteDecl, ASTPluginDecl, ASTBackendDecl, ASTModelRef, ASTAlgoSpec, ASTPluginRef, BoolExprNode, DeployStep } from '@/types/dsl'
import { getSignalFieldSchema, getAlgorithmFieldSchema, getPluginFieldSchema, ALGORITHM_DESCRIPTIONS, PLUGIN_DESCRIPTIONS, SIGNAL_TYPES, PLUGIN_TYPES, BACKEND_TYPES, ALGORITHM_TYPES, serializeBoolExpr, serializeFields } from '@/lib/dslMutations'
import type { FieldSchema, SignalType, RouteInput, RouteModelInput, RouteAlgoInput, RoutePluginInput } from '@/lib/dslMutations'
import ExpressionBuilder from '@/components/ExpressionBuilder'
import DslGuide from '@/components/DslGuide'
import styles from './BuilderPage.module.css'

// Reuse the DSL Editor as a child component in DSL mode
import DslEditorPage from './DslEditorPage'

// ---------- Types for sidebar selection ----------

type EntityKind = 'signal' | 'route' | 'plugin' | 'backend' | 'global'

interface Selection {
  kind: EntityKind
  name: string
}

// ---------- Sidebar section collapsed state ----------

interface SectionState {
  signals: boolean
  routes: boolean
  plugins: boolean
  backends: boolean
  global: boolean
}

// ---------- Deploy Step Item ----------

const DEPLOY_STEP_ORDER: DeployStep[] = ['validating', 'backing_up', 'writing', 'reloading', 'done']

const DeployStepItem: React.FC<{ step: DeployStep; current: DeployStep | null; label: string }> = ({ step, current, label }) => {
  const currentIdx = current ? DEPLOY_STEP_ORDER.indexOf(current) : -1
  const stepIdx = DEPLOY_STEP_ORDER.indexOf(step)
  const isDone = current === 'done' || (currentIdx > stepIdx && current !== 'error')
  const isActive = current === step
  const isError = current === 'error' && isActive

  return (
    <div className={`${styles.deployStepItem} ${isDone ? styles.deployStepDone : ''} ${isActive ? styles.deployStepActive : ''} ${isError ? styles.deployStepError : ''}`}>
      <span className={styles.deployStepIcon}>
        {isDone ? (
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : isActive && !isError ? (
          <span className={styles.deployStepSpinner} />
        ) : isError ? (
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
          </svg>
        ) : (
          <span className={styles.deployStepPending} />
        )}
      </span>
      <span className={styles.deployStepLabel}>{label}</span>
    </div>
  )
}

// ---------- Component ----------

const BuilderPage: React.FC = () => {
  const {
    dslSource,
    diagnostics,
    symbols,
    ast,
    wasmReady,
    wasmError,
    loading,
    mode,
    dirty,
    yamlOutput,
    crdOutput,
    compileError,
    initWasm,
    compile,
    validate,
    parseAST,
    format,
    reset,
    setMode,
    importYaml,
    loadFromRouter,
    mutateSignal,
    addSignal,
    deleteSignal,
    mutatePlugin,
    addPlugin,
    deletePlugin,
    mutateBackend,
    addBackend,
    deleteBackend,
    deleteRoute,
    mutateRoute,
    addRoute,
    mutateGlobal,
    requestDeploy,
    executeDeploy,
    dismissDeploy,
    deploying,
    deployStep,
    deployResult,
    showDeployConfirm,
    deployPreviewCurrent,
    deployPreviewMerged,
    deployPreviewLoading,
    deployPreviewError,
  } = useDSLStore()

  const [selection, setSelection] = useState<Selection | null>(null)
  const [sections, setSections] = useState<SectionState>({
    signals: true,
    routes: true,
    plugins: true,
    backends: true,
    global: true,
  })
  const [addingEntity, setAddingEntity] = useState<EntityKind | null>(null)
  const [outputPanelOpen, setOutputPanelOpen] = useState(true)
  type OutputTab = 'yaml' | 'crd' | 'dsl'
  const [outputTab, setOutputTab] = useState<OutputTab>('yaml')
  const [copied, setCopied] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [guideOpen, setGuideOpen] = useState(false)

  // --- Resizable Guide Drawer drag logic ---
  const GUIDE_MIN_WIDTH = 300
  const GUIDE_MAX_WIDTH = 800
  const GUIDE_DEFAULT_WIDTH = 420
  const [guideWidth, setGuideWidth] = useState(GUIDE_DEFAULT_WIDTH)
  const [isGuideDragging, setIsGuideDragging] = useState(false)
  const isGuideDraggingRef = useRef(false)
  const guideDragStartXRef = useRef(0)
  const guideDragStartWidthRef = useRef(0)

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const diffEditorRef = useRef<any>(null)
  const [diffChangeCount, setDiffChangeCount] = useState(0)

  const handleGuideDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    isGuideDraggingRef.current = true
    setIsGuideDragging(true)
    guideDragStartXRef.current = e.clientX
    guideDragStartWidthRef.current = guideWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [guideWidth])

  useEffect(() => {
    const handleGuideDragMove = (e: MouseEvent) => {
      if (!isGuideDraggingRef.current) return
      const delta = guideDragStartXRef.current - e.clientX
      const newWidth = Math.min(GUIDE_MAX_WIDTH, Math.max(GUIDE_MIN_WIDTH, guideDragStartWidthRef.current + delta))
      setGuideWidth(newWidth)
    }
    const handleGuideDragEnd = () => {
      if (!isGuideDraggingRef.current) return
      isGuideDraggingRef.current = false
      setIsGuideDragging(false)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.addEventListener('mousemove', handleGuideDragMove)
    document.addEventListener('mouseup', handleGuideDragEnd)
    return () => {
      document.removeEventListener('mousemove', handleGuideDragMove)
      document.removeEventListener('mouseup', handleGuideDragEnd)
    }
  }, [])

  // --- Resizable Output Panel drag logic ---
  const OUTPUT_MIN_WIDTH = 200
  const OUTPUT_MAX_RATIO = 0.6
  const OUTPUT_DEFAULT_WIDTH = 380
  const [outputWidth, setOutputWidth] = useState(OUTPUT_DEFAULT_WIDTH)
  const [isDragging, setIsDragging] = useState(false)
  const isDraggingRef = useRef(false)
  const dragStartXRef = useRef(0)
  const dragStartWidthRef = useRef(0)
  const contentRef = useRef<HTMLDivElement>(null)

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDraggingRef.current = true
    setIsDragging(true)
    dragStartXRef.current = e.clientX
    dragStartWidthRef.current = outputWidth
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [outputWidth])

  useEffect(() => {
    const handleDragMove = (e: MouseEvent) => {
      if (!isDraggingRef.current) return
      const delta = dragStartXRef.current - e.clientX
      const containerWidth = contentRef.current?.offsetWidth ?? window.innerWidth
      const maxWidth = Math.floor(containerWidth * OUTPUT_MAX_RATIO)
      const newWidth = Math.min(maxWidth, Math.max(OUTPUT_MIN_WIDTH, dragStartWidthRef.current + delta))
      setOutputWidth(newWidth)
    }
    const handleDragEnd = () => {
      if (!isDraggingRef.current) return
      isDraggingRef.current = false
      setIsDragging(false)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.addEventListener('mousemove', handleDragMove)
    document.addEventListener('mouseup', handleDragEnd)
    return () => {
      document.removeEventListener('mousemove', handleDragMove)
      document.removeEventListener('mouseup', handleDragEnd)
    }
  }, [])
  const [importText, setImportText] = useState('')
  const [importError, setImportError] = useState<string | null>(null)
  const [importUrl, setImportUrl] = useState('')
  const [importUrlLoading, setImportUrlLoading] = useState(false)
  const importTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Initialize WASM on mount
  useEffect(() => {
    initWasm()
  }, [initWasm])

  // Parse AST when entering visual mode or when dslSource changes in visual mode
  useEffect(() => {
    if (mode === 'visual' && wasmReady && dslSource.trim()) {
      parseAST()
    }
  }, [mode, wasmReady, dslSource, parseAST])

  const toggleSection = useCallback((key: keyof SectionState) => {
    setSections((prev) => ({ ...prev, [key]: !prev[key] }))
  }, [])

  const handleModeSwitch = useCallback(
    (newMode: EditorMode) => {
      setMode(newMode)
      // When switching to visual, parse AST
      if (newMode === 'visual' && wasmReady && dslSource.trim()) {
        parseAST()
      }
    },
    [setMode, wasmReady, dslSource, parseAST],
  )

  // --- Entity CRUD handlers ---

  const handleDeleteEntity = useCallback(
    (kind: EntityKind, name: string, subType?: string) => {
      switch (kind) {
        case 'signal':
          if (subType) deleteSignal(subType, name)
          break
        case 'route':
          deleteRoute(name)
          break
        case 'plugin':
          if (subType) deletePlugin(name, subType)
          break
        case 'backend':
          if (subType) deleteBackend(subType, name)
          break
      }
      setSelection(null)
    },
    [deleteSignal, deleteRoute, deletePlugin, deleteBackend],
  )

  const handleUpdateSignalFields = useCallback(
    (signalType: string, name: string, fields: Record<string, unknown>) => {
      mutateSignal(signalType, name, fields)
    },
    [mutateSignal],
  )

  const handleUpdatePluginFields = useCallback(
    (name: string, pluginType: string, fields: Record<string, unknown>) => {
      mutatePlugin(name, pluginType, fields)
    },
    [mutatePlugin],
  )

  const handleUpdateBackendFields = useCallback(
    (backendType: string, name: string, fields: Record<string, unknown>) => {
      mutateBackend(backendType, name, fields)
    },
    [mutateBackend],
  )

  const handleAddSignal = useCallback(
    (signalType: string, name: string, fields: Record<string, unknown>) => {
      addSignal(signalType, name, fields)
      setSelection({ kind: 'signal', name })
      setAddingEntity(null)
    },
    [addSignal],
  )

  const handleAddPlugin = useCallback(
    (name: string, pluginType: string, fields: Record<string, unknown>) => {
      addPlugin(name, pluginType, fields)
      setSelection({ kind: 'plugin', name })
      setAddingEntity(null)
    },
    [addPlugin],
  )

  const handleAddBackend = useCallback(
    (backendType: string, name: string, fields: Record<string, unknown>) => {
      addBackend(backendType, name, fields)
      setSelection({ kind: 'backend', name })
      setAddingEntity(null)
    },
    [addBackend],
  )

  const handleUpdateRoute = useCallback(
    (name: string, input: RouteInput) => {
      mutateRoute(name, input)
    },
    [mutateRoute],
  )

  const handleUpdateGlobalFields = useCallback(
    (fields: Record<string, unknown>) => {
      mutateGlobal(fields)
    },
    [mutateGlobal],
  )

  const handleAddRoute = useCallback(
    (name: string, input: RouteInput) => {
      addRoute(name, input)
      setSelection({ kind: 'route', name })
      setAddingEntity(null)
    },
    [addRoute],
  )

  // --- Import Config handlers ---

  const handleOpenImport = useCallback(() => {
    setImportText('')
    setImportError(null)
    setImportUrl('')
    setImportUrlLoading(false)
    setShowImportModal(true)
    setTimeout(() => importTextareaRef.current?.focus(), 50)
  }, [])

  const handleImportConfirm = useCallback(() => {
    const yaml = importText.trim()
    if (!yaml) { setImportError('Please paste YAML content'); return }
    try {
      importYaml(yaml)
      setShowImportModal(false)
      setImportText('')
      setImportError(null)
    } catch {
      setImportError('Failed to decompile YAML. Make sure it is valid router config YAML.')
    }
  }, [importText, importYaml])

  const handleImportFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target?.result
      if (typeof text === 'string') { setImportText(text); setImportError(null) }
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [])

  const handleImportUrl = useCallback(async () => {
    const url = importUrl.trim()
    if (!url) { setImportError('Please enter a URL'); return }
    try {
      new URL(url)
    } catch {
      setImportError('Invalid URL format'); return
    }
    setImportUrlLoading(true)
    setImportError(null)
    try {
      const resp = await fetch('/api/tools/fetch-raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      })
      const data = await resp.json()
      if (data.error) {
        throw new Error(data.error)
      }
      if (!data.content?.trim()) {
        throw new Error('Remote returned empty content')
      }
      setImportText(data.content)
      setImportError(null)
    } catch (err) {
      setImportError(`Failed to fetch: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setImportUrlLoading(false)
    }
  }, [importUrl])

  const [loadingFromRouter, setLoadingFromRouter] = useState(false)
  const handleLoadFromRouter = useCallback(async () => {
    setLoadingFromRouter(true)
    setImportError(null)
    try {
      await loadFromRouter()
      setShowImportModal(false)
      setImportText('')
    } catch (err) {
      setImportError(`Failed to load from router: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setLoadingFromRouter(false)
    }
  }, [loadFromRouter])

  // Diagnostic counts
  const errorCount = diagnostics.filter((d) => d.level === 'error').length
  const signalCount = ast?.signals?.length ?? symbols?.signals?.length ?? 0
  const routeCount = ast?.routes?.length ?? symbols?.routes?.length ?? 0
  const pluginCount = ast?.plugins?.length ?? symbols?.plugins?.length ?? 0
  const backendCount = ast?.backends?.length ?? symbols?.backends?.length ?? 0
  const hasGlobal = !!ast?.global
  const isValid = errorCount === 0 && wasmReady
  const lineCount = dslSource.split('\n').length

  // Memoize selected entity from AST
  const selectedEntity = useMemo(() => {
    if (!selection || !ast) return null
    switch (selection.kind) {
      case 'signal':
        return ast.signals?.find((s) => s.name === selection.name) ?? null
      case 'route':
        return ast.routes?.find((r) => r.name === selection.name) ?? null
      case 'plugin':
        return ast.plugins?.find((p) => p.name === selection.name) ?? null
      case 'backend':
        return ast.backends?.find((b) => b.name === selection.name) ?? null
      case 'global':
        return ast.global ?? null
      default:
        return null
    }
  }, [selection, ast])

  // Output panel: get the text content for current tab
  const outputContent = useMemo(() => {
    if (outputTab === 'yaml') return yamlOutput || ''
    if (outputTab === 'crd') return crdOutput || ''
    // DSL tab: show the current DSL source as preview
    return dslSource || ''
  }, [outputTab, yamlOutput, crdOutput, dslSource])

  const handleCopyOutput = useCallback(async () => {
    const text = outputContent
    if (!text) return
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        const ta = document.createElement('textarea')
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'
        document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta)
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      const ta = document.createElement('textarea')
      ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0'
      document.body.appendChild(ta); ta.select(); document.execCommand('copy'); document.body.removeChild(ta)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }, [outputContent])

  return (
    <div className={styles.page}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarTitle}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="2" width="5" height="5" rx="1" />
            <rect x="9" y="2" width="5" height="5" rx="1" />
            <rect x="2" y="9" width="5" height="5" rx="1" />
            <rect x="9" y="9" width="5" height="5" rx="1" />
          </svg>
          Config Builder
          {dirty && <span style={{ color: 'var(--color-text-muted)', fontWeight: 400 }}>(unsaved)</span>}
        </div>

        <span className={styles.divider} />

        {/* Mode Switcher */}
        <div className={styles.modeSwitcher}>
          <button
            className={mode === 'visual' ? styles.modeBtnActive : styles.modeBtn}
            onClick={() => handleModeSwitch('visual')}
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="1" y="1" width="6" height="6" rx="1" />
              <rect x="9" y="1" width="6" height="6" rx="1" />
              <rect x="1" y="9" width="6" height="6" rx="1" />
              <rect x="9" y="9" width="6" height="6" rx="1" />
            </svg>
            Visual
          </button>
          <button
            className={mode === 'dsl' ? styles.modeBtnActive : styles.modeBtn}
            onClick={() => handleModeSwitch('dsl')}
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 3h12M2 8h8M2 13h10" strokeLinecap="round" />
            </svg>
            DSL
          </button>
          <button
            className={styles.modeBtn}
            disabled
            title="Natural Language mode — coming soon"
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 4h12M2 8h9M2 12h6" strokeLinecap="round" />
              <circle cx="13" cy="11" r="2" />
            </svg>
            NL
          </button>
        </div>

        <span className={styles.divider} />

        {/* WASM status */}
        {wasmError ? (
          <span className={styles.statusError}>
            <span className={styles.dot} /> WASM Error
          </span>
        ) : wasmReady ? (
          <span className={styles.statusReady}>
            <span className={styles.dot} /> Ready
          </span>
        ) : (
          <span className={styles.statusLoading}>
            <span className={styles.dotPulse} /> Loading WASM…
          </span>
        )}

        <div className={styles.toolbarRight}>
          <button
            className={styles.toolbarBtn}
            onClick={mode !== 'nl' ? handleOpenImport : undefined}
            disabled={!wasmReady || mode === 'nl'}
            title={mode === 'nl' ? 'Import Config is not available in NL mode (coming soon)' : 'Import router config'}
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 2v8M5 7l3 3 3-3" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M2 11v2a1 1 0 001 1h10a1 1 0 001-1v-2" strokeLinecap="round" />
            </svg>
            Import Config
          </button>
          <button
            className={styles.toolbarBtn}
            onClick={format}
            disabled={!wasmReady || !dslSource.trim()}
            title="Format DSL"
          >
            Format
          </button>
          <button
            className={styles.toolbarBtn}
            onClick={validate}
            disabled={!wasmReady || !dslSource.trim()}
            title="Validate"
          >
            Validate
          </button>
          <button
            className={styles.toolbarBtnPrimary}
            onClick={compile}
            disabled={!wasmReady || !dslSource.trim() || loading}
            title="Compile (Ctrl+Enter)"
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M4 2l8 6-8 6V2z" fill="currentColor" />
            </svg>
            {loading ? 'Compiling…' : 'Compile'}
          </button>
          <button
            className={styles.toolbarBtnDeploy}
            onClick={requestDeploy}
            disabled={!wasmReady || !dslSource.trim() || loading || deploying}
            title="Deploy config to router"
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 2v8M5 7l3 3 3-3" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M2 12v1a1 1 0 001 1h10a1 1 0 001-1v-1" strokeLinecap="round" />
            </svg>
            {deploying ? 'Deploying…' : 'Deploy'}
          </button>
          <span className={styles.divider} />
          <button
            className={guideOpen ? styles.toolbarBtnActive : styles.toolbarBtn}
            onClick={() => setGuideOpen(!guideOpen)}
            title={guideOpen ? 'Close DSL Guide' : 'Open DSL Guide'}
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 2h9a2 2 0 012 2v10l-3-2H2V2z" strokeLinejoin="round" />
              <path d="M5 6h5M5 9h3" strokeLinecap="round" />
            </svg>
            Guide
          </button>
          <button
            className={outputPanelOpen ? styles.toolbarBtnActive : styles.toolbarBtn}
            onClick={() => setOutputPanelOpen(!outputPanelOpen)}
            title={outputPanelOpen ? 'Hide Output Panel' : 'Show Output Panel'}
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="1" y="1" width="14" height="14" rx="2" />
              <path d="M10 1v14" />
            </svg>
            Output
          </button>
          <span className={styles.divider} />
          <button className={styles.toolbarBtnDanger} onClick={reset} title="Reset">
            Reset
          </button>
        </div>
      </div>

      {/* Main Content — editor + output panel */}
      <div className={styles.content} ref={contentRef}>
        {/* Editor area (switches by mode) */}
        <div className={styles.editorArea}>
          {mode === 'visual' && (
            <VisualMode
              ast={ast}
              dslSource={dslSource}
              diagnostics={diagnostics}
              selection={selection}
              onSelect={setSelection}
              sections={sections}
              onToggleSection={toggleSection}
              selectedEntity={selectedEntity}
              signalCount={signalCount}
              routeCount={routeCount}
              pluginCount={pluginCount}
              backendCount={backendCount}
              hasGlobal={hasGlobal}
              wasmReady={wasmReady}
              wasmError={wasmError}
              addingEntity={addingEntity}
              onSetAddingEntity={setAddingEntity}
              onDeleteEntity={handleDeleteEntity}
              onUpdateSignalFields={handleUpdateSignalFields}
              onUpdatePluginFields={handleUpdatePluginFields}
              onUpdateBackendFields={handleUpdateBackendFields}
              onAddSignal={handleAddSignal}
              onAddPlugin={handleAddPlugin}
              onAddBackend={handleAddBackend}
              onUpdateRoute={handleUpdateRoute}
              onUpdateGlobalFields={handleUpdateGlobalFields}
              onAddRoute={handleAddRoute}
              errorCount={errorCount}
              isValid={isValid}
              onModeSwitch={handleModeSwitch}
            />
          )}
          {mode === 'dsl' && (
            <div className={styles.dslModeContainer}>
              <DslEditorPage embedded hideOutput />
            </div>
          )}
          {mode === 'nl' && (
            <div className={styles.nlPlaceholder}>
              <div className={styles.nlPlaceholderIcon}>🤖</div>
              <div className={styles.nlPlaceholderTitle}>Natural Language Mode</div>
              <div>Describe your routing configuration in plain English and let AI generate DSL for you.</div>
              <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
                Coming soon — Phase 6
              </div>
            </div>
          )}
        </div>

        {/* Global Output Panel — always visible, resizable */}
        {outputPanelOpen && (
          <>
          <div
            className={styles.resizeHandle}
            onMouseDown={handleDragStart}
          >
            <div className={styles.resizeHandleLine} />
          </div>
          <div className={styles.outputPanel} style={{ width: outputWidth }}>
            <div className={styles.outputPanelTabs}>
              <button
                className={outputTab === 'yaml' ? styles.outputPanelTabActive : styles.outputPanelTab}
                onClick={() => setOutputTab('yaml')}
              >
                YAML
              </button>
              <button
                className={outputTab === 'crd' ? styles.outputPanelTabActive : styles.outputPanelTab}
                onClick={() => setOutputTab('crd')}
              >
                CRD
              </button>
              <button
                className={outputTab === 'dsl' ? styles.outputPanelTabActive : styles.outputPanelTab}
                onClick={() => setOutputTab('dsl')}
              >
                DSL
              </button>
              <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 'var(--spacing-xs)' }}>
                {outputContent && (
                  <button className={styles.outputPanelCopyBtn} onClick={handleCopyOutput} title="Copy to clipboard">
                    {copied ? (
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="var(--color-success)" strokeWidth="2">
                        <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                    ) : (
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <rect x="5" y="5" width="9" height="9" rx="1" />
                        <path d="M2 11V2h9" strokeLinecap="round" />
                      </svg>
                    )}
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                )}
                <button
                  className={styles.outputPanelCloseBtn}
                  onClick={() => setOutputPanelOpen(false)}
                  title="Close panel"
                >
                  <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
            </div>
            <div className={styles.outputPanelContent}>
              {compileError && (
                <div className={styles.outputPanelError}>{compileError}</div>
              )}
              {outputContent ? (
                <pre className={styles.outputPanelCode}>{outputContent}</pre>
              ) : (
                <div className={styles.emptyState}>
                  <div className={styles.emptyIcon}>&#9889;</div>
                  <div>{outputTab === 'dsl' ? 'DSL source is empty' : <>Press <strong>Compile</strong> to generate {outputTab.toUpperCase()} output</>}</div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
                    Ctrl+Enter to compile
                  </div>
                </div>
              )}
            </div>
          </div>
          </>
        )}
        {!outputPanelOpen && (
          <button
            className={styles.outputPanelToggle}
            onClick={() => setOutputPanelOpen(true)}
            title="Show Output Panel"
          >
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M10 2l-4 6 4 6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>

      {/* Status Bar */}
      <div className={styles.statusBar}>
        <div className={`${styles.statusItem} ${isValid ? styles.statusValid : styles.statusInvalid}`}>
          {isValid ? (
            <svg className={styles.statusCheckmark} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg className={styles.statusCheckmark} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
          {isValid ? 'Valid' : `${errorCount} error${errorCount !== 1 ? 's' : ''}`}
        </div>
        <div className={styles.statusItem}>Signals: {signalCount}</div>
        <div className={styles.statusItem}>Routes: {routeCount}</div>
        <div className={styles.statusItem}>Plugins: {pluginCount}</div>
        <div className={styles.statusItem}>Backends: {backendCount}</div>
        {mode === 'dsl' && <div className={styles.statusItem}>Lines: {lineCount}</div>}
        <div className={styles.statusItem}>
          Mode: {mode === 'visual' ? 'Visual' : mode === 'dsl' ? 'DSL' : 'NL'}
        </div>
      </div>

      {/* Hidden file input for YAML import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml,.json"
        style={{ display: 'none' }}
        onChange={handleImportFile}
      />

      {/* Import Config Modal */}
      {showImportModal && (
        <div className={styles.modalOverlay} onClick={() => setShowImportModal(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3 className={styles.modalTitle}>Import Config</h3>
              <button className={styles.modalClose} onClick={() => setShowImportModal(false)}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <div className={styles.modalBody}>
              <p className={styles.modalHint}>
                Paste a router config YAML below, load from a file, fetch from a URL, or load the current router config directly. It will be decompiled into DSL.
              </p>
              <div className={styles.importUrlRow}>
                <input
                  className={styles.importUrlInput}
                  type="url"
                  value={importUrl}
                  onChange={(e) => { setImportUrl(e.target.value); setImportError(null) }}
                  placeholder="https://example.com/config.yaml"
                  onKeyDown={(e) => { if (e.key === 'Enter') handleImportUrl() }}
                />
                <button
                  className={styles.toolbarBtn}
                  onClick={handleImportUrl}
                  disabled={importUrlLoading || !importUrl.trim()}
                >
                  {importUrlLoading ? (
                    <>
                      <span className={styles.dotPulse} />
                      Fetching…
                    </>
                  ) : (
                    <>
                      <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M6 2a4 4 0 100 8 4 4 0 000-8z" />
                        <path d="M2 6h8M6 2v8" strokeLinecap="round" />
                        <path d="M14 14l-3.5-3.5" strokeLinecap="round" />
                      </svg>
                      Fetch
                    </>
                  )}
                </button>
              </div>
              <textarea
                ref={importTextareaRef}
                className={styles.importTextarea}
                value={importText}
                onChange={(e) => { setImportText(e.target.value); setImportError(null) }}
                placeholder="Paste YAML config here..."
                spellCheck={false}
              />
              {importError && <div className={styles.importError}>{importError}</div>}
            </div>
            <div className={styles.modalFooter}>
              <button className={styles.toolbarBtn} onClick={() => fileInputRef.current?.click()}>
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 14h12M8 2v9M5 5l3-3 3 3" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Load File
              </button>
              <button
                className={styles.toolbarBtnPrimary}
                onClick={handleLoadFromRouter}
                disabled={loadingFromRouter}
                title="Load the current running router config and decompile to DSL"
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <rect x="2" y="2" width="12" height="12" rx="2" />
                  <path d="M8 5v6M5 8l3 3 3-3" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {loadingFromRouter ? 'Loading…' : 'Load from Router'}
              </button>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 'var(--spacing-sm)' }}>
                <button className={styles.toolbarBtn} onClick={() => setShowImportModal(false)}>Cancel</button>
                <button className={styles.toolbarBtnPrimary} onClick={handleImportConfirm} disabled={!importText.trim()}>Import</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DSL Guide Drawer */}
      {guideOpen && createPortal(
        <div className={styles.guideDrawerOverlay} onClick={() => { if (!isGuideDraggingRef.current) setGuideOpen(false) }}>
          <div className={styles.guideDrawer} style={{ width: guideWidth }} onClick={(e) => e.stopPropagation()}>
            <div
              className={styles.guideDrawerResizeHandle}
              onMouseDown={handleGuideDragStart}
            >
              <div className={styles.guideDrawerResizeLine} />
            </div>
            <div className={styles.guideDrawerHeader}>
              <span className={styles.guideDrawerTitle}>
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 2h9a2 2 0 012 2v10l-3-2H2V2z" strokeLinejoin="round" />
                  <path d="M5 6h5M5 9h3" strokeLinecap="round" />
                </svg>
                DSL Language Guide
              </span>
              <button className={styles.guideDrawerClose} onClick={() => setGuideOpen(false)} title="Close Guide">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <div className={styles.guideDrawerBody}>
              <DslGuide onInsertSnippet={(snippet) => {
                if (mode !== 'dsl') setMode('dsl')
                const store = useDSLStore.getState()
                const src = store.dslSource
                store.setDslSource(src ? src.trimEnd() + '\n\n' + snippet + '\n' : snippet + '\n')
                setGuideOpen(false)
              }} />
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Deploy Confirmation Modal with Diff Preview */}
      {showDeployConfirm && createPortal(
        <div className={styles.modalOverlay} onClick={dismissDeploy}>
          <div className={styles.deployDiffModal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.deployModalHeader}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--color-warning)" strokeWidth="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              <span>Deploy to Router — Config Diff</span>
              <div style={{ flex: 1 }} />
              <span className={styles.deployDiffLabels}>
                <span className={styles.deployDiffLabelOld}>Current</span>
                <span style={{ margin: '0 0.25rem', color: 'var(--color-text-muted)' }}>&rarr;</span>
                <span className={styles.deployDiffLabelNew}>After Deploy</span>
              </span>
            </div>
            {/* Diff navigation bar — prev/next change buttons */}
            {!deployPreviewLoading && !deployPreviewError && (
              <div className={styles.deployDiffNav}>
                <button
                  className={styles.deployDiffNavBtn}
                  title="Previous Change (↑)"
                  onClick={() => {
                    const editor = diffEditorRef.current
                    if (!editor) return
                    const nav = editor.getLineChanges()
                    if (!nav || nav.length === 0) return
                    const modifiedEditor = editor.getModifiedEditor()
                    const currentLine = modifiedEditor.getPosition()?.lineNumber ?? 1
                    // find previous change
                    for (let i = nav.length - 1; i >= 0; i--) {
                      const startLine = nav[i].modifiedStartLineNumber || nav[i].originalStartLineNumber
                      if (startLine < currentLine) {
                        modifiedEditor.revealLineInCenter(startLine)
                        modifiedEditor.setPosition({ lineNumber: startLine, column: 1 })
                        return
                      }
                    }
                    // wrap around to last change
                    const last = nav[nav.length - 1]
                    const lastLine = last.modifiedStartLineNumber || last.originalStartLineNumber
                    modifiedEditor.revealLineInCenter(lastLine)
                    modifiedEditor.setPosition({ lineNumber: lastLine, column: 1 })
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 10l4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
                <button
                  className={styles.deployDiffNavBtn}
                  title="Next Change (↓)"
                  onClick={() => {
                    const editor = diffEditorRef.current
                    if (!editor) return
                    const nav = editor.getLineChanges()
                    if (!nav || nav.length === 0) return
                    const modifiedEditor = editor.getModifiedEditor()
                    const currentLine = modifiedEditor.getPosition()?.lineNumber ?? 1
                    // find next change
                    for (let i = 0; i < nav.length; i++) {
                      const startLine = nav[i].modifiedStartLineNumber || nav[i].originalStartLineNumber
                      if (startLine > currentLine) {
                        modifiedEditor.revealLineInCenter(startLine)
                        modifiedEditor.setPosition({ lineNumber: startLine, column: 1 })
                        return
                      }
                    }
                    // wrap around to first change
                    const first = nav[0]
                    const firstLine = first.modifiedStartLineNumber || first.originalStartLineNumber
                    modifiedEditor.revealLineInCenter(firstLine)
                    modifiedEditor.setPosition({ lineNumber: firstLine, column: 1 })
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
                <span className={styles.deployDiffNavInfo}>
                  {diffChangeCount === 0 ? 'No changes' : `${diffChangeCount} change${diffChangeCount > 1 ? 's' : ''}`}
                </span>
              </div>
            )}
            <div className={styles.deployDiffBody}>
              {deployPreviewLoading && (
                <div className={styles.deployDiffLoading}>
                  <div className={styles.spinner} />
                  Loading config diff...
                </div>
              )}
              {deployPreviewError && (
                <div className={styles.deployDiffError}>
                  Failed to load preview: {deployPreviewError}
                </div>
              )}
              {!deployPreviewLoading && !deployPreviewError && (
                <DiffEditor
                  original={deployPreviewCurrent}
                  modified={deployPreviewMerged}
                  language="yaml"
                  theme="vs-dark"
                  onMount={(editor) => {
                    diffEditorRef.current = editor
                    // Update change count once diff is computed
                    const updateCount = () => {
                      const changes = editor.getLineChanges()
                      setDiffChangeCount(changes?.length ?? 0)
                    }
                    // Monaco diff is async; poll briefly then use onDidUpdateDiff
                    const timer = setTimeout(updateCount, 500)
                    try { editor.onDidUpdateDiff(updateCount) } catch { /* older Monaco */ }
                    return () => clearTimeout(timer)
                  }}
                  options={{
                    readOnly: true,
                    renderSideBySide: true,
                    minimap: { enabled: true },
                    scrollBeyondLastLine: false,
                    fontSize: 12,
                    lineNumbers: 'on',
                    wordWrap: 'on',
                    renderOverviewRuler: true,
                    renderIndicators: true,
                    contextmenu: false,
                    scrollbar: { verticalScrollbarSize: 8, horizontalScrollbarSize: 8 },
                  }}
                />
              )}
            </div>
            <div className={styles.deployModalFooter}>
              <p style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', margin: 0, flex: 1 }}>
                A backup of the current config will be created before deployment.
              </p>
              <button className={styles.toolbarBtn} onClick={dismissDeploy}>Cancel</button>
              <button
                className={styles.toolbarBtnDeploy}
                onClick={executeDeploy}
                disabled={deployPreviewLoading || !!deployPreviewError}
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M8 2v8M5 7l3 3 3-3" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M2 12v1a1 1 0 001 1h10a1 1 0 001-1v-1" strokeLinecap="round" />
                </svg>
                Deploy Now
              </button>
            </div>
          </div>
        </div>,
        document.body
      )}

      {/* Deploy Progress / Result Toast */}
      {(deploying || deployResult) && createPortal(
        <div className={styles.deployToast}>
          {deploying && (
            <div className={styles.deployProgress}>
              <div className={styles.deployStepList}>
                <DeployStepItem step="validating" current={deployStep} label="Validating config" />
                <DeployStepItem step="backing_up" current={deployStep} label="Creating backup" />
                <DeployStepItem step="writing" current={deployStep} label="Writing config" />
                <DeployStepItem step="reloading" current={deployStep} label="Router reloading" />
              </div>
            </div>
          )}
          {deployResult && !deploying && (
            <div className={deployResult.status === 'success' ? styles.deployResultSuccess : styles.deployResultError}>
              <div className={styles.deployResultIcon}>
                {deployResult.status === 'success' ? (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
                  </svg>
                )}
              </div>
              <span className={styles.deployResultMsg}>{deployResult.message}</span>
              <button className={styles.deployResultDismiss} onClick={dismissDeploy}>
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          )}
        </div>,
        document.body
      )}

      {/* Drag overlay — prevents Monaco/iframes from capturing mouse during resize */}
      {(isDragging || isGuideDragging) && createPortal(
        <div className={styles.dragOverlay} />,
        document.body
      )}
    </div>
  )
}

// ===================================================================
// Visual Mode — Sidebar + Detail Panel
// ===================================================================

interface VisualModeProps {
  ast: ReturnType<typeof useDSLStore.getState>['ast']
  dslSource: string
  diagnostics: Diagnostic[]
  selection: Selection | null
  onSelect: (sel: Selection | null) => void
  sections: SectionState
  onToggleSection: (key: keyof SectionState) => void
  selectedEntity: ASTSignalDecl | ASTRouteDecl | ASTPluginDecl | ASTBackendDecl | { fields: Record<string, unknown> } | null
  signalCount: number
  routeCount: number
  pluginCount: number
  backendCount: number
  hasGlobal: boolean
  wasmReady: boolean
  wasmError: string | null
  addingEntity: EntityKind | null
  onSetAddingEntity: (kind: EntityKind | null) => void
  onDeleteEntity: (kind: EntityKind, name: string, subType?: string) => void
  onUpdateSignalFields: (signalType: string, name: string, fields: Record<string, unknown>) => void
  onUpdatePluginFields: (name: string, pluginType: string, fields: Record<string, unknown>) => void
  onUpdateBackendFields: (backendType: string, name: string, fields: Record<string, unknown>) => void
  onAddSignal: (signalType: string, name: string, fields: Record<string, unknown>) => void
  onAddPlugin: (name: string, pluginType: string, fields: Record<string, unknown>) => void
  onAddBackend: (backendType: string, name: string, fields: Record<string, unknown>) => void
  onUpdateRoute: (name: string, input: RouteInput) => void
  onUpdateGlobalFields: (fields: Record<string, unknown>) => void
  onAddRoute: (name: string, input: RouteInput) => void
  errorCount: number
  isValid: boolean
  onModeSwitch: (mode: EditorMode) => void
}

const VisualMode: React.FC<VisualModeProps> = ({
  ast,
  diagnostics,
  selection,
  onSelect,
  sections,
  onToggleSection,
  selectedEntity,
  signalCount,
  routeCount,
  pluginCount,
  backendCount,
  hasGlobal,
  wasmReady,
  wasmError,
  addingEntity,
  onSetAddingEntity,
  onDeleteEntity,
  onUpdateSignalFields,
  onUpdatePluginFields,
  onUpdateBackendFields,
  onAddSignal,
  onAddPlugin,
  onAddBackend,
  onUpdateRoute,
  onUpdateGlobalFields,
  onAddRoute,
  errorCount,
  isValid,
  onModeSwitch,
}) => {
  // Collect available signal names for expression builder
  // Complexity signals are referenced as "<name>:easy", "<name>:medium", "<name>:hard" in route conditions
  const availableSignals = useMemo(() => {
    const result: { signalType: string; name: string }[] = []
    for (const s of ast?.signals ?? []) {
      if (s.signalType === 'complexity') {
        result.push({ signalType: s.signalType, name: `${s.name}:easy` })
        result.push({ signalType: s.signalType, name: `${s.name}:medium` })
        result.push({ signalType: s.signalType, name: `${s.name}:hard` })
      } else {
        result.push({ signalType: s.signalType, name: s.name })
      }
    }
    return result
  }, [ast?.signals])
  // Collect available plugin names for toggle panel
  const availablePlugins = useMemo(() => ast?.plugins?.map(p => ({ name: p.name, pluginType: p.pluginType })) ?? [], [ast?.plugins])
  // Collect available model names from all routes for selection
  const availableModels = useMemo(() => {
    const modelSet = new Set<string>()
    ast?.routes?.forEach(r => r.models.forEach(m => { if (m.model) modelSet.add(m.model) }))
    return Array.from(modelSet).sort()
  }, [ast?.routes])

  // Validation panel state
  const [validationOpen, setValidationOpen] = useState(true)
  const errorDiags = useMemo(() => diagnostics.filter(d => d.level === 'error'), [diagnostics])
  const warnDiags = useMemo(() => diagnostics.filter(d => d.level === 'warning'), [diagnostics])
  const constraintDiags = useMemo(() => diagnostics.filter(d => d.level === 'constraint'), [diagnostics])

  const handleApplyFix = useCallback((diag: Diagnostic, newText: string) => {
    const store = useDSLStore.getState()
    const src = store.dslSource
    const lines = src.split('\n')
    if (diag.line < 1 || diag.line > lines.length) return

    const lineContent = lines[diag.line - 1]
    let startCol = diag.column
    let endCol = diag.column
    while (startCol > 1 && /[\w\-.]/.test(lineContent[startCol - 2])) startCol--
    while (endCol <= lineContent.length && /[\w\-.]/.test(lineContent[endCol - 1])) endCol++

    const before = lineContent.slice(0, startCol - 1)
    const after = lineContent.slice(endCol - 1)
    lines[diag.line - 1] = before + newText + after

    const newSrc = lines.join('\n')
    useDSLStore.getState().setDslSource(newSrc)
    // Re-parse AST for visual mode
    if (useDSLStore.getState().wasmReady) useDSLStore.getState().parseAST()
  }, [])

  return (
    <div className={styles.visualContainer}>
      <div className={styles.visualRow}>
      {/* Sidebar */}
      <div className={styles.sidebar}>
        {/* Dashboard home link */}
        <div
          className={selection === null && !addingEntity ? styles.sidebarHomeActive : styles.sidebarHome}
          onClick={() => { onSetAddingEntity(null); onSelect(null) }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
          Dashboard
        </div>

        {/* Signals */}
        <SidebarSection
          title="Signals"
          count={signalCount}
          open={sections.signals}
          onToggle={() => onToggleSection('signals')}
          onAdd={() => { onSetAddingEntity('signal'); onSelect(null) }}
        >
          {ast?.signals?.map((s) => (
            <li
              key={s.name}
              className={
                selection?.kind === 'signal' && selection.name === s.name
                  ? styles.sidebarItemActive
                  : styles.sidebarItem
              }
              onClick={() => { onSetAddingEntity(null); onSelect({ kind: 'signal', name: s.name }) }}
            >
              <SignalIcon className={styles.sidebarItemIcon} />
              <span className={styles.sidebarItemName}>{s.name}</span>
              <span className={styles.sidebarItemType}>{s.signalType}</span>
            </li>
          ))}
        </SidebarSection>

        {/* Routes */}
        <SidebarSection
          title="Routes"
          count={routeCount}
          open={sections.routes}
          onToggle={() => onToggleSection('routes')}
          onAdd={() => { onSetAddingEntity('route'); onSelect(null) }}
        >
          {ast?.routes?.map((r) => (
            <li
              key={r.name}
              className={
                selection?.kind === 'route' && selection.name === r.name
                  ? styles.sidebarItemActive
                  : styles.sidebarItem
              }
              onClick={() => { onSetAddingEntity(null); onSelect({ kind: 'route', name: r.name }) }}
            >
              <RouteIcon className={styles.sidebarItemIcon} />
              <span className={styles.sidebarItemName}>{r.name}</span>
              <span className={styles.sidebarItemType}>P{r.priority}</span>
            </li>
          ))}
        </SidebarSection>

        {/* Plugins */}
        <SidebarSection
          title="Plugins"
          count={pluginCount}
          open={sections.plugins}
          onToggle={() => onToggleSection('plugins')}
          onAdd={() => { onSetAddingEntity('plugin'); onSelect(null) }}
        >
          {ast?.plugins?.map((p) => (
            <li
              key={p.name}
              className={
                selection?.kind === 'plugin' && selection.name === p.name
                  ? styles.sidebarItemActive
                  : styles.sidebarItem
              }
              onClick={() => { onSetAddingEntity(null); onSelect({ kind: 'plugin', name: p.name }) }}
            >
              <PluginIcon className={styles.sidebarItemIcon} />
              <span className={styles.sidebarItemName}>{p.name}</span>
              <span className={styles.sidebarItemType}>{p.pluginType}</span>
            </li>
          ))}
        </SidebarSection>

        {/* Backends */}
        <SidebarSection
          title="Backends"
          count={backendCount}
          open={sections.backends}
          onToggle={() => onToggleSection('backends')}
          onAdd={() => { onSetAddingEntity('backend'); onSelect(null) }}
        >
          {ast?.backends?.map((b) => (
            <li
              key={b.name}
              className={
                selection?.kind === 'backend' && selection.name === b.name
                  ? styles.sidebarItemActive
                  : styles.sidebarItem
              }
              onClick={() => { onSetAddingEntity(null); onSelect({ kind: 'backend', name: b.name }) }}
            >
              <BackendIcon className={styles.sidebarItemIcon} />
              <span className={styles.sidebarItemName}>{b.name}</span>
              <span className={styles.sidebarItemType}>{b.backendType}</span>
            </li>
          ))}
        </SidebarSection>

        {/* Global */}
        <SidebarSection
          title="Global"
          count={hasGlobal ? 1 : 0}
          open={sections.global}
          onToggle={() => onToggleSection('global')}
        >
          {hasGlobal && (
            <li
              className={
                selection?.kind === 'global' ? styles.sidebarItemActive : styles.sidebarItem
              }
              onClick={() => { onSetAddingEntity(null); onSelect({ kind: 'global', name: 'global' }) }}
            >
              <GlobalIcon className={styles.sidebarItemIcon} />
              <span className={styles.sidebarItemName}>Global Settings</span>
            </li>
          )}
        </SidebarSection>
      </div>

      {/* Main panel */}
      <div className={styles.mainPanel}>
        {!wasmReady && !wasmError && (
          <div className={styles.wasmOverlay}>
            <div className={styles.spinner} />
            Loading Signal Compiler…
          </div>
        )}

        <div className={styles.mainPanelContent}>
          {addingEntity === 'signal' ? (
            <AddSignalForm onAdd={onAddSignal} onCancel={() => onSetAddingEntity(null)} />
          ) : addingEntity === 'plugin' ? (
            <AddPluginForm onAdd={onAddPlugin} onCancel={() => onSetAddingEntity(null)} />
          ) : addingEntity === 'backend' ? (
            <AddBackendForm onAdd={onAddBackend} onCancel={() => onSetAddingEntity(null)} />
          ) : addingEntity === 'route' ? (
            <AddRouteForm onAdd={onAddRoute} onCancel={() => onSetAddingEntity(null)} availableSignals={availableSignals} availablePlugins={availablePlugins} availableModels={availableModels} />
          ) : !selection ? (
            <DashboardView
              ast={ast}
              signalCount={signalCount}
              routeCount={routeCount}
              pluginCount={pluginCount}
              backendCount={backendCount}
              hasGlobal={hasGlobal}
              isValid={isValid}
              errorCount={errorCount}
              onSelect={onSelect}
              onAddEntity={onSetAddingEntity}
              onModeSwitch={onModeSwitch}
            />
          ) : selection.name === '__list__' ? (
            <EntityListView
              kind={selection.kind}
              ast={ast}
              onSelect={onSelect}
              onBack={() => onSelect(null)}
              onAddEntity={onSetAddingEntity}
            />
          ) : (
            <EntityDetailView
              selection={selection}
              entity={selectedEntity}
              ast={ast}
              onDeleteEntity={onDeleteEntity}
              onUpdateSignalFields={onUpdateSignalFields}
              onUpdatePluginFields={onUpdatePluginFields}
              onUpdateBackendFields={onUpdateBackendFields}
              onUpdateRoute={onUpdateRoute}
              onUpdateGlobalFields={onUpdateGlobalFields}
              availableSignals={availableSignals}
              availablePlugins={availablePlugins}
              availableModels={availableModels}
              onBack={() => onSelect(null)}
            />
          )}
        </div>
      </div>
      </div>{/* end visualRow */}

      {/* Validation Panel */}
      {diagnostics.length > 0 && (
        <div className={styles.validationPanel}>
          <div className={styles.validationHeader} onClick={() => setValidationOpen(!validationOpen)}>
            <span className={styles.validationTitle}>
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Validation
            </span>
            <span className={styles.validationCounts}>
              {errorDiags.length > 0 && <span className={styles.valCountError}>{errorDiags.length} error{errorDiags.length !== 1 ? 's' : ''}</span>}
              {warnDiags.length > 0 && <span className={styles.valCountWarn}>{warnDiags.length} warning{warnDiags.length !== 1 ? 's' : ''}</span>}
              {constraintDiags.length > 0 && <span className={styles.valCountConstraint}>{constraintDiags.length} constraint{constraintDiags.length !== 1 ? 's' : ''}</span>}
            </span>
            <svg
              className={`${styles.validationChevron} ${validationOpen ? styles.validationChevronOpen : ''}`}
              viewBox="0 0 16 16"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          {validationOpen && (
            <div className={styles.validationBody}>
              {errorDiags.length > 0 && (
                <div className={styles.valGroup}>
                  <div className={styles.valGroupTitle}>
                    <svg className={styles.valIconError} viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="8" r="7" /><path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" /></svg>
                    Error ({errorDiags.length})
                  </div>
                  {errorDiags.map((d, i) => (
                    <div key={i} className={styles.valItem}>
                      <span className={styles.valMessage}>
                        Ln {d.line}, Col {d.column}: {d.message}
                      </span>
                      {d.fixes?.map((fix, fi) => (
                        <button key={fi} className={styles.valFixBtn} onClick={() => handleApplyFix(d, fix.newText)} title={fix.description}>
                          Fix
                        </button>
                      ))}
                    </div>
                  ))}
                </div>
              )}
              {warnDiags.length > 0 && (
                <div className={styles.valGroup}>
                  <div className={styles.valGroupTitle}>
                    <svg className={styles.valIconWarn} viewBox="0 0 16 16" fill="currentColor"><path d="M8 1l7 13H1L8 1z" /><path d="M8 6v3M8 11v1" stroke="#000" strokeWidth="1.5" strokeLinecap="round" /></svg>
                    Warning ({warnDiags.length})
                  </div>
                  {warnDiags.map((d, i) => (
                    <div key={i} className={styles.valItem}>
                      <span className={styles.valMessage}>
                        Ln {d.line}, Col {d.column}: {d.message}
                      </span>
                      {d.fixes?.map((fix, fi) => (
                        <button key={fi} className={styles.valFixBtn} onClick={() => handleApplyFix(d, fix.newText)} title={fix.description}>
                          Fix
                        </button>
                      ))}
                    </div>
                  ))}
                </div>
              )}
              {constraintDiags.length > 0 && (
                <div className={styles.valGroup}>
                  <div className={styles.valGroupTitle}>
                    <svg className={styles.valIconConstraint} viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="8" r="7" /><path d="M8 5v4M8 11v1" stroke="#000" strokeWidth="1.5" strokeLinecap="round" /></svg>
                    Constraint ({constraintDiags.length})
                  </div>
                  {constraintDiags.map((d, i) => (
                    <div key={i} className={styles.valItem}>
                      <span className={styles.valMessage}>
                        Ln {d.line}, Col {d.column}: {d.message}
                      </span>
                      {d.fixes?.map((fix, fi) => (
                        <button key={fi} className={styles.valFixBtn} onClick={() => handleApplyFix(d, fix.newText)} title={fix.description}>
                          Fix
                        </button>
                      ))}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ===================================================================
// Sidebar Section (collapsible)
// ===================================================================

interface SidebarSectionProps {
  title: string
  count: number
  open: boolean
  onToggle: () => void
  onAdd?: () => void
  children: React.ReactNode
}

const SidebarSection: React.FC<SidebarSectionProps> = ({ title, count, open, onToggle, onAdd, children }) => (
  <div className={styles.sidebarSection}>
    <div className={styles.sidebarSectionHeader} onClick={onToggle}>
      <span className={styles.sidebarSectionTitle}>
        {title}
        <span className={styles.sidebarCount}>{count}</span>
      </span>
      <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
        {onAdd && (
          <button
            className={styles.sidebarAddBtn}
            onClick={(e) => { e.stopPropagation(); onAdd() }}
            title={`Add ${title.slice(0, -1)}`}
            style={{ width: 'auto', padding: '0.125rem 0.25rem' }}
          >
            +
          </button>
        )}
        <svg
          className={`${styles.sidebarSectionChevron} ${open ? styles.sidebarSectionChevronOpen : ''}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M6 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    </div>
    {open && <ul className={styles.sidebarList}>{children}</ul>}
  </div>
)

// ===================================================================
// Dashboard View (no entity selected)
// ===================================================================

interface DashboardViewProps {
  ast: ReturnType<typeof useDSLStore.getState>['ast']
  signalCount: number
  routeCount: number
  pluginCount: number
  backendCount: number
  hasGlobal: boolean
  isValid: boolean
  errorCount: number
  onSelect: (sel: Selection) => void
  onAddEntity: (kind: EntityKind) => void
  onModeSwitch: (mode: EditorMode) => void
}

/** Serialize a BoolExprNode into a short readable string */
function boolExprToText(node: BoolExprNode | null, maxLen = 60): string {
  if (!node) return '(always)'
  const serialize = (n: BoolExprNode): string => {
    switch (n.type) {
      case 'signal_ref': return `${n.signalType}("${n.signalName}")`
      case 'not': return `NOT ${serialize(n.expr)}`
      case 'and': {
        const l = serialize(n.left), r = serialize(n.right)
        return `${l} AND ${r}`
      }
      case 'or': {
        const l = serialize(n.left), r = serialize(n.right)
        return `(${l} OR ${r})`
      }
    }
  }
  const text = serialize(node)
  return text.length > maxLen ? text.slice(0, maxLen - 3) + '...' : text
}

const DashboardView: React.FC<DashboardViewProps> = ({
  ast,
  signalCount,
  routeCount,
  pluginCount,
  backendCount,
  hasGlobal,
  isValid,
  errorCount,
  onSelect,
  onAddEntity,
  onModeSwitch,
}) => {
  const routes = ast?.routes ?? []
  const defaultRoute = routes.find(r => !r.when)
  const conditionalRoutes = routes.filter(r => !!r.when).sort((a, b) => b.priority - a.priority)

  return (
    <div className={styles.dashboard}>
      {/* Title */}
      <div className={styles.dashboardHeader}>
        <div className={styles.dashboardTitle}>
          <svg width="20" height="20" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="2" y="2" width="5" height="5" rx="1" />
            <rect x="9" y="2" width="5" height="5" rx="1" />
            <rect x="2" y="9" width="5" height="5" rx="1" />
            <rect x="9" y="9" width="5" height="5" rx="1" />
          </svg>
          Semantic Router Config Builder
        </div>
        <div className={`${styles.dashboardBadge} ${isValid ? styles.dashboardBadgeOk : styles.dashboardBadgeErr}`}>
          {isValid ? '✓ Valid' : `${errorCount} error${errorCount !== 1 ? 's' : ''}`}
        </div>
      </div>

      {/* Status Cards */}
      <div className={styles.statsGrid}>
        {[
          { label: 'Signals', count: signalCount, kind: 'signal' as const, icon: <SignalIcon className={styles.statIcon} /> },
          { label: 'Routes', count: routeCount, kind: 'route' as const, icon: <RouteIcon className={styles.statIcon} /> },
          { label: 'Plugins', count: pluginCount, kind: 'plugin' as const, icon: <PluginIcon className={styles.statIcon} /> },
          { label: 'Backends', count: backendCount, kind: 'backend' as const, icon: <BackendIcon className={styles.statIcon} /> },
          { label: 'Global', count: hasGlobal ? 1 : 0, kind: 'global' as const, icon: <GlobalIcon className={styles.statIcon} /> },
        ].map(card => (
          <div key={card.label} className={styles.statCard} onClick={() => card.count > 0 && onSelect({ kind: card.kind, name: card.kind === 'global' ? 'global' : '__list__' })}>
            {card.icon}
            <span className={styles.statValue}>{card.count}</span>
            <span className={styles.statLabel}>{card.label}</span>
            <span className={`${styles.statBadge} ${card.count > 0 ? styles.statBadgeOk : styles.statBadgeEmpty}`}>
              {card.count > 0 ? '✓ valid' : 'empty'}
            </span>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className={styles.dashSection}>
        <div className={styles.dashSectionTitle}>Quick Actions</div>
        <div className={styles.quickActions}>
          <button className={styles.quickActionBtn} onClick={() => onAddEntity('signal')}>
            <span className={styles.quickActionIcon}>+</span> New Signal
          </button>
          <button className={styles.quickActionBtn} onClick={() => onAddEntity('route')}>
            <span className={styles.quickActionIcon}>+</span> New Route
          </button>
          <button className={styles.quickActionBtn} onClick={() => onAddEntity('backend')}>
            <span className={styles.quickActionIcon}>+</span> New Backend
          </button>
          <button className={styles.quickActionBtn} onClick={() => onAddEntity('plugin')}>
            <span className={styles.quickActionIcon}>+</span> New Plugin
          </button>
        </div>
      </div>

      {/* Route Map (visual flow) */}
      {routes.length > 0 && (
        <div className={styles.dashSection}>
          <div className={styles.dashSectionTitle}>Route Map</div>
          <div className={styles.routeMap}>
            <div className={styles.routeMapEntry}>
              <span className={styles.routeMapEntryLabel}>User Query</span>
            </div>
            <div className={styles.routeMapFlow}>
              {conditionalRoutes.map(route => (
                <div key={route.name} className={styles.routeMapBranch} onClick={() => onSelect({ kind: 'route', name: route.name })}>
                  <div className={styles.routeMapCondition}>
                    <span className={styles.routeMapCondIcon}>├─</span>
                    <code className={styles.routeMapCondText}>{boolExprToText(route.when)}</code>
                  </div>
                  <div className={styles.routeMapTarget}>
                    <span className={styles.routeMapTargetArrow}>└→</span>
                    <span className={styles.routeMapRouteName}>&quot;{route.name}&quot;</span>
                    <span className={styles.routeMapTargetArrow}>→</span>
                    <span className={styles.routeMapModel}>
                      {route.models.length > 0 ? route.models.map(m => m.model).join(', ') : '(no model)'}
                    </span>
                  </div>
                </div>
              ))}
              {defaultRoute && (
                <div className={styles.routeMapBranch} onClick={() => onSelect({ kind: 'route', name: defaultRoute.name })}>
                  <div className={styles.routeMapCondition}>
                    <span className={styles.routeMapCondIcon}>└─</span>
                    <code className={styles.routeMapCondText}>(no match / default)</code>
                  </div>
                  <div className={styles.routeMapTarget}>
                    <span className={styles.routeMapTargetArrow}>└→</span>
                    <span className={styles.routeMapRouteName}>&quot;{defaultRoute.name}&quot;</span>
                    <span className={styles.routeMapTargetArrow}>→</span>
                    <span className={styles.routeMapModel}>
                      {defaultRoute.models.length > 0 ? defaultRoute.models.map(m => m.model).join(', ') : '(no model)'}
                    </span>
                  </div>
                </div>
              )}
              {routes.length === 0 && (
                <div className={styles.routeMapEmpty}>No routes defined yet</div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Mode Switcher */}
      <div className={styles.dashSection}>
        <div className={styles.dashSectionTitle}>Editor Mode</div>
        <div className={styles.dashModes}>
          <button className={styles.dashModeBtn} onClick={() => onModeSwitch('visual')}>
            <span className={styles.dashModeBtnIcon}>📐</span>
            <span className={styles.dashModeBtnLabel}>Visual</span>
            <span className={styles.dashModeBtnDesc}>Drag & drop builder</span>
          </button>
          <button className={styles.dashModeBtn} onClick={() => onModeSwitch('dsl')}>
            <span className={styles.dashModeBtnIcon}>📝</span>
            <span className={styles.dashModeBtnLabel}>DSL</span>
            <span className={styles.dashModeBtnDesc}>Code editor</span>
          </button>
          <button className={styles.dashModeBtn} onClick={() => onModeSwitch('nl')}>
            <span className={styles.dashModeBtnIcon}>🤖</span>
            <span className={styles.dashModeBtnLabel}>Natural Language</span>
            <span className={styles.dashModeBtnDesc}>AI-powered (coming soon)</span>
          </button>
        </div>
      </div>
    </div>
  )
}

// ===================================================================
// Entity List View (shows all entities of a kind as a card grid)
// ===================================================================

interface EntityListViewProps {
  kind: EntityKind
  ast: ReturnType<typeof useDSLStore.getState>['ast']
  onSelect: (sel: Selection) => void
  onBack: () => void
  onAddEntity: (kind: EntityKind) => void
}

const EntityListView: React.FC<EntityListViewProps> = ({ kind, ast, onSelect, onBack, onAddEntity }) => {
  const META: Record<string, { title: string; icon: React.FC<{ className?: string }>; color: string }> = {
    signal: { title: 'Signals', icon: SignalIcon, color: 'rgb(118, 185, 0)' },
    route: { title: 'Routes', icon: RouteIcon, color: 'rgb(96, 165, 250)' },
    plugin: { title: 'Plugins', icon: PluginIcon, color: 'rgb(168, 130, 255)' },
    backend: { title: 'Backends', icon: BackendIcon, color: 'rgb(251, 191, 36)' },
  }
  const meta = META[kind]
  if (!meta) return null
  const Icon = meta.icon

  const items: { name: string; type: string; desc?: string }[] = (() => {
    switch (kind) {
      case 'signal':
        return (ast?.signals ?? []).map(s => ({ name: s.name, type: s.signalType, desc: Object.keys(s.fields).length > 0 ? `${Object.keys(s.fields).length} field(s)` : undefined }))
      case 'route':
        return (ast?.routes ?? []).map(r => ({
          name: r.name,
          type: r.when ? `P${r.priority}` : 'default',
          desc: r.models.length > 0 ? r.models.map(m => m.model).join(', ') : undefined,
        }))
      case 'plugin':
        return (ast?.plugins ?? []).map(p => ({ name: p.name, type: p.pluginType, desc: Object.keys(p.fields).length > 0 ? `${Object.keys(p.fields).length} field(s)` : undefined }))
      case 'backend':
        return (ast?.backends ?? []).map(b => ({ name: b.name, type: b.backendType, desc: Object.keys(b.fields).length > 0 ? `${Object.keys(b.fields).length} field(s)` : undefined }))
      default:
        return []
    }
  })()

  return (
    <div className={styles.entityListPanel}>
      <div className={styles.entityListHeader}>
        <button className={styles.backBtn} onClick={onBack} title="Back to Dashboard">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
        </button>
        <Icon className={styles.statIcon} />
        <span className={styles.entityListTitle}>{meta.title}</span>
        <span className={styles.entityListCount}>{items.length}</span>
        <div style={{ marginLeft: 'auto' }}>
          <button className={styles.quickActionBtn} onClick={() => onAddEntity(kind)} style={{ padding: '0.5rem 1rem', fontSize: '0.8125rem' }}>
            <span className={styles.quickActionIcon} style={{ width: 24, height: 24, fontSize: '0.875rem' }}>+</span>
            New {meta.title.replace(/s$/, '')}
          </button>
        </div>
      </div>
      <div className={styles.entityListGrid}>
        {items.map(item => (
          <div
            key={item.name}
            className={styles.entityListCard}
            onClick={() => onSelect({ kind, name: item.name })}
            style={{ '--entity-accent': meta.color } as React.CSSProperties}
          >
            <div className={styles.entityListCardHeader}>
              <Icon className={styles.entityListCardIcon} />
              <span className={styles.entityListCardName}>{item.name}</span>
            </div>
            <span className={styles.entityListCardType}>{item.type}</span>
            {item.desc && <span className={styles.entityListCardDesc}>{item.desc}</span>}
            <div className={styles.entityListCardArrow}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 6 15 12 9 18"/></svg>
            </div>
          </div>
        ))}
      </div>
      {items.length === 0 && (
        <div className={styles.emptyState}>
          <div className={styles.emptyIcon}>
            <Icon className={styles.statIcon} />
          </div>
          <div>No {meta.title.toLowerCase()} defined yet</div>
          <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
            Click the button above to create one
          </div>
        </div>
      )}
    </div>
  )
}

// ===================================================================
// Entity Detail View (editable for Phase 2)
// ===================================================================

interface EntityDetailViewProps {
  selection: Selection
  entity: ASTSignalDecl | ASTRouteDecl | ASTPluginDecl | ASTBackendDecl | { fields: Record<string, unknown> } | null
  ast: ReturnType<typeof useDSLStore.getState>['ast']
  onDeleteEntity: (kind: EntityKind, name: string, subType?: string) => void
  onUpdateSignalFields: (signalType: string, name: string, fields: Record<string, unknown>) => void
  onUpdatePluginFields: (name: string, pluginType: string, fields: Record<string, unknown>) => void
  onUpdateBackendFields: (backendType: string, name: string, fields: Record<string, unknown>) => void
  onUpdateRoute: (name: string, input: RouteInput) => void
  onUpdateGlobalFields: (fields: Record<string, unknown>) => void
  availableSignals: { signalType: string; name: string }[]
  availablePlugins: { name: string; pluginType: string }[]
  availableModels: string[]
  onBack: () => void
}

const EntityDetailView: React.FC<EntityDetailViewProps> = ({
  selection,
  entity,
  ast,
  onDeleteEntity,
  onUpdateSignalFields,
  onUpdatePluginFields,
  onUpdateBackendFields,
  onUpdateRoute,
  onUpdateGlobalFields,
  availableSignals,
  availablePlugins,
  availableModels,
  onBack,
}) => {
  if (!entity) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyIcon}>🔍</div>
        <div>Entity &quot;{selection.name}&quot; not found in current AST</div>
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
          Try compiling or validating your DSL first
        </div>
      </div>
    )
  }

  const subType = 'signalType' in entity ? entity.signalType
    : 'pluginType' in entity ? entity.pluginType
    : 'backendType' in entity ? entity.backendType
    : undefined

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <button className={styles.backBtn} onClick={onBack} title="Back to Dashboard">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"/></svg>
        </button>
        <div className={styles.editorTitle}>
          {selection.kind === 'signal' && <SignalIcon className={styles.statIcon} />}
          {selection.kind === 'route' && <RouteIcon className={styles.statIcon} />}
          {selection.kind === 'plugin' && <PluginIcon className={styles.statIcon} />}
          {selection.kind === 'backend' && <BackendIcon className={styles.statIcon} />}
          {selection.kind === 'global' && <GlobalIcon className={styles.statIcon} />}
          {'name' in entity ? entity.name : 'Global Settings'}
          {'signalType' in entity && <span className={styles.editorBadge}>{entity.signalType}</span>}
          {'pluginType' in entity && <span className={styles.editorBadge}>{entity.pluginType}</span>}
          {'backendType' in entity && <span className={styles.editorBadge}>{entity.backendType}</span>}
        </div>
        {selection.kind !== 'global' && (
          <div className={styles.editorActions}>
            <button
              className={styles.toolbarBtnDanger}
              onClick={() => onDeleteEntity(selection.kind, selection.name, subType)}
              title="Delete this entity"
            >
              Delete
            </button>
          </div>
        )}
      </div>

      {/* Editable Signal form */}
      {selection.kind === 'signal' && 'signalType' in entity && (
        <SignalEditorForm
          signal={entity as ASTSignalDecl}
          onUpdate={(fields) => onUpdateSignalFields((entity as ASTSignalDecl).signalType, (entity as ASTSignalDecl).name, fields)}
        />
      )}

      {/* Editable Plugin form */}
      {selection.kind === 'plugin' && 'pluginType' in entity && (
        <PluginSchemaEditor
          pluginType={(entity as ASTPluginDecl).pluginType}
          fields={'fields' in entity ? (entity.fields as Record<string, unknown>) : {}}
          onUpdate={(fields) => onUpdatePluginFields((entity as ASTPluginDecl).name, (entity as ASTPluginDecl).pluginType, fields)}
          buffered
        />
      )}

      {/* Editable Backend form */}
      {selection.kind === 'backend' && 'backendType' in entity && (
        <GenericFieldsEditor
          fields={'fields' in entity ? (entity.fields as Record<string, unknown>) : {}}
          onUpdate={(fields) => onUpdateBackendFields((entity as ASTBackendDecl).backendType, (entity as ASTBackendDecl).name, fields)}
        />
      )}

      {/* Editable Route form */}
      {selection.kind === 'route' && 'priority' in entity && (
        <RouteEditorForm
          route={entity as ASTRouteDecl}
          onUpdate={(input) => onUpdateRoute((entity as ASTRouteDecl).name, input)}
          availableSignals={availableSignals}
          availablePlugins={availablePlugins}
          availableModels={availableModels}
        />
      )}

      {/* Editable Global form — structured sections */}
      {selection.kind === 'global' && (
        <GlobalSettingsEditor
          fields={'fields' in entity ? (entity.fields as Record<string, unknown>) : {}}
          onUpdate={onUpdateGlobalFields}
          endpoints={ast?.backends?.filter(b => b.backendType === 'vllm_endpoint' || b.backendType === 'provider_profile') ?? []}
          onSelectEndpoint={() => { onBack(); setTimeout(() => onBack(), 0) }}
        />
      )}
    </div>
  )
}

// ===================================================================
// Route DSL Preview generator (shared by RouteEditorForm + AddRouteForm)
// ===================================================================

function generateRouteDslPreview(
  routeName: string,
  description: string,
  priority: number,
  whenExpr: string,
  models: RouteModelInput[],
  algorithm: RouteAlgoInput | undefined,
  plugins: RoutePluginInput[],
): string {
  const descPart = description.trim() ? ` (description = "${description.trim()}")` : ''
  const lines: string[] = [`ROUTE ${routeName}${descPart} {`]
  lines.push(`  PRIORITY ${priority}`)
  if (whenExpr.trim()) {
    lines.push('')
    lines.push(`  WHEN ${whenExpr.trim().replace(/\s+/g, ' ')}`)
  }
  if (models.length > 0) {
    lines.push('')
    const modelStrs = models.filter(m => m.model.trim()).map(m => {
      const attrs: string[] = []
      if (m.reasoning) attrs.push(`reasoning = true`)
      if (m.effort) attrs.push(`effort = "${m.effort}"`)
      if (m.paramSize) attrs.push(`param_size = "${m.paramSize}"`)
      if (m.weight !== undefined) attrs.push(`weight = ${m.weight}`)
      const attrStr = attrs.length > 0 ? ` (${attrs.join(', ')})` : ''
      return `"${m.model}"${attrStr}`
    })
    if (modelStrs.length === 1) {
      lines.push(`  MODEL ${modelStrs[0]}`)
    } else if (modelStrs.length > 1) {
      lines.push(`  MODEL ${modelStrs.join(',\n        ')}`)
    }
  }
  if (algorithm?.algoType) {
    lines.push('')
    const aFields = Object.entries(algorithm.fields).filter(([, v]) => v !== undefined && v !== '')
    if (aFields.length > 0) {
      lines.push(`  ALGORITHM ${algorithm.algoType} {`)
      aFields.forEach(([k, v]) => {
        let formatted: string
        if (Array.isArray(v)) formatted = `[${v.join(', ')}]`
        else if (typeof v === 'string') formatted = `"${v}"`
        else formatted = String(v)
        lines.push(`    ${k}: ${formatted}`)
      })
      lines.push(`  }`)
    } else {
      lines.push(`  ALGORITHM ${algorithm.algoType}`)
    }
  }
  if (plugins.length > 0) {
    lines.push('')
    plugins.forEach(p => {
      if (p.fields && Object.keys(p.fields).length > 0) {
        lines.push(`  PLUGIN ${p.name} {`)
        Object.entries(p.fields).forEach(([k, v]) => {
          let formatted: string
          if (Array.isArray(v)) formatted = `[${v.join(', ')}]`
          else if (typeof v === 'string') formatted = `"${v}"`
          else formatted = String(v)
          lines.push(`    ${k}: ${formatted}`)
        })
        lines.push(`  }`)
      } else {
        lines.push(`  PLUGIN ${p.name}`)
      }
    })
  }
  lines.push('}')
  return lines.join('\n')
}

// ===================================================================
// Signal DSL Preview generator
// ===================================================================

function generateSignalDslPreview(
  signalType: string,
  signalName: string,
  fields: Record<string, unknown>,
): string {
  const body = serializeFields(fields)
  if (!body.trim()) {
    return `SIGNAL ${signalType} ${signalName} {}`
  }
  return `SIGNAL ${signalType} ${signalName} {\n${body}\n}`
}

// ===================================================================
// Global DSL Preview generator (matches Go decompiler format)
// ===================================================================

function generateGlobalDslPreview(fields: Record<string, unknown>): string {
  const body = serializeFields(fields, '  ', { blankLineBefore: true })
  if (!body.trim()) return 'GLOBAL {}'
  return `GLOBAL {\n${body}\n}`
}

// ===================================================================
// Generic DSL Preview Panel (reusable for Signal, Global, etc.)
// ===================================================================

const DslPreviewPanel: React.FC<{
  title?: string
  dslText: string
}> = ({ title = 'DSL Preview', dslText }) => {
  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>{title}</span>
      </div>
      <pre className={styles.dslPreviewCode}>{dslText}</pre>
    </div>
  )
}

// ===================================================================
// Route validation hints (local, pre-submit)
// ===================================================================

interface ValidationIssue {
  level: 'error' | 'warning' | 'constraint'
  message: string
}

function validateRouteInput(
  routeName: string,
  models: RouteModelInput[],
  algorithm: RouteAlgoInput | undefined,
  _plugins: RoutePluginInput[],
): ValidationIssue[] {
  const issues: ValidationIssue[] = []

  // Route name
  if (!routeName.trim()) {
    issues.push({ level: 'error', message: 'Route name is required' })
  }

  // Models
  const validModels = models.filter(m => m.model.trim())
  if (validModels.length === 0) {
    issues.push({ level: 'warning', message: 'No model specified — route needs at least one MODEL' })
  }

  // Algorithm field validation
  if (algorithm?.algoType) {
    const schema = getAlgorithmFieldSchema(algorithm.algoType)

    // Check required fields
    schema.filter(f => f.required).forEach(f => {
      const v = algorithm.fields[f.key]
      if (v === undefined || v === '' || v === null || (Array.isArray(v) && v.length === 0)) {
        issues.push({ level: 'error', message: `Algorithm field "${f.label}" is required` })
      }
    })

    // Number range checks
    const fields = algorithm.fields
    if (algorithm.algoType === 'confidence') {
      const t = fields['threshold']
      if (t !== undefined && t !== '' && typeof t === 'number' && (t < -100 || t > 0)) {
        issues.push({ level: 'warning', message: `Threshold ${t} — typically negative log-prob (e.g. -1.0)` })
      }
    }
    if (algorithm.algoType === 'remom') {
      const mc = fields['max_concurrent']
      if (mc !== undefined && mc !== '' && typeof mc === 'number' && mc < 0) {
        issues.push({ level: 'error', message: `max_concurrent cannot be negative (got ${mc})` })
      }
      const temp = fields['temperature']
      if (temp !== undefined && temp !== '' && typeof temp === 'number' && temp < 0) {
        issues.push({ level: 'error', message: `temperature cannot be negative (got ${temp})` })
      }
    }
    if (algorithm.algoType === 'elo') {
      const k = fields['k_factor']
      if (k !== undefined && k !== '' && typeof k === 'number' && k <= 0) {
        issues.push({ level: 'warning', message: `k_factor should be positive (got ${k})` })
      }
    }
    if (algorithm.algoType === 'latency_aware') {
      for (const key of ['tpot_percentile', 'ttft_percentile']) {
        const v = fields[key]
        if (v !== undefined && v !== '' && typeof v === 'number' && (v < 1 || v > 100)) {
          issues.push({ level: 'error', message: `${key} must be 1-100 (got ${v})` })
        }
      }
    }
    if (algorithm.algoType === 'ratings' || algorithm.algoType === 'remom') {
      const mc = fields['max_concurrent']
      if (mc !== undefined && mc !== '' && typeof mc === 'number' && mc < 0) {
        issues.push({ level: 'error', message: `max_concurrent cannot be negative (got ${mc})` })
      }
    }

    // Multi-model recommendation
    if (validModels.length < 2 && ['confidence', 'ratings', 'elo', 'hybrid', 'automix'].includes(algorithm.algoType)) {
      issues.push({ level: 'constraint', message: `Algorithm "${algorithm.algoType}" works best with multiple models` })
    }
  }

  return issues
}

// ===================================================================
// Route DSL Preview with validation badges
// ===================================================================

const ISSUE_ICONS: Record<string, string> = {
  error: '✕',
  warning: '⚠',
  constraint: 'ℹ',
}

const ISSUE_COLORS: Record<string, string> = {
  error: '#ff5555',
  warning: '#f1c40f',
  constraint: '#5dade2',
}

const RouteDslPreviewPanel: React.FC<{
  dslText: string
  issues: ValidationIssue[]
  /** Diagnostics from WASM for this route (line-matched) */
  wasmDiagnostics?: { level: string; message: string }[]
}> = ({ dslText, issues, wasmDiagnostics = [] }) => {
  const allIssues = useMemo(() => {
    const merged: ValidationIssue[] = [...issues]
    wasmDiagnostics.forEach(d => {
      merged.push({ level: d.level as ValidationIssue['level'], message: d.message })
    })
    return merged
  }, [issues, wasmDiagnostics])

  const errorCount = allIssues.filter(i => i.level === 'error').length
  const warnCount = allIssues.filter(i => i.level === 'warning').length
  const constraintCount = allIssues.filter(i => i.level === 'constraint').length

  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>
          DSL Preview
          {allIssues.length > 0 && (
            <span style={{ marginLeft: '0.5rem', fontSize: '0.625rem', fontWeight: 400 }}>
              {errorCount > 0 && <span style={{ color: ISSUE_COLORS.error, marginRight: '0.5rem' }}>{errorCount} error{errorCount > 1 ? 's' : ''}</span>}
              {warnCount > 0 && <span style={{ color: ISSUE_COLORS.warning, marginRight: '0.5rem' }}>{warnCount} warning{warnCount > 1 ? 's' : ''}</span>}
              {constraintCount > 0 && <span style={{ color: ISSUE_COLORS.constraint }}>{constraintCount} hint{constraintCount > 1 ? 's' : ''}</span>}
            </span>
          )}
        </span>
      </div>
      <pre className={styles.dslPreviewCode}>{dslText}</pre>
      {allIssues.length > 0 && (
        <div style={{
          padding: '0.5rem var(--spacing-md)',
          borderTop: '1px solid var(--color-border)',
          display: 'flex', flexDirection: 'column', gap: '0.25rem',
        }}>
          {allIssues.map((issue, i) => (
            <div key={i} style={{
              display: 'flex', alignItems: 'flex-start', gap: '0.5rem',
              fontSize: '0.6875rem', lineHeight: 1.4,
            }}>
              <span style={{
                color: ISSUE_COLORS[issue.level],
                fontWeight: 700, flexShrink: 0, width: '1rem', textAlign: 'center',
              }}>
                {ISSUE_ICONS[issue.level]}
              </span>
              <span style={{ color: ISSUE_COLORS[issue.level] }}>{issue.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ===================================================================
// Route Editor Form (editable)
// ===================================================================

function astModelToInput(m: ASTModelRef): RouteModelInput {
  return {
    model: m.model,
    reasoning: m.reasoning,
    effort: m.effort,
    lora: m.lora,
    paramSize: m.paramSize,
    weight: m.weight,
    reasoningFamily: m.reasoningFamily,
  }
}

function astAlgoToInput(a?: ASTAlgoSpec): RouteAlgoInput | undefined {
  if (!a) return undefined
  return { algoType: a.algoType, fields: { ...a.fields } }
}

function astPluginRefToInput(p: ASTPluginRef): RoutePluginInput {
  return { name: p.name, fields: p.fields ? { ...p.fields } : undefined }
}

const RouteEditorForm: React.FC<{
  route: ASTRouteDecl
  onUpdate: (input: RouteInput) => void
  availableSignals: { signalType: string; name: string }[]
  availablePlugins: { name: string; pluginType: string }[]
  availableModels: string[]
}> = ({ route, onUpdate, availableSignals, availablePlugins, availableModels }) => {
  const [description, setDescription] = useState(route.description ?? '')
  const [priority, setPriority] = useState(route.priority)
  const [whenExpr, setWhenExpr] = useState(() =>
    route.when ? serializeBoolExpr(route.when as unknown as Record<string, unknown>) : ''
  )
  const [models, setModels] = useState<RouteModelInput[]>(() => route.models.map(astModelToInput))
  const [algorithm, setAlgorithm] = useState<RouteAlgoInput | undefined>(() => astAlgoToInput(route.algorithm))
  const [plugins, setPlugins] = useState<RoutePluginInput[]>(() => route.plugins.map(astPluginRefToInput))

  // Sync from parent when route changes
  useEffect(() => {
    setDescription(route.description ?? '')
    setPriority(route.priority)
    setWhenExpr(route.when ? serializeBoolExpr(route.when as unknown as Record<string, unknown>) : '')
    setModels(route.models.map(astModelToInput))
    setAlgorithm(astAlgoToInput(route.algorithm))
    setPlugins(route.plugins.map(astPluginRefToInput))
  }, [route.name, route.priority, route.description, route.when, route.models, route.algorithm, route.plugins])

  const handleSave = useCallback(() => {
    onUpdate({
      description: description.trim() || undefined,
      priority,
      when: whenExpr.trim() || undefined,
      models,
      algorithm: algorithm?.algoType ? algorithm : undefined,
      plugins,
    })
  }, [description, priority, whenExpr, models, algorithm, plugins, onUpdate])

  // Model helpers
  const addModel = useCallback(() => {
    setModels(prev => [...prev, { model: '' }])
  }, [])

  const removeModel = useCallback((idx: number) => {
    setModels(prev => prev.filter((_, i) => i !== idx))
  }, [])

  const updateModel = useCallback((idx: number, patch: Partial<RouteModelInput>) => {
    setModels(prev => prev.map((m, i) => i === idx ? { ...m, ...patch } : m))
  }, [])

  // Plugin toggle helpers
  const activePluginNames = useMemo(() => new Set(plugins.map(p => p.name)), [plugins])

  const togglePlugin = useCallback((pluginName: string) => {
    setPlugins(prev => {
      const exists = prev.find(p => p.name === pluginName)
      if (exists) return prev.filter(p => p.name !== pluginName)
      return [...prev, { name: pluginName }]
    })
  }, [])

  const updatePluginFields = useCallback((pluginName: string, fields: Record<string, unknown>) => {
    setPlugins(prev => prev.map(p => p.name === pluginName ? { ...p, fields } : p))
  }, [])

  // Expression builder: tree-based, managed by ExpressionBuilder component

  // Generate DSL preview & validation
  const dslPreview = useMemo(() =>
    generateRouteDslPreview(route.name, description, priority, whenExpr, models, algorithm, plugins),
    [route.name, description, priority, whenExpr, models, algorithm, plugins]
  )

  const validationIssues = useMemo(() =>
    validateRouteInput(route.name, models, algorithm, plugins),
    [route.name, models, algorithm, plugins]
  )

  // Get WASM diagnostics scoped to this route
  const diagnostics = useDSLStore(s => s.diagnostics)
  const routeDiagnostics = useMemo(() => {
    if (!route.pos?.Line) return []
    const startLine = route.pos.Line
    return diagnostics.filter(d => d.line >= startLine && d.line <= startLine + 50)
      .map(d => ({ level: d.level, message: d.message }))
  }, [diagnostics, route.pos])

  return (
    <>
      {/* Header with Save */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Route Configuration</span>
          <button className={styles.toolbarBtnPrimary} onClick={handleSave} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>
            Save
          </button>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 'var(--spacing-md)' }}>
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Description</label>
              <input
                className={styles.fieldInput}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Route description..."
              />
            </div>
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Priority <span style={{ color: 'var(--color-danger)' }}>*</span></label>
              <input
                className={styles.fieldInput}
                type="number"
                value={priority}
                onChange={(e) => setPriority(Number(e.target.value) || 0)}
                style={{ width: '100px' }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* WHEN Expression Builder */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>WHEN (Expression Builder)</span>
        </div>
        <div style={{ padding: 'var(--spacing-md)', minHeight: '350px', maxHeight: '50vh', display: 'flex', flexDirection: 'column' }}>
          <ExpressionBuilder
            value={whenExpr}
            onChange={setWhenExpr}
            initialAstExpr={route.when as unknown as Record<string, unknown> | null}
            availableSignals={availableSignals}
          />
        </div>
      </div>

      {/* Models */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Models ({models.length})</span>
          <button className={styles.toolbarBtn} onClick={addModel} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>
            + Add Model
          </button>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
          {models.length === 0 && (
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>No models configured. Add at least one model.</span>
          )}
          {models.map((m, idx) => (
            <div key={idx} className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelIndex}>{idx + 1}</span>
                <ModelNameInput
                  value={m.model}
                  availableModels={availableModels}
                  onChange={(v) => updateModel(idx, { model: v })}
                />
                <button
                  className={styles.toolbarBtnDanger}
                  onClick={() => removeModel(idx)}
                  style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)', flexShrink: 0 }}
                  title="Remove model"
                >
                  ×
                </button>
              </div>
              <div className={styles.modelAttrs}>
                <label className={styles.modelAttrCheck}>
                  <input
                    type="checkbox"
                    checked={m.reasoning ?? false}
                    onChange={(e) => updateModel(idx, { reasoning: e.target.checked || undefined })}
                    style={{ accentColor: 'var(--color-primary)' }}
                  />
                  reasoning
                </label>
                <div className={styles.modelAttrField}>
                  <span className={styles.modelAttrLabel}>effort:</span>
                  <div style={{ minWidth: '90px' }}>
                    <CustomSelect
                      value={m.effort ?? ''}
                      options={['', 'low', 'medium', 'high']}
                      onChange={(v) => updateModel(idx, { effort: v || undefined })}
                      placeholder="—"
                    />
                  </div>
                </div>
                <div className={styles.modelAttrField}>
                  <span className={styles.modelAttrLabel}>weight:</span>
                  <input
                    className={styles.fieldInput}
                    style={{ width: '60px', fontSize: 'var(--text-xs)', padding: '0.25rem 0.5rem' }}
                    type="number"
                    step="any"
                    value={m.weight !== undefined ? m.weight : ''}
                    onChange={(e) => updateModel(idx, { weight: e.target.value ? Number(e.target.value) : undefined })}
                    placeholder="—"
                  />
                </div>
                <div className={styles.modelAttrField}>
                  <span className={styles.modelAttrLabel}>param_size:</span>
                  <input
                    className={styles.fieldInput}
                    style={{ width: '70px', fontSize: 'var(--text-xs)', padding: '0.25rem 0.5rem' }}
                    value={m.paramSize ?? ''}
                    onChange={(e) => updateModel(idx, { paramSize: e.target.value || undefined })}
                    placeholder="—"
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Algorithm {models.length >= 2 ? '' : '(optional — for multi-model)'}</span>
          {!algorithm && (
            <button className={styles.toolbarBtn} onClick={() => setAlgorithm({ algoType: 'confidence', fields: {} })} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>
              + Add
            </button>
          )}
          {algorithm && (
            <button className={styles.toolbarBtnDanger} onClick={() => setAlgorithm(undefined)} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>
              Remove
            </button>
          )}
        </div>
        {algorithm && (
          <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Algorithm Type</label>
              <CustomSelect
                value={algorithm.algoType}
                options={[...ALGORITHM_TYPES]}
                onChange={(v) => setAlgorithm({ algoType: v, fields: {} })}
              />
              {ALGORITHM_DESCRIPTIONS[algorithm.algoType] && (
                <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                  {ALGORITHM_DESCRIPTIONS[algorithm.algoType]}
                </span>
              )}
            </div>
            <AlgorithmSchemaEditor
              algoType={algorithm.algoType}
              fields={algorithm.fields}
              onUpdate={(f) => setAlgorithm({ ...algorithm, fields: f })}
            />
          </div>
        )}
        {!algorithm && (
          <div style={{ padding: 'var(--spacing-md)', fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
            No algorithm configured. {models.length >= 2 ? 'Recommended when using multiple models.' : ''}
          </div>
        )}
      </div>

      {/* Plugins Toggle Panel */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Plugins ({plugins.length})</span>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
          {/* Toggle chips for available plugins */}
          {availablePlugins.length > 0 && (
            <div className={styles.pluginToggleGrid}>
              {availablePlugins.map(p => {
                const active = activePluginNames.has(p.name)
                return (
                  <button
                    key={p.name}
                    className={active ? styles.pluginToggleActive : styles.pluginToggle}
                    onClick={() => togglePlugin(p.name)}
                    title={`${active ? 'Remove' : 'Add'} plugin ${p.name}`}
                  >
                    <span className={styles.pluginToggleCheck}>{active ? '✓' : '○'}</span>
                    <span className={styles.pluginToggleName}>{p.name}</span>
                    <span className={styles.pluginToggleType}>{p.pluginType}</span>
                  </button>
                )
              })}
            </div>
          )}
          {availablePlugins.length === 0 && (
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>No plugins defined. Create plugins first.</span>
          )}

          {/* Active plugin configuration editors */}
          {plugins.length > 0 && (
            <div style={{ marginTop: 'var(--spacing-sm)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
              <span className={styles.fieldLabel} style={{ display: 'block' }}>Plugin Configuration</span>
              {plugins.map(p => {
                // Resolve pluginType: from top-level template, or treat name as type for inline plugins
                const tmpl = availablePlugins.find(ap => ap.name === p.name)
                const pluginType = tmpl?.pluginType ?? p.name
                return (
                  <div key={p.name} className={styles.pluginOverride}>
                    <PluginSchemaEditor
                      pluginType={pluginType}
                      pluginName={p.name}
                      fields={p.fields ?? {}}
                      onUpdate={(f) => updatePluginFields(p.name, f)}
                      compact
                    />
                  </div>
                )
              })}
            </div>
          )}

          {/* Manual plugin add (for inline plugins not in templates) */}
          <ManualPluginAdder
            existingNames={activePluginNames}
            onAdd={(name, fields) => setPlugins(prev => [...prev, { name, fields }])}
          />
        </div>
      </div>

      {/* DSL Preview with validation */}
      <RouteDslPreviewPanel
        dslText={dslPreview}
        issues={validationIssues}
        wasmDiagnostics={routeDiagnostics}
      />
    </>
  )
}

// ===================================================================
// Algorithm Schema Editor (typed fields per algorithm type)
// ===================================================================

const AlgorithmSchemaEditor: React.FC<{
  algoType: string
  fields: Record<string, unknown>
  onUpdate: (fields: Record<string, unknown>) => void
}> = ({ algoType, fields, onUpdate }) => {
  const schema = useMemo(() => getAlgorithmFieldSchema(algoType), [algoType])

  const updateField = useCallback((key: string, value: unknown) => {
    onUpdate({ ...fields, [key]: value })
  }, [fields, onUpdate])

  if (schema.length === 0) {
    return (
      <div style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>
        No configurable fields for this algorithm type.
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
      {schema.map((field) => (
        <FieldEditor
          key={field.key}
          schema={field}
          value={fields[field.key]}
          onChange={(v) => updateField(field.key, v)}
        />
      ))}
      {/* Extra custom fields not in schema */}
      <ExtraFieldsEditor fields={fields} schemaKeys={schema.map(f => f.key)} onUpdate={onUpdate} />
    </div>
  )
}

/** Show any extra fields in the record that aren't covered by the schema */
const ExtraFieldsEditor: React.FC<{
  fields: Record<string, unknown>
  schemaKeys: string[]
  onUpdate: (fields: Record<string, unknown>) => void
}> = ({ fields, schemaKeys, onUpdate }) => {
  const extraEntries = useMemo(() => {
    const known = new Set(schemaKeys)
    return Object.entries(fields).filter(([k]) => !known.has(k))
  }, [fields, schemaKeys])

  const [newKey, setNewKey] = useState('')

  const updateField = useCallback((key: string, rawValue: string) => {
    const parsed = tryParseValue(rawValue)
    onUpdate({ ...fields, [key]: parsed })
  }, [fields, onUpdate])

  const deleteField = useCallback((key: string) => {
    const next = { ...fields }
    delete next[key]
    onUpdate(next)
  }, [fields, onUpdate])

  const addField = useCallback(() => {
    const k = newKey.trim()
    if (!k || k in fields) return
    onUpdate({ ...fields, [k]: '' })
    setNewKey('')
  }, [newKey, fields, onUpdate])

  return (
    <>
      {extraEntries.map(([key, value]) => (
        <div key={key} style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--spacing-sm)' }}>
          <span style={{
            minWidth: '100px', fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)',
            fontFamily: 'var(--font-mono)', paddingTop: '0.5rem',
          }}>
            {key}
          </span>
          <input
            className={styles.fieldInput}
            style={{ flex: 1, fontSize: 'var(--text-xs)' }}
            value={typeof value === 'string' ? value : JSON.stringify(value)}
            onChange={(e) => updateField(key, e.target.value)}
          />
          <button
            className={styles.toolbarBtnDanger}
            onClick={() => deleteField(key)}
            style={{ padding: '0.375rem', fontSize: 'var(--text-xs)', flexShrink: 0 }}
          >
            ×
          </button>
        </div>
      ))}
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
        <input
          className={styles.fieldInput}
          style={{ flex: 1, fontSize: 'var(--text-xs)' }}
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          placeholder="Add custom field..."
          onKeyDown={(e) => e.key === 'Enter' && addField()}
        />
        <button
          className={styles.toolbarBtn}
          onClick={addField}
          disabled={!newKey.trim()}
          style={{ padding: '0.375rem 0.5rem', fontSize: 'var(--text-xs)' }}
        >
          + Add
        </button>
      </div>
    </>
  )
}

// ===================================================================
// Model Name Input (combo: dropdown + manual text input)
// ===================================================================

const ModelNameInput: React.FC<{
  value: string
  availableModels: string[]
  onChange: (value: string) => void
}> = ({ value, availableModels, onChange }) => {
  const [open, setOpen] = useState(false)
  const [inputValue, setInputValue] = useState(value)
  const ref = useRef<HTMLDivElement>(null)

  // Sync external value
  useEffect(() => { setInputValue(value) }, [value])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const filteredModels = useMemo(() => {
    if (!inputValue.trim()) return availableModels
    const lower = inputValue.toLowerCase()
    return availableModels.filter(m => m.toLowerCase().includes(lower))
  }, [inputValue, availableModels])

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value)
    onChange(e.target.value)
    if (!open && availableModels.length > 0) setOpen(true)
  }, [onChange, open, availableModels.length])

  const handleSelect = useCallback((model: string) => {
    setInputValue(model)
    onChange(model)
    setOpen(false)
  }, [onChange])

  return (
    <div ref={ref} style={{ flex: 1, position: 'relative' }}>
      <div style={{ display: 'flex', gap: '0' }}>
        <input
          className={styles.fieldInput}
          style={{ flex: 1, fontSize: 'var(--text-xs)', borderTopRightRadius: availableModels.length > 0 ? 0 : undefined, borderBottomRightRadius: availableModels.length > 0 ? 0 : undefined }}
          value={inputValue}
          onChange={handleInputChange}
          onFocus={() => availableModels.length > 0 && setOpen(true)}
          placeholder="model name (e.g. qwen3:70b)"
        />
        {availableModels.length > 0 && (
          <button
            className={styles.modelDropdownBtn}
            onClick={() => setOpen(!open)}
            title="Select from known models"
            type="button"
          >
            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </div>
      {open && filteredModels.length > 0 && (
        <div className={styles.customSelectDropdown}>
          {filteredModels.map(m => (
            <div
              key={m}
              className={m === value ? styles.customSelectOptionActive : styles.customSelectOption}
              onClick={() => handleSelect(m)}
            >
              {m === value && (
                <svg className={styles.customSelectCheck} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
              {m !== value && <span className={styles.customSelectPlaceholder} />}
              {m}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ===================================================================
// Manual Plugin Adder (for inline plugins not in templates)
// ===================================================================

const ManualPluginAdder: React.FC<{
  existingNames: Set<string>
  onAdd: (name: string, fields?: Record<string, unknown>) => void
}> = ({ existingNames, onAdd }) => {
  const [name, setName] = useState('')

  const handleAdd = useCallback(() => {
    const n = name.trim()
    if (!n || existingNames.has(n)) return
    onAdd(n)
    setName('')
  }, [name, existingNames, onAdd])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', marginTop: 'var(--spacing-xs)' }}>
      <input
        className={styles.fieldInput}
        style={{ flex: 1, fontSize: 'var(--text-xs)' }}
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Add inline plugin by name..."
        onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
      />
      <button
        className={styles.toolbarBtn}
        onClick={handleAdd}
        disabled={!name.trim() || existingNames.has(name.trim())}
        style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}
      >
        + Add
      </button>
    </div>
  )
}

// ===================================================================
// Add Route Form
// ===================================================================

const AddRouteForm: React.FC<{
  onAdd: (name: string, input: RouteInput) => void
  onCancel: () => void
  availableSignals: { signalType: string; name: string }[]
  availablePlugins: { name: string; pluginType: string }[]
  availableModels: string[]
}> = ({ onAdd, onCancel, availableSignals, availablePlugins, availableModels }) => {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [priority, setPriority] = useState(100)
  const [whenExpr, setWhenExpr] = useState('')
  const [models, setModels] = useState<RouteModelInput[]>([{ model: '' }])
  const [algorithm, setAlgorithm] = useState<RouteAlgoInput | undefined>(undefined)
  const [plugins, setPlugins] = useState<RoutePluginInput[]>([])

  const handleSubmit = useCallback(() => {
    const n = name.trim().replace(/\s+/g, '_')
    if (!n) return
    onAdd(n, {
      description: description.trim() || undefined,
      priority,
      when: whenExpr.trim() || undefined,
      models: models.filter(m => m.model.trim()),
      algorithm: algorithm?.algoType ? algorithm : undefined,
      plugins,
    })
  }, [name, description, priority, whenExpr, models, algorithm, plugins, onAdd])

  const addModel = useCallback(() => {
    setModels(prev => [...prev, { model: '' }])
  }, [])

  const removeModel = useCallback((idx: number) => {
    setModels(prev => prev.filter((_, i) => i !== idx))
  }, [])

  const updateModel = useCallback((idx: number, patch: Partial<RouteModelInput>) => {
    setModels(prev => prev.map((m, i) => i === idx ? { ...m, ...patch } : m))
  }, [])

  // Expression builder: tree-based, managed by ExpressionBuilder component

  // Generate DSL preview & validation for AddRouteForm
  const routeName = useMemo(() => name.trim().replace(/\s+/g, '_') || 'new_route', [name])
  const dslPreview = useMemo(() =>
    generateRouteDslPreview(routeName, description, priority, whenExpr, models, algorithm, plugins),
    [routeName, description, priority, whenExpr, models, algorithm, plugins]
  )
  const validationIssues = useMemo(() =>
    validateRouteInput(name.trim(), models, algorithm, plugins),
    [name, models, algorithm, plugins]
  )

  const activePluginNames = useMemo(() => new Set(plugins.map(p => p.name)), [plugins])

  const togglePlugin = useCallback((pluginName: string) => {
    setPlugins(prev => {
      const exists = prev.find(p => p.name === pluginName)
      if (exists) return prev.filter(p => p.name !== pluginName)
      return [...prev, { name: pluginName }]
    })
  }, [])

  const updatePluginFields = useCallback((pluginName: string, fields: Record<string, unknown>) => {
    setPlugins(prev => prev.map(p => p.name === pluginName ? { ...p, fields } : p))
  }, [])

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <RouteIcon className={styles.statIcon} />
          New Route
        </div>
        <div className={styles.editorActions}>
          <button className={styles.toolbarBtn} onClick={onCancel}>Cancel</button>
          <button className={styles.toolbarBtnPrimary} onClick={handleSubmit} disabled={!name.trim()}>Create</button>
        </div>
      </div>

      {/* Basic fields */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Route Configuration</span>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 'var(--spacing-md)' }}>
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Name <span style={{ color: 'var(--color-danger)' }}>*</span></label>
              <input className={styles.fieldInput} value={name} onChange={(e) => setName(e.target.value)} placeholder="my_route" autoFocus />
            </div>
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Priority <span style={{ color: 'var(--color-danger)' }}>*</span></label>
              <input className={styles.fieldInput} type="number" value={priority} onChange={(e) => setPriority(Number(e.target.value) || 0)} style={{ width: '100px' }} />
            </div>
          </div>
          <div className={styles.fieldGroup}>
            <label className={styles.fieldLabel}>Description</label>
            <input className={styles.fieldInput} value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Route description..." />
          </div>
        </div>
      </div>

      {/* WHEN Expression Builder */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>WHEN (Expression Builder)</span>
        </div>
        <div style={{ padding: 'var(--spacing-md)', minHeight: '350px', maxHeight: '50vh', display: 'flex', flexDirection: 'column' }}>
          <ExpressionBuilder
            value={whenExpr}
            onChange={setWhenExpr}
            availableSignals={availableSignals}
          />
        </div>
      </div>

      {/* Models */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Models ({models.length})</span>
          <button className={styles.toolbarBtn} onClick={addModel} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>+ Add</button>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
          {models.map((m, idx) => (
            <div key={idx} className={styles.modelCard}>
              <div className={styles.modelCardHeader}>
                <span className={styles.modelIndex}>{idx + 1}</span>
                <ModelNameInput
                  value={m.model}
                  availableModels={availableModels}
                  onChange={(v) => updateModel(idx, { model: v })}
                />
                {models.length > 1 && (
                  <button className={styles.toolbarBtnDanger} onClick={() => removeModel(idx)} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)', flexShrink: 0 }}>×</button>
                )}
              </div>
              <div className={styles.modelAttrs}>
                <label className={styles.modelAttrCheck}>
                  <input type="checkbox" checked={m.reasoning ?? false} onChange={(e) => updateModel(idx, { reasoning: e.target.checked || undefined })} style={{ accentColor: 'var(--color-primary)' }} />
                  reasoning
                </label>
                <div className={styles.modelAttrField}>
                  <span className={styles.modelAttrLabel}>effort:</span>
                  <div style={{ minWidth: '90px' }}>
                    <CustomSelect value={m.effort ?? ''} options={['', 'low', 'medium', 'high']} onChange={(v) => updateModel(idx, { effort: v || undefined })} placeholder="—" />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Algorithm */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Algorithm</span>
          {!algorithm && (
            <button className={styles.toolbarBtn} onClick={() => setAlgorithm({ algoType: 'confidence', fields: {} })} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>+ Add</button>
          )}
          {algorithm && (
            <button className={styles.toolbarBtnDanger} onClick={() => setAlgorithm(undefined)} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>Remove</button>
          )}
        </div>
        {algorithm && (
          <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
            <div className={styles.fieldGroup}>
              <label className={styles.fieldLabel}>Algorithm Type</label>
              <CustomSelect value={algorithm.algoType} options={[...ALGORITHM_TYPES]} onChange={(v) => setAlgorithm({ algoType: v, fields: {} })} />
              {ALGORITHM_DESCRIPTIONS[algorithm.algoType] && (
                <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                  {ALGORITHM_DESCRIPTIONS[algorithm.algoType]}
                </span>
              )}
            </div>
            <AlgorithmSchemaEditor algoType={algorithm.algoType} fields={algorithm.fields} onUpdate={(f) => setAlgorithm({ ...algorithm, fields: f })} />
          </div>
        )}
      </div>

      {/* Plugins Toggle Panel */}
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Plugins ({plugins.length})</span>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
          {availablePlugins.length > 0 ? (
            <div className={styles.pluginToggleGrid}>
              {availablePlugins.map(p => {
                const active = activePluginNames.has(p.name)
                return (
                  <button key={p.name} className={active ? styles.pluginToggleActive : styles.pluginToggle} onClick={() => togglePlugin(p.name)}>
                    <span className={styles.pluginToggleCheck}>{active ? '✓' : '○'}</span>
                    <span className={styles.pluginToggleName}>{p.name}</span>
                  </button>
                )
              })}
            </div>
          ) : (
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)' }}>No plugins defined yet.</span>
          )}

          {/* Active plugin configuration editors */}
          {plugins.length > 0 && (
            <div style={{ marginTop: 'var(--spacing-sm)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
              <span className={styles.fieldLabel} style={{ display: 'block' }}>Plugin Configuration</span>
              {plugins.map(p => {
                const tmpl = availablePlugins.find(ap => ap.name === p.name)
                const pType = tmpl?.pluginType ?? p.name
                return (
                  <div key={p.name} className={styles.pluginOverride}>
                    <PluginSchemaEditor
                      pluginType={pType}
                      pluginName={p.name}
                      fields={p.fields ?? {}}
                      onUpdate={(f) => updatePluginFields(p.name, f)}
                      compact
                    />
                  </div>
                )
              })}
            </div>
          )}

          {/* Manual plugin add (for inline plugins not in templates) */}
          <ManualPluginAdder
            existingNames={activePluginNames}
            onAdd={(name) => setPlugins(prev => [...prev, { name }])}
          />
        </div>
      </div>

      {/* DSL Preview with validation */}
      <RouteDslPreviewPanel
        dslText={dslPreview}
        issues={validationIssues}
      />
    </div>
  )
}

// ===================================================================
// Signal Editor Form (dynamic fields by signal type)
// ===================================================================

const SignalEditorForm: React.FC<{
  signal: ASTSignalDecl
  onUpdate: (fields: Record<string, unknown>) => void
}> = ({ signal, onUpdate }) => {
  const schema = useMemo(() => getSignalFieldSchema(signal.signalType), [signal.signalType])
  const [localFields, setLocalFields] = useState<Record<string, unknown>>(() => ({ ...signal.fields }))

  // Sync from parent when signal changes
  useEffect(() => {
    setLocalFields({ ...signal.fields })
  }, [signal.name, signal.signalType, signal.fields])

  const updateField = useCallback((key: string, value: unknown) => {
    setLocalFields(prev => {
      const next = { ...prev, [key]: value }
      return next
    })
  }, [])

  const handleSave = useCallback(() => {
    onUpdate(localFields)
  }, [localFields, onUpdate])

  const dslPreview = useMemo(() =>
    generateSignalDslPreview(signal.signalType, signal.name, localFields),
    [signal.signalType, signal.name, localFields]
  )

  return (
    <>
      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Fields</span>
          <button className={styles.toolbarBtnPrimary} onClick={handleSave} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>
            Save
          </button>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
          {schema.map((field) => (
            <FieldEditor
              key={field.key}
              schema={field}
              value={localFields[field.key]}
              onChange={(v) => updateField(field.key, v)}
            />
          ))}
        </div>
      </div>
      <DslPreviewPanel dslText={dslPreview} />
    </>
  )
}

// ===================================================================
// Plugin Schema Editor (typed fields per plugin type)
// ===================================================================

const PluginSchemaEditor: React.FC<{
  pluginType: string
  pluginName?: string
  fields: Record<string, unknown>
  onUpdate: (fields: Record<string, unknown>) => void
  /** If true, show Save button and buffer changes locally */
  buffered?: boolean
  /** If true, hide the outer container header (compact mode for inline use) */
  compact?: boolean
}> = ({ pluginType, pluginName, fields, onUpdate, buffered = false, compact = false }) => {
  const schema = useMemo(() => getPluginFieldSchema(pluginType), [pluginType])
  const [localFields, setLocalFields] = useState<Record<string, unknown>>(() => ({ ...fields }))

  useEffect(() => {
    setLocalFields({ ...fields })
  }, [fields])

  const currentFields = buffered ? localFields : fields
  const doUpdate = buffered ? setLocalFields : onUpdate

  const updateField = useCallback((key: string, value: unknown) => {
    doUpdate({ ...currentFields, [key]: value })
  }, [currentFields, doUpdate])

  const handleSave = useCallback(() => {
    if (buffered) onUpdate(localFields)
  }, [buffered, localFields, onUpdate])

  if (compact) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)', padding: 'var(--spacing-sm) 0' }}>
        {pluginName && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)' }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--color-primary)', fontWeight: 600 }}>{pluginName}</span>
            <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)' }}>{pluginType}</span>
          </div>
        )}
        {schema.map((field) => (
          <FieldEditor
            key={field.key}
            schema={field}
            value={currentFields[field.key]}
            onChange={(v) => updateField(field.key, v)}
          />
        ))}
        <ExtraFieldsEditor
          fields={currentFields}
          schemaKeys={schema.map(f => f.key)}
          onUpdate={doUpdate}
        />
      </div>
    )
  }

  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>
          {pluginName ?? 'Configuration'}
          {PLUGIN_DESCRIPTIONS[pluginType] && (
            <span style={{ fontWeight: 400, fontSize: '0.625rem', color: 'var(--color-text-muted)', marginLeft: '0.5rem' }}>
              — {PLUGIN_DESCRIPTIONS[pluginType]}
            </span>
          )}
        </span>
        {buffered && (
          <button className={styles.toolbarBtnPrimary} onClick={handleSave} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>
            Save
          </button>
        )}
      </div>
      <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
        {schema.map((field) => (
          <FieldEditor
            key={field.key}
            schema={field}
            value={currentFields[field.key]}
            onChange={(v) => updateField(field.key, v)}
          />
        ))}
        {/* Extra custom fields not in schema */}
        <ExtraFieldsEditor
          fields={currentFields}
          schemaKeys={schema.map(f => f.key)}
          onUpdate={doUpdate}
        />
      </div>
    </div>
  )
}

// ===================================================================
// Global Settings Editor — structured form with 4 sections
// ===================================================================

function getObj(fields: Record<string, unknown>, key: string): Record<string, unknown> {
  const v = fields[key]
  if (v && typeof v === 'object' && !Array.isArray(v)) return v as Record<string, unknown>
  return {}
}

function getBool(obj: Record<string, unknown>, key: string, def = false): boolean {
  const v = obj[key]
  if (typeof v === 'boolean') return v
  return def
}

function getStr(obj: Record<string, unknown>, key: string, def = ''): string {
  const v = obj[key]
  if (typeof v === 'string') return v
  if (typeof v === 'number') return String(v)
  return def
}

function getNum(obj: Record<string, unknown>, key: string, def = 0): number {
  const v = obj[key]
  if (typeof v === 'number') return v
  if (typeof v === 'string') { const n = parseFloat(v); if (!isNaN(n)) return n }
  return def
}

const GlobalSettingsEditor: React.FC<{
  fields: Record<string, unknown>
  onUpdate: (fields: Record<string, unknown>) => void
  endpoints: ASTBackendDecl[]
  onSelectEndpoint: () => void
}> = ({ fields, onUpdate, endpoints }) => {
  const [local, setLocal] = useState<Record<string, unknown>>(() => structuredClone(fields))
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({})

  useEffect(() => {
    setLocal(structuredClone(fields))
  }, [fields])

  const toggleCollapse = useCallback((key: string) => {
    setCollapsedSections(prev => ({ ...prev, [key]: !prev[key] }))
  }, [])

  const setField = useCallback((key: string, value: unknown) => {
    setLocal(prev => ({ ...prev, [key]: value }))
  }, [])

  const setNestedField = useCallback((parentKey: string, childKey: string, value: unknown) => {
    setLocal(prev => {
      const parent = getObj(prev, parentKey)
      return { ...prev, [parentKey]: { ...parent, [childKey]: value } }
    })
  }, [])

  const setDeepField = useCallback((p1: string, p2: string, p3: string, value: unknown) => {
    setLocal(prev => {
      const parent = getObj(prev, p1)
      const child = getObj(parent, p2)
      return { ...prev, [p1]: { ...parent, [p2]: { ...child, [p3]: value } } }
    })
  }, [])

  const handleSave = useCallback(() => {
    onUpdate(local)
  }, [local, onUpdate])

  const dslPreview = useMemo(() => generateGlobalDslPreview(local), [local])

  // --- Extract structured data ---
  const promptGuard = getObj(local, 'prompt_guard')
  const hallucination = getObj(local, 'hallucination_mitigation')
  const observability = getObj(local, 'observability')
  const tracing = getObj(observability, 'tracing')
  const metrics = getObj(observability, 'metrics')
  const authz = getObj(local, 'authz')
  const ratelimit = getObj(local, 'ratelimit')
  const modelSelection = getObj(local, 'model_selection')
  const reasoningFamilies = getObj(local, 'reasoning_families') as Record<string, unknown>
  const looper = getObj(local, 'looper')

  // Collect vllm_endpoint and provider_profile backends
  const vllmEndpoints = endpoints.filter(e => e.backendType === 'vllm_endpoint')
  const providerProfiles = endpoints.filter(e => e.backendType === 'provider_profile')

  return (
    <div className={styles.globalEditor}>
      {/* Save button bar */}
      <div className={styles.globalSaveBar}>
        <span className={styles.globalSaveHint}>Edit global defaults and cross-cutting settings</span>
        <button className={styles.toolbarBtnPrimary} onClick={handleSave} style={{ padding: '0.375rem 1rem', fontSize: 'var(--text-xs)' }}>
          Save
        </button>
      </div>

      {/* ── Section 1: Routing ── */}
      <div className={styles.gsSection}>
        <div className={styles.gsSectionHeader} onClick={() => toggleCollapse('routing')}>
          <svg className={styles.gsSectionChevron} data-open={!collapsedSections['routing']} width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 2l4 3-4 3"/></svg>
          <span className={styles.gsSectionTitle}>Routing</span>
        </div>
        {!collapsedSections['routing'] && (
          <div className={styles.gsSectionBody}>
            <div className={styles.gsRow}>
              <label className={styles.gsLabel}>Default Model</label>
              <input
                className={styles.fieldInput}
                value={getStr(local, 'default_model')}
                onChange={e => setField('default_model', e.target.value)}
                placeholder="qwen2.5:3b"
              />
            </div>
            <div className={styles.gsRow}>
              <label className={styles.gsLabel}>Strategy</label>
              <div className={styles.gsRadioGroup}>
                {['priority', 'confidence'].map(s => (
                  <label key={s} className={styles.gsRadio}>
                    <input type="radio" name="gs-strategy" checked={getStr(local, 'strategy') === s} onChange={() => setField('strategy', s)} />
                    <span>{s}</span>
                  </label>
                ))}
              </div>
            </div>
            <div className={styles.gsRow}>
              <label className={styles.gsLabel}>Default Reasoning Effort</label>
              <div className={styles.gsRadioGroup}>
                {['low', 'medium', 'high'].map(e => (
                  <label key={e} className={styles.gsRadio}>
                    <input type="radio" name="gs-effort" checked={getStr(local, 'default_reasoning_effort') === e} onChange={() => setField('default_reasoning_effort', e)} />
                    <span>{e}</span>
                  </label>
                ))}
              </div>
            </div>
            <div className={styles.gsRow}>
              <label className={styles.gsLabel}>Model Selection</label>
              <div className={styles.gsInlineRow}>
                <label className={styles.gsCheckbox}>
                  <input type="checkbox" checked={getBool(modelSelection, 'enabled')} onChange={e => setNestedField('model_selection', 'enabled', e.target.checked)} />
                  <span>Enabled</span>
                </label>
                {getBool(modelSelection, 'enabled') && (
                  <div className={styles.gsInlineField}>
                    <span className={styles.gsSmallLabel}>Method:</span>
                    <input className={styles.fieldInput} style={{ width: '8rem' }} value={getStr(modelSelection, 'method')} onChange={e => setNestedField('model_selection', 'method', e.target.value)} placeholder="knn" />
                  </div>
                )}
              </div>
            </div>

            {/* Reasoning Families */}
            <div className={styles.gsRow}>
              <label className={styles.gsLabel}>Reasoning Families</label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', width: '100%' }}>
                {Object.entries(reasoningFamilies).map(([name, val]) => {
                  const entry = (val && typeof val === 'object' ? val : {}) as Record<string, unknown>
                  return (
                    <div key={name} style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
                      <input className={styles.fieldInput} style={{ width: '5rem' }} value={name} readOnly title="Family name" />
                      <span className={styles.gsSmallLabel}>type:</span>
                      <input className={styles.fieldInput} style={{ width: '10rem' }} value={getStr(entry, 'type')}
                        onChange={e => {
                          const families = { ...reasoningFamilies }
                          families[name] = { ...entry, type: e.target.value }
                          setField('reasoning_families', families)
                        }}
                        placeholder="chat_template_kwargs"
                      />
                      <span className={styles.gsSmallLabel}>param:</span>
                      <input className={styles.fieldInput} style={{ width: '7rem' }} value={getStr(entry, 'parameter')}
                        onChange={e => {
                          const families = { ...reasoningFamilies }
                          families[name] = { ...entry, parameter: e.target.value }
                          setField('reasoning_families', families)
                        }}
                        placeholder="thinking"
                      />
                      <button className={styles.toolbarBtn} style={{ padding: '0.2rem 0.4rem', fontSize: 'var(--text-xs)' }}
                        onClick={() => {
                          const families = { ...reasoningFamilies }
                          delete families[name]
                          setField('reasoning_families', families)
                        }}
                        title="Remove"
                      >&times;</button>
                    </div>
                  )
                })}
                <button className={styles.toolbarBtn} style={{ alignSelf: 'flex-start', padding: '0.2rem 0.5rem', fontSize: 'var(--text-xs)' }}
                  onClick={() => {
                    const families = { ...reasoningFamilies }
                    const newName = `family_${Object.keys(families).length + 1}`
                    families[newName] = { type: 'chat_template_kwargs', parameter: 'thinking' }
                    setField('reasoning_families', families)
                  }}
                >+ Add Family</button>
              </div>
            </div>

            {/* Looper */}
            <div className={styles.gsRow}>
              <label className={styles.gsLabel}>Looper Endpoint</label>
              <input
                className={styles.fieldInput}
                value={getStr(looper, 'endpoint')}
                onChange={e => setNestedField('looper', 'endpoint', e.target.value)}
                placeholder="http://looper:8080"
              />
            </div>
            {getStr(looper, 'endpoint') && (
              <div className={styles.gsRow}>
                <label className={styles.gsLabel}>Looper Timeout (s)</label>
                <input
                  className={styles.fieldInput}
                  type="number"
                  style={{ width: '6rem' }}
                  value={getNum(looper, 'timeout_seconds', 30)}
                  onChange={e => setNestedField('looper', 'timeout_seconds', parseInt(e.target.value) || 0)}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Section 2: Safety ── */}
      <div className={styles.gsSection}>
        <div className={styles.gsSectionHeader} onClick={() => toggleCollapse('safety')}>
          <svg className={styles.gsSectionChevron} data-open={!collapsedSections['safety']} width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 2l4 3-4 3"/></svg>
          <span className={styles.gsSectionTitle}>Safety</span>
        </div>
        {!collapsedSections['safety'] && (
          <div className={styles.gsSectionBody}>
            {/* Prompt Guard */}
            <div className={styles.gsSubSection}>
              <div className={styles.gsSubHeader}>
                <label className={styles.gsCheckbox}>
                  <input type="checkbox" checked={getBool(promptGuard, 'enabled')}
                    onChange={e => {
                      const cur = getObj(local, 'prompt_guard')
                      if (e.target.checked) {
                        setField('prompt_guard', {
                          ...cur,
                          enabled: true,
                          threshold: getNum(cur, 'threshold', 0.7),
                          model_type: getStr(cur, 'model_type', 'candle'),
                        })
                      } else {
                        setField('prompt_guard', { ...cur, enabled: false })
                      }
                    }}
                  />
                  <span className={styles.gsSubTitle}>Prompt Guard</span>
                </label>
              </div>
              {getBool(promptGuard, 'enabled') && (
                <div className={styles.gsSubBody}>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Threshold</label>
                    <input className={styles.fieldInput} type="number" step="0.1" min="0" max="1" style={{ width: '6rem' }}
                      value={getNum(promptGuard, 'threshold', 0.7)}
                      onChange={e => setNestedField('prompt_guard', 'threshold', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Model</label>
                    <div className={styles.gsRadioGroup}>
                      {[{ label: 'Candle (local)', val: 'candle' }, { label: 'vLLM (external)', val: 'vllm' }].map(o => (
                        <label key={o.val} className={styles.gsRadio}>
                          <input type="radio" name="gs-pg-model" checked={getStr(promptGuard, 'model_type', 'candle') === o.val} onChange={() => setNestedField('prompt_guard', 'model_type', o.val)} />
                          <span>{o.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Hallucination Mitigation */}
            <div className={styles.gsSubSection}>
              <div className={styles.gsSubHeader}>
                <label className={styles.gsCheckbox}>
                  <input type="checkbox" checked={getBool(hallucination, 'enabled')}
                    onChange={e => {
                      const cur = getObj(local, 'hallucination_mitigation')
                      if (e.target.checked) {
                        const fcm = getObj(cur, 'fact_check_model')
                        const hm = getObj(cur, 'hallucination_model')
                        setField('hallucination_mitigation', {
                          ...cur,
                          enabled: true,
                          fact_check_model: { ...fcm, threshold: getNum(fcm, 'threshold', 0.7) },
                          hallucination_model: { ...hm, threshold: getNum(hm, 'threshold', 0.5) },
                          use_nli: getBool(cur, 'use_nli', false),
                        })
                      } else {
                        setField('hallucination_mitigation', { ...cur, enabled: false })
                      }
                    }}
                  />
                  <span className={styles.gsSubTitle}>Hallucination Mitigation</span>
                </label>
              </div>
              {getBool(hallucination, 'enabled') && (
                <div className={styles.gsSubBody}>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Fact-Check Threshold</label>
                    <input className={styles.fieldInput} type="number" step="0.1" min="0" max="1" style={{ width: '6rem' }}
                      value={getNum(getObj(hallucination, 'fact_check_model'), 'threshold', 0.7)}
                      onChange={e => setDeepField('hallucination_mitigation', 'fact_check_model', 'threshold', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Hallucination Threshold</label>
                    <input className={styles.fieldInput} type="number" step="0.1" min="0" max="1" style={{ width: '6rem' }}
                      value={getNum(getObj(hallucination, 'hallucination_model'), 'threshold', 0.5)}
                      onChange={e => setDeepField('hallucination_mitigation', 'hallucination_model', 'threshold', parseFloat(e.target.value) || 0)}
                    />
                  </div>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>NLI Model</label>
                    <label className={styles.gsCheckbox}>
                      <input type="checkbox" checked={getBool(hallucination, 'use_nli', false)} onChange={e => setNestedField('hallucination_mitigation', 'use_nli', e.target.checked)} />
                      <span>Enhanced explanations</span>
                    </label>
                  </div>
                </div>
              )}
            </div>

            {/* Authz */}
            <div className={styles.gsSubSection}>
              <div className={styles.gsSubHeader}>
                <span className={styles.gsSubTitle}>Authorization</span>
              </div>
              <div className={styles.gsSubBody}>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Fail Open</label>
                  <label className={styles.gsCheckbox}>
                    <input type="checkbox" checked={getBool(authz, 'fail_open')} onChange={e => setNestedField('authz', 'fail_open', e.target.checked)} />
                    <span>Allow on auth failure</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Rate Limit */}
            <div className={styles.gsSubSection}>
              <div className={styles.gsSubHeader}>
                <span className={styles.gsSubTitle}>Rate Limit</span>
              </div>
              <div className={styles.gsSubBody}>
                <div className={styles.gsRow}>
                  <label className={styles.gsLabel}>Fail Open</label>
                  <label className={styles.gsCheckbox}>
                    <input type="checkbox" checked={getBool(ratelimit, 'fail_open')} onChange={e => setNestedField('ratelimit', 'fail_open', e.target.checked)} />
                    <span>Allow on rate limit failure</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Section 3: Observability ── */}
      <div className={styles.gsSection}>
        <div className={styles.gsSectionHeader} onClick={() => toggleCollapse('observability')}>
          <svg className={styles.gsSectionChevron} data-open={!collapsedSections['observability']} width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 2l4 3-4 3"/></svg>
          <span className={styles.gsSectionTitle}>Observability</span>
        </div>
        {!collapsedSections['observability'] && (
          <div className={styles.gsSectionBody}>
            {/* Tracing */}
            <div className={styles.gsSubSection}>
              <div className={styles.gsSubHeader}>
                <label className={styles.gsCheckbox}>
                  <input type="checkbox" checked={getBool(tracing, 'enabled')}
                    onChange={e => {
                      const obs = getObj(local, 'observability')
                      const tr = getObj(obs, 'tracing')
                      if (e.target.checked) {
                        const samp = getObj(tr, 'sampling')
                        const exp = getObj(tr, 'exporter')
                        setField('observability', {
                          ...obs,
                          tracing: {
                            ...tr,
                            enabled: true,
                            provider: getStr(tr, 'provider', 'opentelemetry'),
                            exporter: {
                              ...exp,
                              endpoint: getStr(exp, 'endpoint', 'localhost:4317'),
                            },
                            sampling: {
                              ...samp,
                              type: getStr(samp, 'type', 'probabilistic'),
                              rate: getNum(samp, 'rate', 0.1),
                            },
                          },
                        })
                      } else {
                        setField('observability', { ...obs, tracing: { ...tr, enabled: false } })
                      }
                    }}
                  />
                  <span className={styles.gsSubTitle}>Tracing</span>
                </label>
              </div>
              {getBool(tracing, 'enabled') && (
                <div className={styles.gsSubBody}>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Provider</label>
                    <input className={styles.fieldInput} value={getStr(tracing, 'provider', 'opentelemetry')}
                      onChange={e => {
                        const obs = getObj(local, 'observability')
                        const tr = getObj(obs, 'tracing')
                        setField('observability', { ...obs, tracing: { ...tr, provider: e.target.value } })
                      }}
                      placeholder="opentelemetry"
                    />
                  </div>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Endpoint</label>
                    <input className={styles.fieldInput}
                      value={getStr(getObj(tracing, 'exporter'), 'endpoint', '')}
                      onChange={e => {
                        const obs = getObj(local, 'observability')
                        const tr = getObj(obs, 'tracing')
                        const exp = getObj(tr, 'exporter')
                        setField('observability', { ...obs, tracing: { ...tr, exporter: { ...exp, endpoint: e.target.value } } })
                      }}
                      placeholder="localhost:4317"
                    />
                  </div>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Sampling</label>
                    <div className={styles.gsInlineRow}>
                      <input className={styles.fieldInput} style={{ width: '8rem' }}
                        value={getStr(getObj(tracing, 'sampling'), 'type', 'probabilistic')}
                        onChange={e => {
                          const obs = getObj(local, 'observability')
                          const tr = getObj(obs, 'tracing')
                          const samp = getObj(tr, 'sampling')
                          setField('observability', { ...obs, tracing: { ...tr, sampling: { ...samp, type: e.target.value } } })
                        }}
                        placeholder="probabilistic"
                      />
                      <span className={styles.gsSmallLabel}>Rate:</span>
                      <input className={styles.fieldInput} type="number" step="0.1" min="0" max="1" style={{ width: '5rem' }}
                        value={getNum(getObj(tracing, 'sampling'), 'rate', 0.1)}
                        onChange={e => {
                          const obs = getObj(local, 'observability')
                          const tr = getObj(obs, 'tracing')
                          const samp = getObj(tr, 'sampling')
                          setField('observability', { ...obs, tracing: { ...tr, sampling: { ...samp, rate: parseFloat(e.target.value) || 0 } } })
                        }}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Metrics */}
            <div className={styles.gsSubSection}>
              <div className={styles.gsSubHeader}>
                <label className={styles.gsCheckbox}>
                  <input type="checkbox" checked={getBool(metrics, 'enabled', true)}
                    onChange={e => {
                      const obs = getObj(local, 'observability')
                      const met = getObj(obs, 'metrics')
                      setField('observability', { ...obs, metrics: { ...met, enabled: e.target.checked } })
                    }}
                  />
                  <span className={styles.gsSubTitle}>Metrics</span>
                </label>
              </div>
              {getBool(metrics, 'enabled', true) && (
                <div className={styles.gsSubBody}>
                  <div className={styles.gsRow}>
                    <label className={styles.gsLabel}>Windowed Metrics</label>
                    <label className={styles.gsCheckbox}>
                      <input type="checkbox" checked={getBool(metrics, 'windowed', false)}
                        onChange={e => {
                          const obs = getObj(local, 'observability')
                          const met = getObj(obs, 'metrics')
                          setField('observability', { ...obs, metrics: { ...met, windowed: e.target.checked } })
                        }}
                      />
                      <span>Enable windowed aggregation</span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Section 4: Endpoints Summary (read-only) ── */}
      <div className={styles.gsSection}>
        <div className={styles.gsSectionHeader} onClick={() => toggleCollapse('endpoints')}>
          <svg className={styles.gsSectionChevron} data-open={!collapsedSections['endpoints']} width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 2l4 3-4 3"/></svg>
          <span className={styles.gsSectionTitle}>Endpoints</span>
          <span className={styles.gsCountBadge}>{vllmEndpoints.length + providerProfiles.length}</span>
        </div>
        {!collapsedSections['endpoints'] && (
          <div className={styles.gsSectionBody}>
            {vllmEndpoints.length === 0 && providerProfiles.length === 0 ? (
              <div className={styles.gsEmptyHint}>
                No endpoints defined. Add endpoints via <strong>Backends &rarr; vllm_endpoint</strong> or <strong>provider_profile</strong>.
              </div>
            ) : (
              <div className={styles.gsEndpointTable}>
                <div className={styles.gsEndpointHeader}>
                  <span>Name</span>
                  <span>Address</span>
                  <span>Type</span>
                  <span>Weight</span>
                </div>
                {vllmEndpoints.map(ep => (
                  <div key={ep.name} className={styles.gsEndpointRow}>
                    <span className={styles.gsEndpointName}>{ep.name}</span>
                    <span className={styles.gsEndpointAddr}>
                      {getStr(ep.fields, 'address', '—')}:{getStr(ep.fields, 'port', '—')}
                    </span>
                    <span className={styles.gsEndpointType}>{getStr(ep.fields, 'type', 'vllm')}</span>
                    <span>{getStr(ep.fields, 'weight', '1')}</span>
                  </div>
                ))}
                {providerProfiles.map(pp => (
                  <div key={pp.name} className={styles.gsEndpointRow}>
                    <span className={styles.gsEndpointName}>{pp.name}</span>
                    <span className={styles.gsEndpointAddr}>{getStr(pp.fields, 'base_url', '—')}</span>
                    <span className={styles.gsEndpointType}>{getStr(pp.fields, 'type', 'openai')}</span>
                    <span>—</span>
                  </div>
                ))}
              </div>
            )}
            <div className={styles.gsEndpointHint}>
              Endpoints are managed in <strong>Backends</strong>. This is a read-only summary.
            </div>
          </div>
        )}
      </div>

      {/* DSL Preview */}
      <DslPreviewPanel dslText={dslPreview} />
    </div>
  )
}

// ===================================================================
// Generic Fields Editor (for backends — key/value editing)
// ===================================================================

const GenericFieldsEditor: React.FC<{
  fields: Record<string, unknown>
  onUpdate: (fields: Record<string, unknown>) => void
}> = ({ fields, onUpdate }) => {
  const [localFields, setLocalFields] = useState<Record<string, unknown>>(() => ({ ...fields }))
  const [newKey, setNewKey] = useState('')

  useEffect(() => {
    setLocalFields({ ...fields })
  }, [fields])

  const handleSave = useCallback(() => {
    onUpdate(localFields)
  }, [localFields, onUpdate])

  const updateField = useCallback((key: string, rawValue: string) => {
    setLocalFields(prev => {
      const parsed = tryParseValue(rawValue)
      return { ...prev, [key]: parsed }
    })
  }, [])

  const deleteField = useCallback((key: string) => {
    setLocalFields(prev => {
      const next = { ...prev }
      delete next[key]
      return next
    })
  }, [])

  const addField = useCallback(() => {
    const k = newKey.trim()
    if (!k || k in localFields) return
    setLocalFields(prev => ({ ...prev, [k]: '' }))
    setNewKey('')
  }, [newKey, localFields])

  return (
    <div className={styles.dslPreview}>
      <div className={styles.dslPreviewHeader}>
        <span className={styles.dslPreviewTitle}>Fields</span>
        <button className={styles.toolbarBtnPrimary} onClick={handleSave} style={{ padding: '0.25rem 0.5rem', fontSize: 'var(--text-xs)' }}>
          Save
        </button>
      </div>
      <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-sm)' }}>
        {Object.entries(localFields).map(([key, value]) => (
          <div key={key} style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--spacing-sm)' }}>
            <span style={{
              minWidth: '120px', fontSize: 'var(--text-xs)', color: 'var(--color-text-secondary)',
              fontFamily: 'var(--font-mono)', paddingTop: '0.5rem',
            }}>
              {key}
            </span>
            <input
              className={styles.fieldInput}
              style={{ flex: 1, fontSize: 'var(--text-xs)' }}
              value={typeof value === 'string' ? value : JSON.stringify(value)}
              onChange={(e) => updateField(key, e.target.value)}
            />
            <button
              className={styles.toolbarBtnDanger}
              onClick={() => deleteField(key)}
              style={{ padding: '0.375rem', fontSize: 'var(--text-xs)', flexShrink: 0 }}
              title="Remove field"
            >
              ×
            </button>
          </div>
        ))}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-sm)', marginTop: 'var(--spacing-sm)' }}>
          <input
            className={styles.fieldInput}
            style={{ flex: 1, fontSize: 'var(--text-xs)' }}
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            placeholder="New field name..."
            onKeyDown={(e) => e.key === 'Enter' && addField()}
          />
          <button
            className={styles.toolbarBtn}
            onClick={addField}
            disabled={!newKey.trim()}
            style={{ padding: '0.375rem 0.5rem', fontSize: 'var(--text-xs)' }}
          >
            + Add
          </button>
        </div>
      </div>
    </div>
  )
}

// ===================================================================
// Custom Select Dropdown (dark-theme friendly)
// ===================================================================

const CustomSelect: React.FC<{
  value: string
  options: string[]
  onChange: (value: string) => void
  placeholder?: string
}> = ({ value, options, onChange, placeholder = '— select —' }) => {
  const [open, setOpen] = useState(false)
  const triggerRef = useRef<HTMLDivElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<{ top: number; left: number; width: number }>({ top: 0, left: 0, width: 0 })

  // Compute dropdown position when opening
  useLayoutEffect(() => {
    if (!open || !triggerRef.current) return
    const rect = triggerRef.current.getBoundingClientRect()
    setPos({ top: rect.bottom + 4, left: rect.left, width: rect.width })
  }, [open])

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      const target = e.target as Node
      if (triggerRef.current?.contains(target)) return
      if (dropdownRef.current?.contains(target)) return
      setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  // Close on Escape
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [open])

  return (
    <div className={styles.customSelect} ref={triggerRef}>
      <div
        className={open ? styles.customSelectTriggerOpen : styles.customSelectTrigger}
        onClick={() => setOpen(!open)}
      >
        <span>{value || placeholder}</span>
        <svg
          className={`${styles.customSelectChevron} ${open ? styles.customSelectChevronOpen : ''}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path d="M4 6l4 4 4-4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      {open && createPortal(
        <div
          ref={dropdownRef}
          className={styles.customSelectDropdown}
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: pos.width }}
        >
          {options.map((opt) => (
            <div
              key={opt}
              className={opt === value ? styles.customSelectOptionActive : styles.customSelectOption}
              onClick={() => { onChange(opt); setOpen(false) }}
            >
              {opt === value ? (
                <svg className={styles.customSelectCheck} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <span className={styles.customSelectPlaceholder} />
              )}
              {opt || '(none)'}
            </div>
          ))}
        </div>,
        document.body
      )}
    </div>
  )
}

// ===================================================================
// Field Editor — renders a single form field based on schema
// ===================================================================

const FieldEditor: React.FC<{
  schema: FieldSchema
  value: unknown
  onChange: (value: unknown) => void
}> = ({ schema, value, onChange }) => {
  switch (schema.type) {
    case 'string':
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label} {schema.required && <span style={{ color: 'var(--color-danger)' }}>*</span>}
          </label>
          <input
            className={styles.fieldInput}
            value={(value as string) ?? ''}
            onChange={(e) => onChange(e.target.value)}
            placeholder={schema.placeholder}
          />
          {schema.description && (
            <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)' }}>{schema.description}</span>
          )}
        </div>
      )
    case 'number':
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label} {schema.required && <span style={{ color: 'var(--color-danger)' }}>*</span>}
          </label>
          <input
            className={styles.fieldInput}
            type="number"
            step="any"
            value={value !== undefined && value !== null ? String(value) : ''}
            onChange={(e) => {
              const v = e.target.value
              onChange(v === '' ? undefined : Number(v))
            }}
            placeholder={schema.placeholder}
          />
        </div>
      )
    case 'boolean':
      return (
        <div className={styles.fieldGroup}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={!!value}
              onChange={(e) => onChange(e.target.checked)}
              style={{ accentColor: 'var(--color-primary)' }}
            />
            <span className={styles.fieldLabel} style={{ textTransform: 'none' }}>{schema.label}</span>
          </label>
        </div>
      )
    case 'select':
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label} {schema.required && <span style={{ color: 'var(--color-danger)' }}>*</span>}
          </label>
          <CustomSelect
            value={(value as string) ?? ''}
            options={schema.options ?? []}
            onChange={(v) => onChange(v || undefined)}
            placeholder="— select —"
          />
        </div>
      )
    case 'string[]':
      return (
        <StringArrayEditor
          label={schema.label}
          required={schema.required}
          value={(value as string[]) ?? []}
          onChange={onChange}
          placeholder={schema.placeholder}
        />
      )
    case 'number[]': {
      // Handle: number[], or JSON string like "[32, 4]"
      let arr: number[] = []
      if (Array.isArray(value)) {
        arr = value.map(Number).filter(n => !isNaN(n))
      } else if (typeof value === 'string') {
        try { const parsed = JSON.parse(value); if (Array.isArray(parsed)) arr = parsed.map(Number).filter(n => !isNaN(n)) } catch { /* ignore */ }
      }
      return (
        <NumberArrayEditor
          label={schema.label}
          required={schema.required}
          value={arr}
          onChange={onChange}
          placeholder={schema.placeholder}
          description={schema.description}
        />
      )
    }
    case 'json':
      return (
        <div className={styles.fieldGroup}>
          <label className={styles.fieldLabel}>
            {schema.label} {schema.required && <span style={{ color: 'var(--color-danger)' }}>*</span>}
          </label>
          <textarea
            className={styles.fieldTextarea}
            value={value !== undefined && value !== null ? (typeof value === 'string' ? value : JSON.stringify(value, null, 2)) : ''}
            onChange={(e) => {
              try {
                onChange(JSON.parse(e.target.value))
              } catch {
                onChange(e.target.value)
              }
            }}
            rows={3}
            style={{ fontSize: 'var(--text-xs)' }}
          />
          {schema.description && (
            <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)' }}>{schema.description}</span>
          )}
        </div>
      )
    default:
      return null
  }
}

// ===================================================================
// String Array Editor (for keywords, candidates, etc.)
// ===================================================================

const StringArrayEditor: React.FC<{
  label: string
  required?: boolean
  value: string[]
  onChange: (value: string[]) => void
  placeholder?: string
}> = ({ label, required, value, onChange, placeholder }) => {
  const [inputValue, setInputValue] = useState('')

  const addItem = useCallback(() => {
    const v = inputValue.trim()
    if (v && !value.includes(v)) {
      onChange([...value, v])
      setInputValue('')
    }
  }, [inputValue, value, onChange])

  const removeItem = useCallback((idx: number) => {
    onChange(value.filter((_, i) => i !== idx))
  }, [value, onChange])

  return (
    <div className={styles.fieldGroup}>
      <label className={styles.fieldLabel}>
        {label} {required && <span style={{ color: 'var(--color-danger)' }}>*</span>}
      </label>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', minHeight: '1.5rem' }}>
        {value.map((item, idx) => (
          <span
            key={idx}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
              padding: '0.125rem 0.5rem', fontSize: 'var(--text-xs)',
              background: 'var(--color-bg-tertiary)', border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)',
              color: 'var(--color-text)',
            }}
          >
            {item}
            <button
              onClick={() => removeItem(idx)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                color: 'var(--color-text-muted)', fontSize: '0.75rem', lineHeight: 1,
              }}
            >
              ×
            </button>
          </span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 'var(--spacing-sm)' }}>
        <input
          className={styles.fieldInput}
          style={{ flex: 1 }}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={placeholder}
          onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addItem())}
        />
        <button
          className={styles.toolbarBtn}
          onClick={addItem}
          disabled={!inputValue.trim()}
          style={{ padding: '0.375rem 0.5rem', fontSize: 'var(--text-xs)' }}
        >
          + Add
        </button>
      </div>
    </div>
  )
}

// ===================================================================
// Number Array Editor (for breadth_schedule, etc.)
// ===================================================================

const NumberArrayEditor: React.FC<{
  label: string
  required?: boolean
  value: number[]
  onChange: (value: number[]) => void
  placeholder?: string
  description?: string
}> = ({ label, required, value, onChange, placeholder, description }) => {
  const [inputValue, setInputValue] = useState('')

  const addItem = useCallback(() => {
    const v = inputValue.trim()
    if (v === '') return
    const num = Number(v)
    if (isNaN(num)) return
    onChange([...value, num])
    setInputValue('')
  }, [inputValue, value, onChange])

  const removeItem = useCallback((idx: number) => {
    onChange(value.filter((_, i) => i !== idx))
  }, [value, onChange])

  return (
    <div className={styles.fieldGroup}>
      <label className={styles.fieldLabel}>
        {label} {required && <span style={{ color: 'var(--color-danger)' }}>*</span>}
      </label>
      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.25rem', minHeight: '1.75rem' }}>
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>[</span>
        {value.map((item, idx) => (
          <span
            key={idx}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.25rem',
              padding: '0.125rem 0.5rem', fontSize: 'var(--text-xs)',
              background: 'var(--color-bg-tertiary)', border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-mono)',
              color: 'var(--color-text)',
            }}
          >
            {item}
            <button
              onClick={() => removeItem(idx)}
              style={{
                background: 'none', border: 'none', cursor: 'pointer', padding: 0,
                color: 'var(--color-text-muted)', fontSize: '0.75rem', lineHeight: 1,
              }}
            >
              ×
            </button>
          </span>
        ))}
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--color-text-muted)', fontFamily: 'var(--font-mono)' }}>]</span>
      </div>
      <div style={{ display: 'flex', gap: 'var(--spacing-sm)' }}>
        <input
          className={styles.fieldInput}
          style={{ flex: 1 }}
          type="number"
          step="any"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder={placeholder}
          onKeyDown={(e) => e.key === 'Enter' && (e.preventDefault(), addItem())}
        />
        <button
          className={styles.toolbarBtn}
          onClick={addItem}
          disabled={!inputValue.trim() || isNaN(Number(inputValue))}
          style={{ padding: '0.375rem 0.5rem', fontSize: 'var(--text-xs)' }}
        >
          + Add
        </button>
      </div>
      {description && (
        <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)' }}>{description}</span>
      )}
    </div>
  )
}

// ===================================================================
// Add Signal Form
// ===================================================================

const AddSignalForm: React.FC<{
  onAdd: (signalType: string, name: string, fields: Record<string, unknown>) => void
  onCancel: () => void
}> = ({ onAdd, onCancel }) => {
  const [signalType, setSignalType] = useState<SignalType>('domain')
  const [name, setName] = useState('')
  const schema = useMemo(() => getSignalFieldSchema(signalType), [signalType])
  const [fields, setFields] = useState<Record<string, unknown>>({})

  // Reset fields when signal type changes
  useEffect(() => {
    setFields({})
  }, [signalType])

  const updateField = useCallback((key: string, value: unknown) => {
    setFields(prev => ({ ...prev, [key]: value }))
  }, [])

  const handleSubmit = useCallback(() => {
    const n = name.trim().replace(/\s+/g, '_')
    if (!n) return
    onAdd(signalType, n, fields)
  }, [signalType, name, fields, onAdd])

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <SignalIcon className={styles.statIcon} />
          New Signal
        </div>
        <div className={styles.editorActions}>
          <button className={styles.toolbarBtn} onClick={onCancel}>Cancel</button>
          <button
            className={styles.toolbarBtnPrimary}
            onClick={handleSubmit}
            disabled={!name.trim()}
          >
            Create
          </button>
        </div>
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>Signal Type <span style={{ color: 'var(--color-danger)' }}>*</span></label>
        <CustomSelect
          value={signalType}
          options={[...SIGNAL_TYPES]}
          onChange={(v) => setSignalType(v as SignalType)}
        />
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>Name <span style={{ color: 'var(--color-danger)' }}>*</span></label>
        <input
          className={styles.fieldInput}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my_signal_name"
          autoFocus
        />
      </div>

      <div className={styles.dslPreview}>
        <div className={styles.dslPreviewHeader}>
          <span className={styles.dslPreviewTitle}>Fields</span>
        </div>
        <div style={{ padding: 'var(--spacing-md)', display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
          {schema.map((field) => (
            <FieldEditor
              key={field.key}
              schema={field}
              value={fields[field.key]}
              onChange={(v) => updateField(field.key, v)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

// ===================================================================
// Add Plugin Form
// ===================================================================

const AddPluginForm: React.FC<{
  onAdd: (name: string, pluginType: string, fields: Record<string, unknown>) => void
  onCancel: () => void
}> = ({ onAdd, onCancel }) => {
  const [pluginType, setPluginType] = useState<string>(PLUGIN_TYPES[0])
  const [name, setName] = useState('')
  const [fields, setFields] = useState<Record<string, unknown>>({ enabled: true })

  // Reset fields when plugin type changes
  const handlePluginTypeChange = useCallback((v: string) => {
    setPluginType(v)
    setFields({ enabled: true })
  }, [])

  const handleSubmit = useCallback(() => {
    const n = name.trim().replace(/\s+/g, '_')
    if (!n) return
    onAdd(n, pluginType, fields)
  }, [name, pluginType, fields, onAdd])

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <PluginIcon className={styles.statIcon} />
          New Plugin
        </div>
        <div className={styles.editorActions}>
          <button className={styles.toolbarBtn} onClick={onCancel}>Cancel</button>
          <button className={styles.toolbarBtnPrimary} onClick={handleSubmit} disabled={!name.trim()}>Create</button>
        </div>
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>Plugin Type <span style={{ color: 'var(--color-danger)' }}>*</span></label>
        <CustomSelect
          value={pluginType}
          options={[...PLUGIN_TYPES]}
          onChange={handlePluginTypeChange}
        />
        {PLUGIN_DESCRIPTIONS[pluginType] && (
          <span style={{ fontSize: '0.625rem', color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
            {PLUGIN_DESCRIPTIONS[pluginType]}
          </span>
        )}
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>Name <span style={{ color: 'var(--color-danger)' }}>*</span></label>
        <input className={styles.fieldInput} value={name} onChange={(e) => setName(e.target.value)} placeholder="my_plugin" autoFocus />
      </div>

      <PluginSchemaEditor pluginType={pluginType} fields={fields} onUpdate={setFields} />
    </div>
  )
}

// ===================================================================
// Add Backend Form
// ===================================================================

const AddBackendForm: React.FC<{
  onAdd: (backendType: string, name: string, fields: Record<string, unknown>) => void
  onCancel: () => void
}> = ({ onAdd, onCancel }) => {
  const [backendType, setBackendType] = useState<string>(BACKEND_TYPES[0])
  const [name, setName] = useState('')
  const [fields, setFields] = useState<Record<string, unknown>>({})

  const handleSubmit = useCallback(() => {
    const n = name.trim().replace(/\s+/g, '_')
    if (!n) return
    onAdd(backendType, n, fields)
  }, [name, backendType, fields, onAdd])

  return (
    <div className={styles.editorPanel}>
      <div className={styles.editorHeader}>
        <div className={styles.editorTitle}>
          <BackendIcon className={styles.statIcon} />
          New Backend
        </div>
        <div className={styles.editorActions}>
          <button className={styles.toolbarBtn} onClick={onCancel}>Cancel</button>
          <button className={styles.toolbarBtnPrimary} onClick={handleSubmit} disabled={!name.trim()}>Create</button>
        </div>
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>Backend Type <span style={{ color: 'var(--color-danger)' }}>*</span></label>
        <CustomSelect
          value={backendType}
          options={[...BACKEND_TYPES]}
          onChange={(v) => setBackendType(v)}
        />
      </div>

      <div className={styles.fieldGroup}>
        <label className={styles.fieldLabel}>Name <span style={{ color: 'var(--color-danger)' }}>*</span></label>
        <input className={styles.fieldInput} value={name} onChange={(e) => setName(e.target.value)} placeholder="my_backend" autoFocus />
      </div>

      <GenericFieldsEditor fields={fields} onUpdate={setFields} />
    </div>
  )
}

// ===================================================================
// Helpers
// ===================================================================

function tryParseValue(raw: string): unknown {
  const trimmed = raw.trim()
  if (trimmed === 'true') return true
  if (trimmed === 'false') return false
  if (trimmed === '') return ''
  if (/^-?\d+$/.test(trimmed)) return parseInt(trimmed, 10)
  if (/^-?\d+\.\d+$/.test(trimmed)) return parseFloat(trimmed)
  try {
    const parsed = JSON.parse(trimmed)
    if (typeof parsed === 'object') return parsed
  } catch { /* not JSON */ }
  return raw
}

// ===================================================================
// SVG Icons
// ===================================================================

const SignalIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M2 12V8M5 12V6M8 12V4M11 12V7M14 12V2" strokeLinecap="round" />
  </svg>
)

const RouteIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <path d="M2 8h4l2-4h6M8 8l2 4h4" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

const PluginIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="3" y="1" width="10" height="14" rx="2" />
    <path d="M6 5h4M6 8h4M6 11h2" strokeLinecap="round" />
  </svg>
)

const BackendIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="2" y="2" width="12" height="4" rx="1" />
    <rect x="2" y="10" width="12" height="4" rx="1" />
    <path d="M8 6v4" strokeLinecap="round" />
  </svg>
)

const GlobalIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="8" cy="8" r="6" />
    <path d="M2 8h12M8 2c-2 2-2 10 0 12M8 2c2 2 2 10 0 12" />
  </svg>
)

export default BuilderPage
