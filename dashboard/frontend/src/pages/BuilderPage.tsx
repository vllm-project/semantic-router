import React, { useEffect, useCallback, useState, useMemo, useRef } from 'react'

import { selectHasUnsavedChanges, useDSLStore } from '@/stores/dslStore'
import type { EditorMode } from '@/types/dsl'

import styles from './BuilderPage.module.css'
const DslEditorPage = React.lazy(() => import('./DslEditorPage'))
import {
  BuilderDeployConfirmModal,
  BuilderDeployToast,
  BuilderDragOverlay,
} from './builderPageDeployOverlays'
import { VisualMode } from './builderPageVisualShell'
import { BuilderGuideDrawer } from './builderPageGuideDrawer'
import { BuilderImportModal } from './builderPageImportModal'
import { BuilderOutputPanel } from './builderPageOutputPanel'
import { useResizableWidth } from './builderPageResizeHooks'
import { BuilderStatusBar } from './builderPageStatusBar'
import { BuilderToolbar } from './builderPageToolbar'
import {
  chooseDefaultBuilderRoutingScope,
  listBuilderRoutingScopes,
  resolveBuilderRoutingScope,
  summarizeBuilderRoutingScopes,
} from './builderPageRoutingScopeSupport'
import { useBuilderScopedEntityMutations } from './useBuilderScopedEntityMutations'
import { useReadonly } from '@/contexts/ReadonlyContext'
import { useAuth } from '@/contexts/AuthContext'
import { canDeployConfig } from '@/utils/accessControl'
import type { EntityKind, SectionState, Selection } from './builderPageTypes'

// ---------- Component ----------

const BuilderPage: React.FC = () => {
  const {
    dslSource,
    diagnostics,
    symbols,
    ast,
    compilerReady,
    compilerError,
    loading,
    mode,
    renderedYamlOutput,
    yamlOutput,
    crdOutput,
    compileError,
    initCompiler,
    pauseEditorWork,
    compile,
    validate,
    parseAST,
    format,
    reset,
    setMode,
    importYaml,
    loadFromRouter,
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

  // Derived from the store: the source differs from the last load, import, reset, or
  // successful deploy snapshot. The reload guard and the (unsaved) label read this.
  const unsaved = useDSLStore(selectHasUnsavedChanges)

  const { serverReadonly, runtimeConfigWritable, isLoading: readonlyLoading } = useReadonly()
  const { user } = useAuth()
  const hasDeployPermission = canDeployConfig(user)

  const [selection, setSelection] = useState<Selection | null>(null)
  const [sections, setSections] = useState<SectionState>({
    models: true,
    signals: true,
    projectionPartitions: true,
    projectionScores: true,
    projectionMappings: true,
    routes: true,
    plugins: true,
  })
  const [addingEntity, setAddingEntity] = useState<EntityKind | null>(null)
  const [activeRoutingScopeId, setActiveRoutingScopeId] = useState('__auto__')
  const [outputPanelOpen, setOutputPanelOpen] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [guideOpen, setGuideOpen] = useState(false)

  const contentRef = useRef<HTMLDivElement>(null)
  const {
    width: guideWidth,
    isDragging: isGuideDragging,
    handleDragStart: handleGuideDragStart,
  } = useResizableWidth({
    initialWidth: 420,
    minWidth: 300,
    getMaxWidth: () => 800,
    stopPropagation: true,
  })
  const {
    width: outputWidth,
    isDragging,
    handleDragStart,
  } = useResizableWidth({
    initialWidth: 320,
    minWidth: 260,
    getMaxWidth: () => Math.floor((contentRef.current?.offsetWidth ?? window.innerWidth) * 0.42),
  })
  const [importText, setImportText] = useState('')
  const [importError, setImportError] = useState<string | null>(null)
  const [configLoadError, setConfigLoadError] = useState<string | null>(null)
  const [importUrl, setImportUrl] = useState('')
  const [importUrlLoading, setImportUrlLoading] = useState(false)
  const importTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const autoLoadedDefaultConfigRef = useRef(false)
  const autoLoadingDefaultConfigRef = useRef(false)
  const routingScopes = useMemo(() => listBuilderRoutingScopes(ast), [ast])
  const activeRoutingScope = useMemo(
    () => routingScopes.find((scope) => scope.id === activeRoutingScopeId) ?? null,
    [activeRoutingScopeId, routingScopes],
  )
  const visualAst = useMemo(
    () => resolveBuilderRoutingScope(ast, activeRoutingScopeId) ?? ast,
    [activeRoutingScopeId, ast],
  )

  useEffect(() => {
    if (!ast || routingScopes.length === 0) return
    if (!routingScopes.some((scope) => scope.id === activeRoutingScopeId)) {
      setActiveRoutingScopeId(chooseDefaultBuilderRoutingScope(ast))
      setSelection(null)
      setAddingEntity(null)
    }
  }, [activeRoutingScopeId, ast, routingScopes])

  // Check compiler availability on mount
  useEffect(() => {
    initCompiler()
    return () => pauseEditorWork()
  }, [initCompiler, pauseEditorWork])

  // The visual workspace is the primary authoring entrypoint. DSL remains one
  // click away for precision edits and round-trip inspection.
  useEffect(() => {
    setMode('visual')
  }, [setMode])

  // Refresh when entering visual mode; edits are analyzed by the store.
  useEffect(() => {
    if (mode === 'visual' && compilerReady && useDSLStore.getState().dslSource.trim()) {
      parseAST()
    }
  }, [mode, compilerReady, parseAST])

  const toggleSection = useCallback((key: keyof SectionState) => {
    setSections((prev) => ({ ...prev, [key]: !prev[key] }))
  }, [])

  const handleModeSwitch = useCallback(
    (newMode: EditorMode) => {
      setMode(newMode)
      setOutputPanelOpen(true)
    },
    [setMode],
  )
  const deployDisabled =
    readonlyLoading || serverReadonly || !runtimeConfigWritable || !hasDeployPermission

  const {
    handleAddModel,
    handleAddPlugin,
    handleAddProjectionMapping,
    handleAddProjectionPartition,
    handleAddProjectionScore,
    handleAddRoute,
    handleAddSignal,
    handleDeleteEntity,
    handleUpdateModelFields,
    handleUpdatePluginFields,
    handleUpdateProjectionMappingFields,
    handleUpdateProjectionPartitionFields,
    handleUpdateProjectionScoreFields,
    handleUpdateRoute,
    handleUpdateSignalFields,
  } = useBuilderScopedEntityMutations({
    recipeName: activeRoutingScope?.recipeName ?? null,
    setAddingEntity,
    setSelection,
  })

  // --- Import Config handlers ---

  const handleOpenImport = useCallback(() => {
    setImportText('')
    setImportError(null)
    setImportUrl('')
    setImportUrlLoading(false)
    setShowImportModal(true)
    setTimeout(() => importTextareaRef.current?.focus(), 50)
  }, [])

  const handleImportConfirm = useCallback(async () => {
    const yaml = importText.trim()
    if (!yaml) {
      setImportError('Please paste YAML content')
      return
    }
    try {
      await importYaml(yaml)
      await compile()
      setShowImportModal(false)
      setImportText('')
      setImportError(null)
      setConfigLoadError(null)
      autoLoadedDefaultConfigRef.current = true
    } catch (err) {
      setImportError(`Failed to import YAML: ${err instanceof Error ? err.message : String(err)}`)
    }
  }, [importText, importYaml, compile])

  const handleImportFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = ev.target?.result
      if (typeof text === 'string') {
        setImportText(text)
        setImportError(null)
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [])

  const handleImportUrl = useCallback(async () => {
    const url = importUrl.trim()
    if (!url) {
      setImportError('Please enter a URL')
      return
    }
    try {
      new URL(url)
    } catch {
      setImportError('Invalid URL format')
      return
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
      await compile()
      setShowImportModal(false)
      setImportText('')
      setConfigLoadError(null)
      autoLoadedDefaultConfigRef.current = true
    } catch (err) {
      setImportError(
        `Failed to load from router: ${err instanceof Error ? err.message : String(err)}`,
      )
    } finally {
      setLoadingFromRouter(false)
    }
  }, [loadFromRouter, compile])

  const handleRequestDeploy = useCallback(() => {
    if (deployDisabled) {
      return
    }
    requestDeploy()
  }, [deployDisabled, requestDeploy])

  // On first entry, load current router config and compile it by default.
  useEffect(() => {
    if (
      !compilerReady ||
      readonlyLoading ||
      dslSource.trim() ||
      autoLoadedDefaultConfigRef.current ||
      autoLoadingDefaultConfigRef.current
    ) {
      return
    }

    autoLoadingDefaultConfigRef.current = true
    let cancelled = false
    const loadDefaultConfig = async () => {
      setLoadingFromRouter(true)
      setConfigLoadError(null)
      try {
        await loadFromRouter()
        if (!cancelled) {
          await compile()
          autoLoadedDefaultConfigRef.current = true
        }
      } catch (err) {
        if (!cancelled) {
          setConfigLoadError(
            `Failed to load router config: ${err instanceof Error ? err.message : String(err)}`,
          )
        }
      } finally {
        autoLoadingDefaultConfigRef.current = false
        if (!cancelled) {
          setLoadingFromRouter(false)
        }
      }
    }
    void loadDefaultConfig()
    return () => {
      cancelled = true
    }
  }, [compilerReady, readonlyLoading, dslSource, loadFromRouter, compile])

  // Diagnostic counts
  const validationErrorCount = diagnostics.filter((d) => d.level === 'error').length
  const errorCount = validationErrorCount + (compileError ? 1 : 0) + (configLoadError ? 1 : 0)
  const modelCount = ast?.models?.length ?? symbols?.models?.length ?? 0
  const totalRoutingSummary = useMemo(
    () => summarizeBuilderRoutingScopes(ast, symbols, dslSource),
    [ast, symbols, dslSource],
  )
  const {
    signalCount,
    projectionPartitionCount,
    projectionScoreCount,
    projectionMappingCount,
    routeCount,
    pluginCount,
  } = useMemo(() => summarizeBuilderRoutingScopes(visualAst, null), [visualAst])
  const { recipeCount, entrypointCount } = totalRoutingSummary
  const isValid = errorCount === 0 && compilerReady
  const lineCount = dslSource.split('\n').length

  // Memoize selected entity from AST
  const selectedEntity = useMemo(() => {
    if (!selection || !visualAst) return null
    switch (selection.kind) {
      case 'model':
        return visualAst.models?.find((m) => m.name === selection.name) ?? null
      case 'signal':
        return visualAst.signals?.find((s) => s.name === selection.name) ?? null
      case 'projection-partition':
        return (
          visualAst.projectionPartitions?.find((partition) => partition.name === selection.name) ??
          null
        )
      case 'projection-score':
        return visualAst.projectionScores?.find((score) => score.name === selection.name) ?? null
      case 'projection-mapping':
        return (
          visualAst.projectionMappings?.find((mapping) => mapping.name === selection.name) ?? null
        )
      case 'route':
        return visualAst.routes?.find((r) => r.name === selection.name) ?? null
      case 'plugin':
        return visualAst.plugins?.find((p) => p.name === selection.name) ?? null
      default:
        return null
    }
  }, [selection, visualAst])

  return (
    <div className={styles.page}>
      <BuilderToolbar
        unsaved={unsaved}
        mode={mode}
        compilerReady={compilerReady}
        compilerError={compilerError}
        dslSource={dslSource}
        loading={loading}
        deploying={deploying}
        deployDisabled={deployDisabled}
        deployDisabledReason={
          readonlyLoading
            ? 'Checking deploy permissions...'
            : serverReadonly
              ? 'Deploy is disabled by the server-wide read-only policy'
              : !runtimeConfigWritable
                ? 'Deploy requires a writable runtime configuration mount'
                : !hasDeployPermission
                  ? 'Deploy requires the config.deploy permission'
                  : undefined
        }
        showBuilderSecondaryActions
        guideOpen={guideOpen}
        outputPanelOpen={outputPanelOpen}
        onModeSwitch={handleModeSwitch}
        onImport={handleOpenImport}
        onCompile={compile}
        onRequestDeploy={handleRequestDeploy}
        onFormat={format}
        onValidate={validate}
        onToggleGuide={() => setGuideOpen(!guideOpen)}
        onToggleOutput={() => setOutputPanelOpen(!outputPanelOpen)}
        onReset={() => {
          setConfigLoadError(null)
          autoLoadedDefaultConfigRef.current = true
          reset()
        }}
      />

      {configLoadError && (
        <div className={styles.workspaceError} role="alert">
          {configLoadError} Use Import to retry loading or choose a corrected config.
        </div>
      )}
      {compileError && (
        <div className={styles.workspaceError} role="alert">
          {compileError}
        </div>
      )}

      {/* Main Content — editor + output panel */}
      <div className={styles.content} ref={contentRef}>
        {/* Editor area (switches by mode) */}
        <div className={styles.editorArea}>
          {mode === 'visual' && (
            <VisualMode
              ast={visualAst}
              dslSource={dslSource}
              diagnostics={diagnostics}
              selection={selection}
              onSelect={setSelection}
              sections={sections}
              onToggleSection={toggleSection}
              selectedEntity={selectedEntity}
              modelCount={modelCount}
              signalCount={signalCount}
              projectionPartitionCount={projectionPartitionCount}
              projectionScoreCount={projectionScoreCount}
              projectionMappingCount={projectionMappingCount}
              routeCount={routeCount}
              pluginCount={pluginCount}
              compilerReady={compilerReady}
              compilerError={compilerError}
              addingEntity={addingEntity}
              onSetAddingEntity={setAddingEntity}
              onDeleteEntity={handleDeleteEntity}
              onUpdateModelFields={handleUpdateModelFields}
              onUpdateSignalFields={handleUpdateSignalFields}
              onUpdateProjectionPartitionFields={handleUpdateProjectionPartitionFields}
              onUpdateProjectionScoreFields={handleUpdateProjectionScoreFields}
              onUpdateProjectionMappingFields={handleUpdateProjectionMappingFields}
              onUpdatePluginFields={handleUpdatePluginFields}
              onAddModel={handleAddModel}
              onAddSignal={handleAddSignal}
              onAddProjectionPartition={handleAddProjectionPartition}
              onAddProjectionScore={handleAddProjectionScore}
              onAddProjectionMapping={handleAddProjectionMapping}
              onAddPlugin={handleAddPlugin}
              onUpdateRoute={handleUpdateRoute}
              onAddRoute={handleAddRoute}
              errorCount={errorCount}
              isValid={isValid}
              onModeSwitch={handleModeSwitch}
              routingScopes={routingScopes}
              activeRoutingScopeId={activeRoutingScopeId}
              onRoutingScopeChange={(scopeId) => {
                setActiveRoutingScopeId(scopeId)
                setSelection(null)
                setAddingEntity(null)
              }}
            />
          )}
          {mode === 'dsl' && (
            <div className={styles.dslModeContainer}>
              <React.Suspense fallback={<div role="status">Loading DSL editor…</div>}>
                <DslEditorPage embedded hideOutput />
              </React.Suspense>
            </div>
          )}
        </div>

        <BuilderOutputPanel
          open={outputPanelOpen}
          width={outputWidth}
          yamlOutput={renderedYamlOutput || yamlOutput}
          crdOutput={crdOutput}
          dslSource={dslSource}
          dslTabLabel="DSL"
          compileError={compileError}
          onDragStart={handleDragStart}
          onOpen={() => setOutputPanelOpen(true)}
          onClose={() => setOutputPanelOpen(false)}
        />
      </div>

      <BuilderStatusBar
        isValid={isValid}
        errorCount={errorCount}
        modelCount={modelCount}
        signalCount={signalCount}
        routeCount={routeCount}
        pluginCount={pluginCount}
        recipeCount={recipeCount}
        entrypointCount={entrypointCount}
        lineCount={lineCount}
        mode={mode}
      />

      {/* Hidden file input for YAML import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml,.json"
        style={{ display: 'none' }}
        onChange={handleImportFile}
      />

      <BuilderImportModal
        open={showImportModal}
        importUrl={importUrl}
        importText={importText}
        importError={importError}
        importUrlLoading={importUrlLoading}
        loadingFromRouter={loadingFromRouter}
        importTextareaRef={importTextareaRef}
        onClose={() => setShowImportModal(false)}
        onImportUrlChange={(value) => {
          setImportUrl(value)
          setImportError(null)
        }}
        onImportTextChange={(value) => {
          setImportText(value)
          setImportError(null)
        }}
        onImportUrl={handleImportUrl}
        onSelectFile={() => fileInputRef.current?.click()}
        onLoadFromRouter={handleLoadFromRouter}
        onConfirm={handleImportConfirm}
      />

      <BuilderGuideDrawer
        open={guideOpen}
        width={guideWidth}
        isDragging={isGuideDragging}
        onClose={() => setGuideOpen(false)}
        onDragStart={handleGuideDragStart}
        onInsertSnippet={(snippet) => {
          if (mode !== 'dsl') setMode('dsl')
          const store = useDSLStore.getState()
          const src = store.dslSource
          store.setDslSource(src ? src.trimEnd() + '\n\n' + snippet + '\n' : snippet + '\n')
          setGuideOpen(false)
        }}
      />

      <BuilderDeployConfirmModal
        open={showDeployConfirm}
        loading={deployPreviewLoading}
        error={deployPreviewError}
        currentYaml={deployPreviewCurrent}
        mergedYaml={deployPreviewMerged}
        onClose={dismissDeploy}
        onConfirm={executeDeploy}
      />

      <BuilderDeployToast
        deploying={deploying}
        deployStep={deployStep}
        deployResult={deployResult}
        onDismiss={dismissDeploy}
      />

      <BuilderDragOverlay active={isDragging || isGuideDragging} />
    </div>
  )
}

export default BuilderPage
