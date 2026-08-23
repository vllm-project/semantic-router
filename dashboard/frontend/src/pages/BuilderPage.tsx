import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import ConfirmDialog from '@/components/ConfirmDialog'
import EditModal, { type EditFormData, type FieldConfig } from '@/components/EditModal'
import ProductIcon from '@/components/ProductIcon'
import { useAuth } from '@/contexts/AuthContext'
import { useDSLStore } from '@/stores/dslStore'
import type { EditorMode } from '@/types/dsl'
import { canManageRouting, canReadRouting } from '@/utils/accessControl'

import styles from './BuilderPage.module.css'
import ConfigPageRoutingScopeState from './ConfigPageRoutingScopeState'
import DslEditorPage from './DslEditorPage'
import {
  compileBuilderRecipe,
  loadManagedRecipeSource,
  projectImportedRecipe,
} from './builderRecipeClient'
import { BuilderGuideDrawer } from './builderPageGuideDrawer'
import { BuilderImportModal } from './builderPageImportModal'
import { BuilderOutputPanel } from './builderPageOutputPanel'
import { useResizableWidth } from './builderPageResizeHooks'
import { mutateBuilderRecipeSource } from './builderPageRoutingScopeSupport'
import { BuilderStatusBar } from './builderPageStatusBar'
import { BuilderToolbar } from './builderPageToolbar'
import type { EntityKind, SectionState, Selection } from './builderPageTypes'
import { VisualMode } from './builderPageVisualShell'
import { useBuilderRecipeClient } from './useBuilderRecipeClient'
import { useBuilderScopedEntityMutations } from './useBuilderScopedEntityMutations'

const DUPLICATE_FIELDS: FieldConfig[] = [
  { name: 'name', label: 'Name', type: 'text', required: true, placeholder: 'My Recipe' },
  {
    name: 'description',
    label: 'Description',
    type: 'textarea',
    placeholder: 'What this Recipe optimizes for',
  },
]

export default function BuilderPage() {
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
    compileError,
    initWasm,
    compile,
    validate,
    parseAST,
    format,
    setMode,
    loadDsl,
  } = useDSLStore()
  const { user } = useAuth()
  const canRead = canReadRouting(user)
  const canManage = canManageRouting(user)
  const client = useBuilderRecipeClient(wasmReady && canRead)
  const selectedRecipe = client.selectedRecipe
  const editable = canManage && Boolean(selectedRecipe) && !selectedRecipe?.immutable

  const [selection, setSelection] = useState<Selection | null>(null)
  const [sections, setSections] = useState<SectionState>({
    signals: true,
    projectionPartitions: true,
    projectionScores: true,
    projectionMappings: true,
    routes: true,
    plugins: true,
  })
  const [addingEntity, setAddingEntity] = useState<EntityKind | null>(null)
  const [outputPanelOpen, setOutputPanelOpen] = useState(true)
  const [guideOpen, setGuideOpen] = useState(false)
  const [showImportModal, setShowImportModal] = useState(false)
  const [showDuplicateModal, setShowDuplicateModal] = useState(false)
  const [pendingRecipeId, setPendingRecipeId] = useState<string | null>(null)
  const [recipePreview, setRecipePreview] = useState('')
  const [importText, setImportText] = useState('')
  const [importError, setImportError] = useState<string | null>(null)
  const [projectionError, setProjectionError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const importTextareaRef = useRef<HTMLTextAreaElement | null>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  const {
    width: outputWidth,
    isDragging,
    handleDragStart,
  } = useResizableWidth({
    initialWidth: 380,
    minWidth: 240,
    getMaxWidth: () => Math.floor((contentRef.current?.offsetWidth ?? window.innerWidth) * 0.6),
  })
  const {
    width: guideWidth,
    isDragging: guideDragging,
    handleDragStart: handleGuideDragStart,
  } = useResizableWidth({
    initialWidth: 420,
    minWidth: 300,
    getMaxWidth: () => 760,
    stopPropagation: true,
  })

  useEffect(() => {
    void initWasm()
  }, [initWasm])
  useEffect(() => {
    setMode('visual')
  }, [setMode])
  useEffect(() => {
    if (mode === 'visual' && wasmReady && dslSource.trim()) parseAST()
  }, [dslSource, mode, parseAST, wasmReady])
  useEffect(() => {
    setSelection(null)
    setAddingEntity(null)
    if (!selectedRecipe || !wasmReady) return
    try {
      setRecipePreview(compileBuilderRecipe(dslSource, selectedRecipe).preview)
    } catch {
      setRecipePreview('')
    }
    // Only refresh the preview when a newly loaded managed revision is selected.
    // Keystrokes are compiled explicitly to keep the editor responsive.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedRecipe?.id, selectedRecipe?.revision, wasmReady])

  const visualAst = useMemo(() => ast?.recipes?.[0]?.program ?? ast, [ast])
  const toggleSection = useCallback((key: keyof SectionState) => {
    setSections((current) => ({ ...current, [key]: !current[key] }))
  }, [])
  const counts = useMemo(
    () => ({
      signals: visualAst?.signals?.length ?? symbols?.signals?.length ?? 0,
      partitions: visualAst?.projectionPartitions?.length ?? 0,
      scores: visualAst?.projectionScores?.length ?? 0,
      mappings: visualAst?.projectionMappings?.length ?? 0,
      routes: visualAst?.routes?.length ?? symbols?.routes?.length ?? 0,
      plugins: visualAst?.plugins?.length ?? symbols?.plugins?.length ?? 0,
    }),
    [symbols, visualAst],
  )
  const errorCount =
    diagnostics.filter((diagnostic) => diagnostic.level === 'error').length + (compileError ? 1 : 0)
  const isValid = wasmReady && errorCount === 0

  const selectedEntity = useMemo(() => {
    if (!selection || !visualAst) return null
    switch (selection.kind) {
      case 'signal':
        return visualAst.signals?.find((item) => item.name === selection.name) ?? null
      case 'projection-partition':
        return visualAst.projectionPartitions?.find((item) => item.name === selection.name) ?? null
      case 'projection-score':
        return visualAst.projectionScores?.find((item) => item.name === selection.name) ?? null
      case 'projection-mapping':
        return visualAst.projectionMappings?.find((item) => item.name === selection.name) ?? null
      case 'route':
        return visualAst.routes?.find((item) => item.name === selection.name) ?? null
      case 'plugin':
        return visualAst.plugins?.find((item) => item.name === selection.name) ?? null
    }
  }, [selection, visualAst])

  const mutations = useBuilderScopedEntityMutations({
    recipeName: selectedRecipe?.name ?? null,
    setAddingEntity,
    setSelection,
  })

  const compileCurrent = useCallback(() => {
    if (!selectedRecipe) return
    try {
      const result = compileBuilderRecipe(useDSLStore.getState().dslSource, selectedRecipe)
      compile()
      setRecipePreview(result.preview)
      setProjectionError(null)
    } catch (cause) {
      setProjectionError(cause instanceof Error ? cause.message : 'The Recipe could not compile.')
    }
  }, [compile, selectedRecipe])

  const revert = useCallback(() => {
    if (!selectedRecipe) return
    const result = loadManagedRecipeSource(selectedRecipe)
    loadDsl(result.source)
    compile()
    parseAST()
    setRecipePreview(result.preview)
    setProjectionError(null)
  }, [compile, loadDsl, parseAST, selectedRecipe])

  const requestRecipeChange = (recipeId: string) => {
    if (recipeId === client.selectedRecipeId) return
    if (dirty) setPendingRecipeId(recipeId)
    else client.selectRecipe(recipeId)
  }

  const applyProjectedDraft = useCallback(
    (source: string, preview: string) => {
      loadDsl(source)
      useDSLStore.getState().setDslSource(source)
      compile()
      parseAST()
      setRecipePreview(preview)
      setMode('visual')
    },
    [compile, loadDsl, parseAST, setMode],
  )

  const confirmImport = () => {
    if (!selectedRecipe || !editable) return
    try {
      const result = projectImportedRecipe(importText.trim(), selectedRecipe)
      applyProjectedDraft(result.source, result.preview)
      setShowImportModal(false)
      setImportText('')
      setImportError(null)
    } catch (cause) {
      setImportError(cause instanceof Error ? cause.message : 'The Recipe could not be imported.')
    }
  }

  if (!canRead) {
    return (
      <ConfigPageRoutingScopeState
        loading={false}
        error="Recipe access is not available for this account."
        onRetry={() => window.location.reload()}
      />
    )
  }
  if (client.loading && !selectedRecipe) {
    return (
      <ConfigPageRoutingScopeState loading error={null} onRetry={() => void client.refresh()} />
    )
  }
  if (!selectedRecipe) {
    return (
      <ConfigPageRoutingScopeState
        loading={false}
        error={client.error}
        onRetry={() => void client.refresh()}
      />
    )
  }

  return (
    <div className={styles.page}>
      <BuilderToolbar
        readOnly={!canManage}
        immutable={selectedRecipe.immutable}
        dirty={dirty}
        mode={mode}
        wasmReady={wasmReady}
        wasmError={wasmError}
        dslSource={dslSource}
        loading={loading}
        saving={client.saving}
        recipes={client.recipes}
        selectedRecipeId={client.selectedRecipeId}
        guideOpen={guideOpen}
        outputPanelOpen={outputPanelOpen}
        onRecipeChange={requestRecipeChange}
        onModeSwitch={(nextMode: EditorMode) => setMode(nextMode)}
        onImport={() => {
          if (editable) {
            setImportError(null)
            setShowImportModal(true)
          }
        }}
        onCompile={compileCurrent}
        onSave={() =>
          void client
            .save()
            .then(compileCurrent)
            .catch(() => undefined)
        }
        onDuplicate={() => setShowDuplicateModal(true)}
        onFormat={format}
        onValidate={validate}
        onToggleGuide={() => setGuideOpen((open) => !open)}
        onToggleOutput={() => setOutputPanelOpen((open) => !open)}
        onRevert={revert}
      />

      {client.error || projectionError ? (
        <div className={styles.builderInlineAlert} role="alert">
          <ProductIcon name="alert" /> {client.error || projectionError}
        </div>
      ) : null}
      {client.notice ? (
        <div className={styles.builderNotice} role="status">
          <ProductIcon name="check" /> {client.notice}
        </div>
      ) : null}
      {selectedRecipe.immutable ? (
        <div className={styles.builderImmutableBanner}>
          <ProductIcon name="info" />
          <span>
            <strong>Built-in Recipe</strong> Duplicate it to make your own.
          </span>
        </div>
      ) : null}

      <div className={styles.content} ref={contentRef}>
        <div className={styles.editorArea}>
          {mode === 'visual' ? (
            <VisualMode
              readOnly={!editable}
              ast={visualAst}
              dslSource={dslSource}
              diagnostics={diagnostics}
              selection={selection}
              onSelect={setSelection}
              sections={sections}
              onToggleSection={toggleSection}
              selectedEntity={selectedEntity}
              signalCount={counts.signals}
              projectionPartitionCount={counts.partitions}
              projectionScoreCount={counts.scores}
              projectionMappingCount={counts.mappings}
              routeCount={counts.routes}
              pluginCount={counts.plugins}
              wasmReady={wasmReady}
              wasmError={wasmError}
              addingEntity={addingEntity}
              onSetAddingEntity={setAddingEntity}
              onDeleteEntity={mutations.handleDeleteEntity}
              onUpdateSignalFields={mutations.handleUpdateSignalFields}
              onUpdateProjectionPartitionFields={mutations.handleUpdateProjectionPartitionFields}
              onUpdateProjectionScoreFields={mutations.handleUpdateProjectionScoreFields}
              onUpdateProjectionMappingFields={mutations.handleUpdateProjectionMappingFields}
              onUpdatePluginFields={mutations.handleUpdatePluginFields}
              onAddSignal={mutations.handleAddSignal}
              onAddProjectionPartition={mutations.handleAddProjectionPartition}
              onAddProjectionScore={mutations.handleAddProjectionScore}
              onAddProjectionMapping={mutations.handleAddProjectionMapping}
              onAddPlugin={mutations.handleAddPlugin}
              onUpdateRoute={mutations.handleUpdateRoute}
              onAddRoute={mutations.handleAddRoute}
              errorCount={errorCount}
              isValid={isValid}
              onModeSwitch={setMode}
            />
          ) : mode === 'dsl' ? (
            <div className={styles.dslModeContainer}>
              <DslEditorPage embedded hideOutput readOnly={!editable} />
            </div>
          ) : null}
        </div>
        <BuilderOutputPanel
          open={outputPanelOpen}
          width={outputWidth}
          recipeDocument={recipePreview}
          dslSource={dslSource}
          compileError={compileError}
          onDragStart={handleDragStart}
          onOpen={() => setOutputPanelOpen(true)}
          onClose={() => setOutputPanelOpen(false)}
        />
      </div>

      <BuilderStatusBar
        isValid={isValid}
        errorCount={errorCount}
        recipeName={selectedRecipe.name}
        revision={selectedRecipe.recipeRevision}
        immutable={selectedRecipe.immutable}
        signalCount={counts.signals}
        routeCount={counts.routes}
        pluginCount={counts.plugins}
        lineCount={dslSource.split('\n').length}
        mode={mode}
      />

      {editable ? (
        <input
          ref={fileInputRef}
          type="file"
          accept=".yaml,.yml,.json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (!file) return
            const reader = new FileReader()
            reader.onload = () => {
              if (typeof reader.result === 'string') setImportText(reader.result)
            }
            reader.readAsText(file)
            event.target.value = ''
          }}
        />
      ) : null}
      <BuilderImportModal
        open={editable && showImportModal}
        importText={importText}
        importError={importError}
        importTextareaRef={importTextareaRef}
        onClose={() => setShowImportModal(false)}
        onImportTextChange={setImportText}
        onSelectFile={() => fileInputRef.current?.click()}
        onConfirm={confirmImport}
      />
      <BuilderGuideDrawer
        open={editable && guideOpen}
        width={guideWidth}
        isDragging={guideDragging}
        onClose={() => setGuideOpen(false)}
        onDragStart={handleGuideDragStart}
        onInsertSnippet={(snippet) => {
          const store = useDSLStore.getState()
          const next = mutateBuilderRecipeSource(
            store.dslSource,
            selectedRecipe.name,
            (body) => `${body.trimEnd()}\n\n${snippet}\n`,
          )
          store.setDslSource(next)
          store.parseAST()
          setGuideOpen(false)
        }}
      />

      <EditModal
        isOpen={showDuplicateModal}
        onClose={() => setShowDuplicateModal(false)}
        title="Duplicate Recipe"
        mode="add"
        data={{
          name: `${selectedRecipe.name} copy`,
          description: selectedRecipe.description ?? '',
        }}
        fields={DUPLICATE_FIELDS}
        onSave={async (data: EditFormData) => {
          const name = String(data.name ?? '').trim()
          if (!name) throw new Error('Name is required.')
          await client.duplicate(name, String(data.description ?? ''))
        }}
      />
      <ConfirmDialog
        isOpen={Boolean(pendingRecipeId)}
        eyebrow="Unsaved draft"
        title="Switch Recipes?"
        description="Your unsaved changes will be discarded."
        tone="warning"
        confirmLabel="Discard and switch"
        onCancel={() => setPendingRecipeId(null)}
        onConfirm={() => {
          if (pendingRecipeId) client.selectRecipe(pendingRecipeId)
          setPendingRecipeId(null)
        }}
      />
      {isDragging || guideDragging ? <div className={styles.dragOverlay} /> : null}
    </div>
  )
}
