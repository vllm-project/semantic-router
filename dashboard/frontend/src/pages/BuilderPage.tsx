import * as React from 'react'
import { useSearchParams } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Braces, FileUp, Rocket, ShieldCheck, X } from 'lucide-react'

import { selectHasUnsavedChanges, useDSLStore } from '@/stores/dslStore'
import type { Diagnostic, DSLFieldObject, QuickFix } from '@/types/dsl'
import { useReadonly } from '@/contexts/ReadonlyContext'
import { useAuth } from '@/contexts/AuthContext'
import { canDeployConfig } from '@/utils/accessControl'
import {
  Alert,
  Button,
  IconTile,
  PageContainer,
  PageHeader,
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetTitle,
  Tooltip,
  toast,
} from '../ui'
import DslGuide from '../components/DslGuide'
import { SignalCapabilityProvider } from './SignalCapabilityContext'
import { SchemaFormHintsProvider } from './ConfigPageSchemaFieldsEditorSupport'
import {
  chooseDefaultBuilderRoutingScope,
  listBuilderRoutingScopes,
  listReferenceableSignals,
  resolveBuilderRoutingScope,
  summarizeBuilderRoutingScopes,
  type BuilderRoutingScope,
} from './builderPageRoutingScopeSupport'
import { useBuilderScopedEntityMutations } from './useBuilderScopedEntityMutations'
import { isSelectedSignal, type EntityKind, type Selection } from './builderPageTypes'
import { EditorCodePane, type CodePaneHandle } from './editor/EditorCodePane'
import { EditorStatusBar, EditorToolbar, type EditorView } from './editor/EditorChrome'
import { EditorDeployDialog } from './editor/EditorDeployDialog'
import {
  EditorCreatePanel,
  EditorEntityPanel,
  type EntityHandlers,
} from './editor/EditorEntityPanel'
import { EditorImportDialog } from './editor/EditorImportDialog'
import { EditorOutline } from './editor/EditorOutline'
import { EditorOutputPane, type OutputKind } from './editor/EditorOutputPane'
import { EditorOverview, OverviewSkeleton } from './editor/EditorOverview'
import { EditorProblems } from './editor/EditorProblems'
import { EditorRouterSettings } from './editor/EditorRouterSettings'
import {
  countDiagnostics,
  entityAtLine,
  groupMeta,
  groupOf,
  listEditorEntities,
  selectionKey,
  type EditorEntity,
} from './editor/editorModel'
import { programFormHints } from './editor/editorHints'

type Panel = 'overview' | 'global' | 'entity' | 'add'

const PROBLEMS_KEY = 'vsr-editor-problems'
const ADDABLE: EntityKind[] = [
  'model',
  'signal',
  'projection-partition',
  'projection-score',
  'projection-mapping',
  'route',
  'plugin',
]

function parseSelection(value: string | null): Selection | null {
  if (!value) return null
  const [kind, signalType, ...rest] = value.split(':')
  const name = rest.join(':')
  if (!ADDABLE.includes(kind as EntityKind) || !name) return null
  return {
    kind: kind as EntityKind,
    name,
    ...(kind === 'signal' && signalType ? { signalType } : {}),
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function readProblemsPreference(): boolean {
  try {
    return localStorage.getItem(PROBLEMS_KEY) !== '0'
  } catch {
    return true
  }
}

function downloadText(text: string, filename: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
  const link = Object.assign(document.createElement('a'), { href: url, download: filename })
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

const BuilderPage: React.FC = () => {
  const {
    dslSource,
    diagnostics,
    ast,
    compilerReady,
    compilerError,
    loading,
    renderedYamlOutput,
    yamlOutput,
    crdOutput,
    compileError,
    dirty,
    lastCompileAt,
    savedSource,
    baseConfigFromRouter,
    initCompiler,
    pauseEditorWork,
    compile,
    validate,
    parseAST,
    format,
    reset,
    setMode,
    setDslSource,
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
  // The source differs from the last load, import, reset, or successful deploy snapshot.
  const unsaved = useDSLStore(selectHasUnsavedChanges)

  const { serverReadonly, runtimeConfigWritable, isLoading: readonlyLoading } = useReadonly()
  const { user } = useAuth()
  const hasDeployPermission = canDeployConfig(user)
  const deployDisabled =
    readonlyLoading || serverReadonly || !runtimeConfigWritable || !hasDeployPermission
  const deployDisabledReason = readonlyLoading
    ? 'Checking deploy permissions…'
    : serverReadonly
      ? 'Deploy is disabled by the server-wide read-only policy'
      : !runtimeConfigWritable
        ? 'Deploy requires a writable runtime configuration mount'
        : !hasDeployPermission
          ? 'Deploy requires the config.deploy permission'
          : undefined

  const [params, setParams] = useSearchParams()
  const setParam = React.useCallback(
    (entries: Record<string, string | null>) =>
      setParams(
        (current) => {
          const next = new URLSearchParams(current)
          for (const [key, value] of Object.entries(entries)) {
            if (value === null) next.delete(key)
            else next.set(key, value)
          }
          return next
        },
        { replace: true },
      ),
    [setParams],
  )
  const modeParam = params.get('mode')
  const view: EditorView = modeParam === 'dsl' || modeParam === 'yaml' ? modeParam : 'visual'
  const selection = parseSelection(params.get('entity'))
  const addingParam = params.get('add') as EntityKind | null
  const adding = addingParam && ADDABLE.includes(addingParam) ? addingParam : null
  const panel: Panel =
    params.get('panel') === 'global' ? 'global' : adding ? 'add' : selection ? 'entity' : 'overview'

  const [configLoadError, setConfigLoadError] = React.useState<string | null>(null)
  const [loadingFromRouter, setLoadingFromRouter] = React.useState(false)
  const [showImport, setShowImport] = React.useState(false)
  const [guideOpen, setGuideOpen] = React.useState(false)
  const [showProblems, setShowProblems] = React.useState(readProblemsPreference)
  const [outputKind, setOutputKind] = React.useState<OutputKind>('yaml')
  const [cursor, setCursor] = React.useState({ line: 1, column: 1 })
  const [validating, setValidating] = React.useState(false)
  const [formatting, setFormatting] = React.useState(false)
  const [deployStarted, setDeployStarted] = React.useState(false)
  const [checkedAt, setCheckedAt] = React.useState<number | null>(null)
  const autoLoaded = React.useRef(false)
  const autoLoading = React.useRef(false)
  const codePane = React.useRef<CodePaneHandle>(null)
  const pendingReveal = React.useRef<{ line: number; endLine: number; column?: number } | null>(
    null,
  )

  React.useEffect(() => {
    try {
      localStorage.setItem(PROBLEMS_KEY, showProblems ? '1' : '0')
    } catch {
      /* storage unavailable */
    }
  }, [showProblems])

  // Scopes ------------------------------------------------------------------
  const routingScopes = React.useMemo(() => listBuilderRoutingScopes(ast), [ast])
  const scopeParam = params.get('scope')
  const scopeId =
    scopeParam && routingScopes.some((scope) => scope.id === scopeParam)
      ? scopeParam
      : chooseDefaultBuilderRoutingScope(ast)
  const activeScope = routingScopes.find((scope) => scope.id === scopeId) ?? null
  const visualAst = React.useMemo(
    () => resolveBuilderRoutingScope(ast, scopeId) ?? ast,
    [ast, scopeId],
  )
  const entities = React.useMemo(
    () => listEditorEntities(visualAst, dslSource),
    [visualAst, dslSource],
  )
  const allEntities = React.useMemo(
    () =>
      [ast, ...(ast?.recipes ?? []).map((recipe) => recipe.program)].flatMap((program) =>
        listEditorEntities(program ?? null, dslSource),
      ),
    [ast, dslSource],
  )
  const scopeStats = React.useCallback(
    (scope: BuilderRoutingScope) => {
      const summary = summarizeBuilderRoutingScopes(resolveBuilderRoutingScope(ast, scope.id), null)
      return `${summary.routeCount} decisions · ${summary.signalCount} signals`
    },
    [ast],
  )

  // Lifecycle ---------------------------------------------------------------
  React.useEffect(() => {
    void initCompiler()
    // Edits always refresh the AST: the outline reads it in every view.
    setMode('visual')
    return () => pauseEditorWork()
  }, [initCompiler, pauseEditorWork, setMode])

  React.useEffect(() => {
    if (compilerReady && useDSLStore.getState().dslSource.trim()) void parseAST()
  }, [compilerReady, parseAST])

  React.useEffect(() => {
    if (
      !compilerReady ||
      readonlyLoading ||
      dslSource.trim() ||
      autoLoaded.current ||
      autoLoading.current
    ) {
      return
    }
    autoLoading.current = true
    let cancelled = false
    const load = async () => {
      setLoadingFromRouter(true)
      setConfigLoadError(null)
      try {
        await loadFromRouter()
        if (!cancelled) {
          await compile()
          autoLoaded.current = true
        }
      } catch (error) {
        if (!cancelled) setConfigLoadError(`Failed to load router config: ${errorMessage(error)}`)
      } finally {
        autoLoading.current = false
        if (!cancelled) setLoadingFromRouter(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [compilerReady, readonlyLoading, dslSource, loadFromRouter, compile])

  React.useEffect(() => {
    setCheckedAt(Date.now())
  }, [diagnostics])

  // The compiled YAML view is always current when it opens.
  React.useEffect(() => {
    if (view !== 'yaml' || !compilerReady || !dslSource.trim()) return
    if (dirty || !yamlOutput) void compile()
    // Opening the view compiles once; later edits happen in the other views.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, compilerReady])

  // A pending reveal runs once the DSL editor has mounted.
  const revealPending = React.useCallback(() => {
    const target = pendingReveal.current
    if (!target) return
    pendingReveal.current = null
    window.requestAnimationFrame(() =>
      codePane.current?.reveal(target.line, target.endLine, target.column ?? 1),
    )
  }, [])

  // Navigation --------------------------------------------------------------
  const setView = (next: EditorView) => {
    if (next === 'dsl') {
      const current = entities.find((item) => selection && item.key === selectionKey(selection))
      if (current) pendingReveal.current = { line: current.line, endLine: current.endLine }
    }
    setParam({ mode: next === 'visual' ? null : next })
  }
  const showOverview = () => setParam({ entity: null, add: null, panel: null })
  const showGlobal = () => setParam({ entity: null, add: null, panel: 'global', mode: null })
  const openEntity = (entity: EditorEntity) => {
    if (view === 'dsl') {
      codePane.current?.reveal(entity.line, entity.endLine)
      setParam({ entity: entity.key, add: null, panel: null })
      return
    }
    setParam({ entity: entity.key, add: null, panel: null, mode: null })
  }
  const startAdd = (kind: EntityKind) =>
    setParam({ add: kind, entity: null, panel: null, mode: null })
  const showSource = (entity: EditorEntity) => {
    pendingReveal.current = { line: entity.line, endLine: entity.endLine }
    if (view === 'dsl') revealPending()
    else setParam({ mode: 'dsl' })
  }

  // Selection plumbing for the shared mutation hook.
  const setSelection: React.Dispatch<React.SetStateAction<Selection | null>> = (value) => {
    const next = typeof value === 'function' ? value(selection) : value
    setParam({ entity: next ? selectionKey(next) : null, add: null, panel: null })
  }
  const setAddingEntity: React.Dispatch<React.SetStateAction<EntityKind | null>> = (value) => {
    const next = typeof value === 'function' ? value(adding) : value
    setParam({ add: next })
  }
  const mutations = useBuilderScopedEntityMutations({
    recipeName: activeScope?.recipeName ?? null,
    setAddingEntity,
    setSelection,
  })

  const saved = (name: string) =>
    toast.success(`Saved ${name}`, {
      description: 'Written to the working copy. Review and deploy to apply it.',
    })
  const handlers: EntityHandlers = {
    onUpdateModel: (name, fields) => {
      mutations.handleUpdateModelFields(name, fields)
      saved(name)
    },
    onUpdateSignal: (signalType, name, fields) => {
      mutations.handleUpdateSignalFields(signalType, name, fields)
      saved(name)
    },
    onUpdatePlugin: (name, pluginType, fields) => {
      mutations.handleUpdatePluginFields(name, pluginType, fields)
      saved(name)
    },
    onUpdateProjection: (kind, name, fields) => {
      if (kind === 'partition') mutations.handleUpdateProjectionPartitionFields(name, fields)
      else if (kind === 'score') mutations.handleUpdateProjectionScoreFields(name, fields)
      else mutations.handleUpdateProjectionMappingFields(name, fields)
      saved(name)
    },
    onUpdateRoute: (name, input) => {
      mutations.handleUpdateRoute(name, input)
      saved(name)
    },
    onAddModel: mutations.handleAddModel,
    onAddSignal: mutations.handleAddSignal,
    onAddPlugin: mutations.handleAddPlugin,
    onAddProjection: (kind, name, fields: DSLFieldObject) => {
      if (kind === 'partition') mutations.handleAddProjectionPartition(name, fields)
      else if (kind === 'score') mutations.handleAddProjectionScore(name, fields)
      else mutations.handleAddProjectionMapping(name, fields)
    },
    onAddRoute: mutations.handleAddRoute,
  }

  const restoreSource = (source: string) => {
    const store = useDSLStore.getState()
    store.setDslSource(source)
    if (store.compilerReady) void store.parseAST()
  }

  const deleteEntity = (kind: EntityKind, name: string, subType?: string) => {
    const before = useDSLStore.getState().dslSource
    mutations.handleDeleteEntity(kind, name, subType)
    if (useDSLStore.getState().dslSource === before) {
      toast.error(`Could not find ${name} in the source`, {
        description: 'Edit it in the DSL view instead.',
      })
      return
    }
    toast(`${groupMeta[groupOf(kind)].one} ${name} deleted`, {
      description: 'Removed from the working copy.',
      action: { label: 'Undo', onClick: () => restoreSource(before) },
    })
  }

  // Selected entity ---------------------------------------------------------
  const selectedEntity = React.useMemo(() => {
    if (!selection || !visualAst) return null
    switch (selection.kind) {
      case 'model':
        return visualAst.models?.find((model) => model.name === selection.name) ?? null
      case 'signal':
        return visualAst.signals?.find((signal) => isSelectedSignal(selection, signal)) ?? null
      case 'projection-partition':
        return visualAst.projectionPartitions?.find((item) => item.name === selection.name) ?? null
      case 'projection-score':
        return visualAst.projectionScores?.find((item) => item.name === selection.name) ?? null
      case 'projection-mapping':
        return visualAst.projectionMappings?.find((item) => item.name === selection.name) ?? null
      case 'route':
        return visualAst.routes?.find((route) => route.name === selection.name) ?? null
      case 'plugin':
        return visualAst.plugins?.find((plugin) => plugin.name === selection.name) ?? null
      default:
        return null
    }
  }, [selection, visualAst])
  const selectedKey = selection ? selectionKey(selection) : null
  const located = entities.find((item) => item.key === selectedKey) ?? null

  const availablePlugins = React.useMemo(
    () =>
      visualAst?.plugins?.map((plugin) => ({ name: plugin.name, pluginType: plugin.pluginType })) ??
      [],
    [visualAst?.plugins],
  )
  const availableModels = React.useMemo(() => {
    const names = new Set<string>((ast?.models ?? []).map((model) => model.name))
    for (const program of [ast, ...(ast?.recipes ?? []).map((recipe) => recipe.program)]) {
      for (const route of program?.routes ?? [])
        for (const model of route.models) names.add(model.model)
    }
    return [...names].filter(Boolean).sort()
  }, [ast])
  const referenceable = React.useMemo(
    () => listReferenceableSignals(visualAst?.signals, visualAst?.projectionMappings),
    [visualAst?.signals, visualAst?.projectionMappings],
  )
  const hints = React.useMemo(() => {
    const base = programFormHints(
      visualAst,
      availableModels,
      selection ? groupMeta[groupOf(selection.kind)].tone : 'violet',
    )
    return {
      ...base,
      // Complexity levels and authz roles are referenced the way the Router reports them.
      signalNames: (type: string) =>
        referenceable.filter((signal) => signal.signalType === type).map((signal) => signal.name),
    }
  }, [availableModels, referenceable, selection, visualAst])

  // Diagnostics -------------------------------------------------------------
  const sourceError = configLoadError ?? compileError
  const errorCount =
    countDiagnostics(diagnostics, 'error') + (compileError ? 1 : 0) + (configLoadError ? 1 : 0)
  const warningCount = countDiagnostics(diagnostics, 'warning')
  const problemsByEntity = React.useMemo(() => {
    const worst = new Map<string, Diagnostic['level']>()
    const rank = { error: 3, warning: 2, constraint: 1 }
    for (const diagnostic of diagnostics) {
      const owner = entityAtLine(entities, diagnostic.line)
      if (!owner) continue
      const current = worst.get(owner.key)
      if (!current || rank[diagnostic.level] > rank[current]) worst.set(owner.key, diagnostic.level)
    }
    return worst
  }, [diagnostics, entities])

  const showDiagnostic = (diagnostic: Diagnostic) => {
    const owner =
      entityAtLine(entities, diagnostic.line) ?? entityAtLine(allEntities, diagnostic.line)
    if (view === 'visual' && owner && entities.includes(owner)) {
      openEntity(owner)
      return
    }
    pendingReveal.current = {
      line: diagnostic.line,
      endLine: diagnostic.line,
      column: diagnostic.column,
    }
    if (view === 'dsl') revealPending()
    else setParam({ mode: 'dsl' })
  }

  const applyFix = (diagnostic: Diagnostic, fix: QuickFix) => {
    const before = useDSLStore.getState().dslSource
    const lines = before.split('\n')
    const text = lines[diagnostic.line - 1]
    if (text === undefined) return
    let start = diagnostic.column
    let end = diagnostic.column
    while (start > 1 && /[\w\-.]/.test(text[start - 2])) start -= 1
    while (end <= text.length && /[\w\-.]/.test(text[end - 1])) end += 1
    lines[diagnostic.line - 1] = text.slice(0, start - 1) + fix.newText + text.slice(end - 1)
    restoreSource(lines.join('\n'))
    toast.success(fix.description || 'Fix applied', {
      description: `Line ${diagnostic.line} now reads ${fix.newText}.`,
      action: { label: 'Undo', onClick: () => restoreSource(before) },
    })
  }

  // Toolbar actions ---------------------------------------------------------
  const runValidate = async () => {
    setValidating(true)
    try {
      await validate()
      const state = useDSLStore.getState()
      const errors = countDiagnostics(state.diagnostics, 'error') + (state.compileError ? 1 : 0)
      const warnings = countDiagnostics(state.diagnostics, 'warning')
      if (errors)
        toast.error(`${errors} ${errors === 1 ? 'error' : 'errors'} found`, {
          description: 'Fix them before you deploy. Select a problem to jump to it.',
        })
      else
        toast.success('The working copy is valid', {
          description: `${warnings} ${warnings === 1 ? 'warning' : 'warnings'} from the compiler.`,
        })
      if (errors || warnings) setShowProblems(true)
    } finally {
      setValidating(false)
    }
  }

  const runFormat = async () => {
    setFormatting(true)
    try {
      await format()
      if (!useDSLStore.getState().compileError) await parseAST()
    } finally {
      setFormatting(false)
    }
  }

  const runCompile = async () => {
    await compile()
    const state = useDSLStore.getState()
    const errors = countDiagnostics(state.diagnostics, 'error') + (state.compileError ? 1 : 0)
    if (errors) {
      setShowProblems(true)
      toast.error(`Compile failed with ${errors} ${errors === 1 ? 'error' : 'errors'}`)
    }
  }

  const discard = () => {
    const before = useDSLStore.getState().dslSource
    restoreSource(savedSource)
    toast('Changes discarded', {
      description: 'The working copy matches the loaded config again.',
      action: { label: 'Undo', onClick: () => restoreSource(before) },
    })
  }

  const clearEditor = () => {
    const before = useDSLStore.getState().dslSource
    setConfigLoadError(null)
    autoLoaded.current = true
    reset()
    showOverview()
    toast('Started from an empty file', {
      action: { label: 'Undo', onClick: () => restoreSource(before) },
    })
  }

  const reloadFromRouter = async () => {
    setLoadingFromRouter(true)
    try {
      await loadFromRouter()
      await compile()
      setConfigLoadError(null)
      autoLoaded.current = true
      toast.success('Reloaded from the Router')
    } catch (error) {
      setConfigLoadError(`Failed to load router config: ${errorMessage(error)}`)
    } finally {
      setLoadingFromRouter(false)
    }
  }

  const importConfig = async (yaml: string) => {
    await importYaml(yaml)
    await compile()
    setConfigLoadError(null)
    autoLoaded.current = true
    showOverview()
  }

  const loadRouterFromDialog = async () => {
    setLoadingFromRouter(true)
    try {
      await loadFromRouter()
      await compile()
      setConfigLoadError(null)
      autoLoaded.current = true
      showOverview()
    } finally {
      setLoadingFromRouter(false)
    }
  }

  const insertSnippet = (snippet: string) => {
    const source = useDSLStore.getState().dslSource
    setDslSource(source ? `${source.trimEnd()}\n\n${snippet}\n` : `${snippet}\n`)
    setGuideOpen(false)
    const line = source ? source.trimEnd().split('\n').length + 2 : 1
    pendingReveal.current = { line, endLine: line + snippet.split('\n').length - 1 }
    if (view === 'dsl') revealPending()
    else setParam({ mode: 'dsl' })
  }

  // Deploy ------------------------------------------------------------------
  const requestReview = async () => {
    if (deployDisabled) return
    setDeployStarted(false)
    await requestDeploy()
    const state = useDSLStore.getState()
    if (!state.showDeployConfirm && state.deployResult?.status === 'error') {
      setShowProblems(true)
      toast.error('Fix the compiler errors first', { description: state.deployResult.message })
      dismissDeploy()
    }
  }
  const closeDeploy = () => {
    dismissDeploy()
    setDeployStarted(false)
  }
  const deployOpen = showDeployConfirm || deploying || (deployStarted && deployResult !== null)
  React.useEffect(() => {
    if (deployStarted && deployResult?.status === 'success') {
      toast.success(deployResult.version ? `Deployed v${deployResult.version}` : 'Deployed', {
        description: deployResult.message,
      })
    }
  }, [deployResult, deployStarted])

  // Render ------------------------------------------------------------------
  const hasSource = Boolean(dslSource.trim())
  const lineCount = dslSource ? dslSource.split('\n').length : 0
  const scopeLabel = activeScope?.recipeName ?? activeScope?.label ?? 'default'
  const busyLoading = loadingFromRouter || (!compilerReady && !compilerError)

  const visualContent =
    panel === 'global' ? (
      <EditorRouterSettings />
    ) : panel === 'add' && adding ? (
      <EditorCreatePanel
        key={`add-${adding}`}
        kind={adding}
        availableModels={availableModels}
        availablePlugins={availablePlugins}
        handlers={handlers}
        onCancel={showOverview}
      />
    ) : panel === 'entity' && selection ? (
      <EditorEntityPanel
        key={`${selectedKey}:${JSON.stringify(selectedEntity)}`}
        selection={selection}
        entity={selectedEntity}
        located={located}
        availableModels={availableModels}
        availablePlugins={availablePlugins}
        handlers={handlers}
        onDelete={deleteEntity}
        onShowSource={showSource}
        onBack={showOverview}
      />
    ) : busyLoading && !ast ? (
      <OverviewSkeleton />
    ) : (
      <EditorOverview
        ast={visualAst}
        scope={activeScope}
        entities={entities}
        errors={errorCount}
        warnings={warningCount}
        readOnly={false}
        onSelect={openEntity}
        onAdd={startAdd}
        onShowProblems={() => setShowProblems(true)}
        onGlobal={showGlobal}
      />
    )

  return (
    <PageContainer width="full">
      <div className="flex min-h-[640px] flex-1 flex-col px-5 pb-5 pt-6 md:px-6">
        <PageHeader
          className="mb-5"
          icon={<IconTile icon={Braces} tone="violet" size="lg" />}
          title="Config editor"
          description="Edit routing as code or with forms, validate it with the compiler and deploy it as a new version."
          actions={
            <>
              <Button
                variant="outline"
                onClick={() => setShowImport(true)}
                disabled={!compilerReady}
              >
                <FileUp />
                Import
              </Button>
              <Tooltip
                content={
                  deployDisabledReason ??
                  (unsaved
                    ? 'Review the merged config, then deploy it'
                    : 'Review the config and deploy it')
                }
              >
                <span>
                  <Button
                    onClick={() => void requestReview()}
                    disabled={deployDisabled || !compilerReady || !hasSource || loading}
                  >
                    <Rocket />
                    Review & deploy
                    {unsaved ? (
                      <span
                        aria-hidden
                        className="size-1.5 rounded-full bg-primary-foreground/80"
                      />
                    ) : null}
                  </Button>
                </span>
              </Tooltip>
            </>
          }
        />

        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-xs">
          <EditorToolbar
            scopes={routingScopes}
            scopeId={scopeId}
            scopeStats={scopeStats}
            onScopeChange={(next) => setParam({ scope: next, entity: null, add: null })}
            view={view}
            onViewChange={setView}
            compilerReady={compilerReady}
            hasSource={hasSource}
            validating={validating}
            formatting={formatting}
            unsaved={unsaved}
            showProblems={showProblems}
            guideOpen={guideOpen}
            onValidate={() => void runValidate()}
            onFormat={() => void runFormat()}
            onDiscard={discard}
            onToggleProblems={() => setShowProblems((current) => !current)}
            onToggleGuide={() => setGuideOpen((current) => !current)}
            onReloadFromRouter={() => void reloadFromRouter()}
            onCopyDsl={() => {
              void navigator.clipboard?.writeText(dslSource)
              toast.success('DSL copied')
            }}
            onDownloadDsl={() => downloadText(dslSource, 'routing.dsl')}
            onClear={clearEditor}
          />

          {configLoadError || compileError ? (
            <div role="alert" className="shrink-0 border-b border-border px-3 py-2.5">
              <Alert
                tone="danger"
                title={
                  configLoadError
                    ? 'The config could not be loaded'
                    : 'The compiler rejected the source'
                }
                actions={
                  configLoadError ? (
                    <Button
                      variant="outline"
                      size="xs"
                      onClick={() => void reloadFromRouter()}
                      loading={loadingFromRouter}
                    >
                      Retry
                    </Button>
                  ) : null
                }
              >
                <span className="break-words font-mono text-[12px]">
                  {configLoadError ?? compileError}
                </span>
                {configLoadError ? (
                  <span> Use Import to retry loading or choose a corrected config.</span>
                ) : null}
              </Alert>
            </div>
          ) : null}

          <div className="flex min-h-0 flex-1">
            <aside className="hidden w-[248px] shrink-0 flex-col border-r border-border bg-surface-subtle/50 lg:flex">
              <EditorOutline
                entities={entities}
                problems={problemsByEntity}
                activeKey={selectedKey}
                target={panel}
                loading={busyLoading}
                onOverview={() => {
                  showOverview()
                  if (view !== 'visual') setParam({ mode: null })
                }}
                onGlobal={showGlobal}
                onSelect={openEntity}
                onAdd={startAdd}
              />
            </aside>

            <section className="relative min-w-0 flex-1" aria-label="Editor content">
              {view === 'visual' ? (
                <motion.div
                  key="visual"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.16 }}
                  className="absolute inset-0 overflow-y-auto bg-surface-subtle/30 px-5 pt-5"
                >
                  <SignalCapabilityProvider
                    active={adding === 'signal' || selection?.kind === 'signal'}
                    scope={{
                      recipe: activeScope?.recipeName || 'default',
                      bindings: ast?.modelBindings,
                    }}
                  >
                    <SchemaFormHintsProvider value={hints}>
                      <div className="pb-5">{visualContent}</div>
                    </SchemaFormHintsProvider>
                  </SignalCapabilityProvider>
                </motion.div>
              ) : view === 'dsl' ? (
                <motion.div
                  key="dsl"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.16 }}
                  className="absolute inset-0"
                >
                  <EditorCodePane
                    ref={codePane}
                    value={dslSource}
                    diagnostics={diagnostics}
                    onChange={setDslSource}
                    onCursor={setCursor}
                    onCompile={() => void runCompile()}
                    onFormat={() => void runFormat()}
                    onReady={revealPending}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="yaml"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.16 }}
                  className="absolute inset-0"
                >
                  <EditorOutputPane
                    yaml={renderedYamlOutput || yamlOutput}
                    crd={crdOutput}
                    kind={outputKind}
                    onKindChange={setOutputKind}
                    compiling={loading}
                    stale={dirty}
                    compiledAt={lastCompileAt}
                    canCompile={compilerReady && hasSource}
                    onCompile={() => void runCompile()}
                  />
                </motion.div>
              )}
            </section>

            <AnimatePresence initial={false}>
              {showProblems ? (
                <motion.aside
                  key="problems"
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: 304, opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                  className="hidden shrink-0 overflow-hidden border-l border-border bg-surface-subtle/50 xl:flex"
                >
                  <div className="flex w-[304px] min-w-[304px] flex-col">
                    <EditorProblems
                      diagnostics={diagnostics}
                      sourceError={sourceError}
                      ownerOf={(diagnostic) =>
                        entityAtLine(entities, diagnostic.line) ??
                        entityAtLine(allEntities, diagnostic.line)
                      }
                      onSelect={showDiagnostic}
                      onFix={applyFix}
                      footer={
                        <span className="flex items-center gap-1.5">
                          <ShieldCheck className="size-3.5" />
                          {checkedAt && hasSource
                            ? `Checked by the DSL compiler ${new Date(checkedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`
                            : 'Checked by the DSL compiler on every edit'}
                        </span>
                      }
                    />
                  </div>
                </motion.aside>
              ) : null}
            </AnimatePresence>
          </div>

          <EditorStatusBar
            errors={errorCount}
            warnings={warningCount}
            ready={compilerReady}
            compilerError={compilerError}
            view={view}
            cursor={cursor}
            entityCount={entities.length}
            lineCount={lineCount}
            scopeLabel={scopeLabel}
            unsaved={unsaved}
            onShowProblems={() => setShowProblems(true)}
          />
        </div>
      </div>

      <EditorImportDialog
        open={showImport}
        onOpenChange={setShowImport}
        onImport={importConfig}
        onLoadFromRouter={loadRouterFromDialog}
      />

      <EditorDeployDialog
        open={deployOpen}
        previewLoading={deployPreviewLoading}
        previewError={deployPreviewError}
        currentYaml={deployPreviewCurrent}
        mergedYaml={deployPreviewMerged}
        errors={errorCount}
        warnings={warningCount}
        baseFromRouter={baseConfigFromRouter}
        deploying={deploying}
        deployStep={deployStep}
        deployResult={deployStarted ? deployResult : null}
        onDeploy={() => {
          setDeployStarted(true)
          void executeDeploy()
        }}
        onRetry={() => {
          closeDeploy()
          void requestReview()
        }}
        onClose={closeDeploy}
        onShowProblems={() => {
          closeDeploy()
          setShowProblems(true)
          setParam({ mode: 'dsl' })
        }}
      />

      <Sheet open={guideOpen} onOpenChange={setGuideOpen}>
        <SheetContent size="sm" aria-describedby="editor-guide-description">
          <div className="relative flex shrink-0 flex-col gap-1 border-b border-border px-4 pb-3 pt-4 pr-12">
            <SheetTitle className="text-base font-semibold">DSL language guide</SheetTitle>
            <SheetDescription
              id="editor-guide-description"
              className="text-xs text-muted-foreground"
            >
              Syntax, every signal, algorithm and plugin type, and snippets you can insert.
            </SheetDescription>
            <SheetClose asChild>
              <Button
                variant="ghost"
                size="icon-xs"
                className="absolute right-3 top-3"
                aria-label="Close DSL language guide"
              >
                <X />
              </Button>
            </SheetClose>
          </div>
          <DslGuide onInsertSnippet={insertSnippet} />
        </SheetContent>
      </Sheet>
    </PageContainer>
  )
}

export default BuilderPage
