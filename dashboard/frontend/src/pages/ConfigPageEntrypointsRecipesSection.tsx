import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Copy, Layers, Network, Plus } from 'lucide-react'

import { useShellStatus } from '../app/useShellStatus'
import ConfirmDialog from '../components/ConfirmDialog'
import type { FieldConfig } from '../components/EditModal'
import { useAuth } from '../contexts/AuthContext'
import type { RecipePackageList, RecipeProbeRunPlan } from '../types/recipe'
import { createProbePlaygroundInvocation } from '../types/playgroundInvocation'
import {
  Alert,
  Button,
  IconTile,
  Tabs,
  TabsList,
  TabsTrigger,
  cn,
  toast,
  useCopy,
  useRetainedValue,
} from '../ui'
import { canDeployConfig } from '../utils/accessControl'
import { listRecipePackages } from '../utils/recipeApi'
import ConfigPageEntrypointSheet from './ConfigPageEntrypointSheet'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageMixtureDialog from './ConfigPageMixtureDialog'
import ConfigPageMoMProbesPanel from './ConfigPageMoMProbesPanel'
import ConfigPageRecipeActivationDialog, {
  type RecipeToggleResult,
  type RecipeToggleTarget,
} from './ConfigPageRecipeActivationDialog'
import ConfigPageRecipeSheet, {
  RECIPE_SHEET_VIEWS,
  type RecipeSheetView,
} from './ConfigPageRecipeSheet'
import ConfigPageRecipesPublicModels from './ConfigPageRecipesPublicModels'
import ConfigPageRecipesTable from './ConfigPageRecipesTable'
import { useBuiltInRecipes } from './configPageBuiltInRecipes'
import { cloneConfigData } from './configPageCanonicalization'
import {
  collectRecipeTargetModels,
  getRecipeDeleteBlocker,
  normalizeRecipeStrategy,
  type RecipeFormState,
  validateRecipeForm,
} from './configPageEntrypointsRecipesSupport'
import type { MixtureRecipeChoice } from './configPageMixtureSupport'
import {
  ENTRYPOINT_SHEET_VIEWS,
  publicModelCards,
  recipeRows,
  type EntrypointSheetView,
  type BuiltInRecipeRow,
  type ConfigRecipeRow,
  type PackageRecipeRow,
  type PublicModelCard,
} from './configPageRecipesPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import type {
  ConfigData,
  EntrypointConfig,
  NormalizedModel,
  RecipeConfig,
} from './configPageSupport'
import { ROUTING_STRATEGIES } from './configPageSupport'

interface ConfigPageEntrypointsRecipesSectionProps {
  config: ConfigData
  isReadonly: boolean
  models: NormalizedModel[]
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
}

export type RecipesPageTab = 'models' | 'recipes' | 'probes'

const TABS: RecipesPageTab[] = ['models', 'recipes', 'probes']

interface PendingEntrypointDelete {
  entrypoint: EntrypointConfig
  index: number
}

interface MixtureEditorState {
  entrypoint?: EntrypointConfig
  index?: number
  recipe?: MixtureRecipeChoice
}

function useRecipePackages() {
  const [data, setData] = useState<RecipePackageList | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    void listRecipePackages({ page: 1, pageSize: 100 }, controller.signal)
      .then((next) => {
        setData(next)
        setError(null)
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setError(reason instanceof Error ? reason.message : 'Recipe packages could not be listed.')
      })
    return () => controller.abort()
  }, [attempt])
  return { data, error, refresh: () => setAttempt((value) => value + 1) }
}

export default function ConfigPageEntrypointsRecipesSection({
  config,
  isReadonly,
  models,
  saveConfig,
  openEditModal,
}: ConfigPageEntrypointsRecipesSectionProps) {
  const navigate = useNavigate()
  const { user } = useAuth()
  const { copy } = useCopy()
  const shell = useShellStatus()
  const routerHealthy = shell.status?.overall?.toLowerCase() === 'healthy'
  const canDeploy = !isReadonly && canDeployConfig(user)
  const builtInRecipes = useBuiltInRecipes()
  const packages = useRecipePackages()
  const [params, setParams] = useSearchParams()
  const update = useCallback(
    (patch: Record<string, string | null>, options?: { replace?: boolean }) =>
      setParams((current) => {
        const next = new URLSearchParams(current)
        for (const [key, value] of Object.entries(patch)) {
          if (value === null) next.delete(key)
          else next.set(key, value)
        }
        return next
      }, options),
    [setParams],
  )

  const tabParam = params.get('tab') as RecipesPageTab | null
  const tab: RecipesPageTab = tabParam && TABS.includes(tabParam) ? tabParam : 'models'
  const viewParam = params.get('view')

  const [entrypointPendingDelete, setEntrypointPendingDelete] =
    useState<PendingEntrypointDelete | null>(null)
  const [recipePendingDelete, setRecipePendingDelete] = useState<RecipeConfig | null>(null)
  const [deletePending, setDeletePending] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [mixtureEditor, setMixtureEditor] = useState<MixtureEditorState | null>(null)
  const [toggle, setToggle] = useState<RecipeToggleTarget | null>(null)
  const [pendingRestart, setPendingRestart] = useState<Map<string, string>>(new Map())
  const [probeSummary, setProbeSummary] = useState<{ total: number; failing: number } | null>(null)

  const cards = useMemo(() => publicModelCards(config), [config])
  const rows = useMemo(
    () => recipeRows(config, packages.data, builtInRecipes.data),
    [builtInRecipes.data, config, packages.data],
  )
  const pendingDigests = useMemo(() => new Set(pendingRestart.keys()), [pendingRestart])

  const activeCard = cards.find((card) => card.key === params.get('ep')) ?? null
  const shownCard = useRetainedValue(activeCard)
  const activeRow = rows.find((row) => row.id === params.get('recipe')) ?? null
  const shownRow = useRetainedValue(activeRow)
  const entrypointView: EntrypointSheetView =
    viewParam && ENTRYPOINT_SHEET_VIEWS.includes(viewParam as EntrypointSheetView)
      ? (viewParam as EntrypointSheetView)
      : 'overview'
  const recipeView: RecipeSheetView =
    viewParam && RECIPE_SHEET_VIEWS.includes(viewParam as RecipeSheetView)
      ? (viewParam as RecipeSheetView)
      : 'overview'
  // A save reloads the configuration and remounts this page, so the editor lives in state;
  // a link's ?new=1 opens it and leaves when it closes or saves.
  const linkedCreate = !isReadonly && params.get('new') === '1'
  useEffect(() => {
    if (linkedCreate) setMixtureEditor((current) => current ?? {})
  }, [linkedCreate])
  const clearCreateLink = () => {
    if (params.has('new')) update({ new: null }, { replace: true })
  }
  const mixture = mixtureEditor

  const openCard = (card: PublicModelCard, view?: EntrypointSheetView) =>
    update({ ep: card.key, view: view ?? null, recipe: null })
  const openRecipe = (id: string) => update({ tab: 'recipes', recipe: id, view: null, ep: null })
  const closeMixture = () => {
    setMixtureEditor(null)
    clearCreateLink()
  }
  const startCreate = (recipe?: MixtureRecipeChoice) => {
    setMixtureEditor({ recipe })
  }
  const startEdit = (card: PublicModelCard) =>
    setMixtureEditor({
      entrypoint: card.entrypoint,
      index: card.index < 0 ? undefined : card.index,
    })

  const openRecipeEditor = (mode: 'add' | 'edit', recipe?: RecipeConfig) => {
    const originalName = recipe?.name ?? null
    const form: RecipeFormState = {
      name: recipe?.name ?? '',
      description: recipe?.description ?? '',
      strategy: normalizeRecipeStrategy(
        recipe?.routing.strategy ?? config.global?.router?.strategy,
      ),
    }
    const fields: FieldConfig<RecipeFormState>[] = [
      {
        name: 'name',
        label: 'Recipe name',
        type: 'text',
        required: true,
        placeholder: 'speed-first',
        description: 'Stable internal policy identifier referenced by entrypoints.',
      },
      {
        name: 'description',
        label: 'Description',
        type: 'textarea',
        placeholder: 'Explain the objective and model allocation policy.',
      },
      {
        name: 'strategy',
        label: 'Decision strategy',
        type: 'select',
        required: true,
        options: [...ROUTING_STRATEGIES],
        description: 'How this recipe chooses among matching decisions.',
      },
    ]
    openEditModal(
      mode === 'add' ? 'Create Recipe' : `Edit Recipe · ${recipe?.name ?? ''}`,
      form,
      fields,
      async (data) => {
        const normalized = validateRecipeForm(data, config, models, originalName)
        const nextConfig = cloneConfigData(config)
        const nextRecipes = [...(nextConfig.recipes ?? [])]
        if (originalName === null) {
          nextRecipes.push(normalized)
        } else {
          const index = nextRecipes.findIndex((item) => item.name === originalName)
          if (index < 0) throw new Error(`Recipe "${originalName}" no longer exists.`)
          nextRecipes[index] = normalized
          if (normalized.name !== originalName) {
            nextConfig.entrypoints = (nextConfig.entrypoints ?? []).map((entrypoint) =>
              entrypoint.recipe === originalName
                ? { ...entrypoint, recipe: normalized.name }
                : entrypoint,
            )
          }
        }
        nextConfig.recipes = nextRecipes
        await saveConfig(nextConfig)
        toast.success(
          originalName === null ? `${normalized.name} created` : `${normalized.name} saved`,
        )
        if (originalName !== null && normalized.name !== originalName) {
          update({ recipe: normalized.name }, { replace: true })
        }
      },
      mode,
    )
  }

  const confirmDeleteEntrypoint = async () => {
    if (!entrypointPendingDelete) return
    setDeletePending(true)
    setDeleteError(null)
    try {
      const nextConfig = cloneConfigData(config)
      nextConfig.entrypoints = (nextConfig.entrypoints ?? []).filter(
        (_, index) => index !== entrypointPendingDelete.index,
      )
      await saveConfig(nextConfig)
      const name = entrypointPendingDelete.entrypoint.model_names[0]
      if (activeCard && activeCard.key === name) update({ ep: null, view: null })
      setEntrypointPendingDelete(null)
      toast(`${name} deleted`, {
        description: 'Clients that call this name now get model_not_found. The recipe stays.',
      })
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete entrypoint.')
    } finally {
      setDeletePending(false)
    }
  }

  const confirmDeleteRecipe = async () => {
    if (!recipePendingDelete) return
    const blocker = getRecipeDeleteBlocker(config, recipePendingDelete.name)
    if (blocker) {
      setDeleteError(blocker)
      return
    }
    setDeletePending(true)
    setDeleteError(null)
    try {
      const nextConfig = cloneConfigData(config)
      nextConfig.recipes = (nextConfig.recipes ?? []).filter(
        (recipe) => recipe.name !== recipePendingDelete.name,
      )
      await saveConfig(nextConfig)
      if (activeRow?.id === recipePendingDelete.name) update({ recipe: null, view: null })
      toast.success(`${recipePendingDelete.name} deleted`)
      setRecipePendingDelete(null)
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete recipe.')
    } finally {
      setDeletePending(false)
    }
  }

  const onToggleDone = (target: RecipeToggleTarget, result: RecipeToggleResult) => {
    setToggle(null)
    packages.refresh()
    const name = target.recipe.name || target.recipe.id
    if (result.status === 'restart_required') {
      setPendingRestart((current) =>
        new Map(current).set(
          target.recipe.recipe_digest,
          result.message ?? 'Restart required: run vllm-sr serve to apply.',
        ),
      )
      toast.warning(`${name} saved`, {
        description: result.message ?? 'Restart required: run vllm-sr serve to apply.',
      })
      return
    }
    setPendingRestart((current) => {
      const next = new Map(current)
      next.delete(target.recipe.recipe_digest)
      return next
    })
    toast.success(target.mode === 'activate' ? `${name} activated` : `${name} deactivated`, {
      description:
        target.mode === 'activate'
          ? 'The Router serves its public models now.'
          : 'The Router runs the configuration it had before the package.',
    })
  }

  const launchProbe = (intent: 'run' | 'edit', plan: RecipeProbeRunPlan) => {
    navigate('/playground', {
      state: { playgroundInvocation: createProbePlaygroundInvocation(intent, plan) },
    })
  }

  const handleNewPublicModel = (row: ConfigRecipeRow | BuiltInRecipeRow) => {
    update({ recipe: null, view: null })
    startCreate(row.kind === 'builtin' ? { bundle: row.bundle, name: row.name } : { name: row.id })
  }

  const pendingNames = rows
    .filter((row): row is PackageRecipeRow => row.kind === 'package')
    .filter((row) => pendingRestart.has(row.summary.recipe_digest))
    .map((row) => row.name)

  return (
    <ConfigPageManagerLayout
      title="Recipes"
      description="Public model names your apps call, and the recipes that decide how each request is routed."
      icon={<IconTile icon={Layers} tone="blue" size="lg" />}
      actions={
        <>
          <Button variant="outline" onClick={() => navigate('/routing/topology')}>
            <Network />
            Open topology
          </Button>
          {isReadonly ? null : (
            <Button onClick={() => startCreate()}>
              <Plus />
              New public model
            </Button>
          )}
        </>
      }
      tabs={
        <Tabs
          value={tab}
          onValueChange={(value) =>
            update(
              {
                tab: value === 'models' ? null : value,
                ep: null,
                recipe: null,
                probe: null,
                view: null,
              },
              { replace: true },
            )
          }
          variant="line"
          className="gap-0"
        >
          <TabsList aria-label="Recipes views">
            <TabsTrigger value="models">
              Public models
              <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
                {cards.length}
              </span>
            </TabsTrigger>
            <TabsTrigger value="recipes">
              Recipes
              <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
                {rows.length}
              </span>
            </TabsTrigger>
            <TabsTrigger value="probes">
              Probes
              {probeSummary ? (
                <span
                  className={cn(
                    'rounded-full px-1.5 font-mono text-[11px] tabular',
                    probeSummary.failing > 0
                      ? 'bg-destructive-soft text-destructive-foreground'
                      : 'bg-secondary text-muted-foreground',
                  )}
                >
                  {probeSummary.failing > 0
                    ? `${probeSummary.failing} failing`
                    : probeSummary.total}
                </span>
              ) : null}
            </TabsTrigger>
          </TabsList>
        </Tabs>
      }
    >
      {pendingNames.length ? (
        <div className="mb-4">
          <Alert
            tone="warning"
            title="Restart required: run vllm-sr serve to apply"
            actions={
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  copy('vllm-sr serve')
                  toast.success('Command copied')
                }}
              >
                <Copy />
                Copy command
              </Button>
            }
          >
            <span className="font-mono">{pendingNames.join(', ')}</span> is saved and applies on the
            next serve. vllm-sr status reports it until then.
          </Alert>
        </div>
      ) : null}

      {packages.data &&
      packages.data.activation_state !== 'none' &&
      packages.data.activation_state !== 'active' ? (
        <div className="mb-4" role="alert">
          <Alert
            tone={packages.data.activation_state === 'inconsistent' ? 'danger' : 'warning'}
            title={
              packages.data.activation_state === 'inconsistent'
                ? 'The recipe activation is inconsistent'
                : 'A recipe activation is recovering'
            }
          >
            {packages.data.activation_state === 'inconsistent'
              ? 'The running configuration does not match the recorded recipe package. Activate or deactivate a package again to restore a known state.'
              : 'The dashboard is finishing an interrupted activation. Refresh in a moment.'}
          </Alert>
        </div>
      ) : null}

      {tab === 'models' ? (
        <ConfigPageRecipesPublicModels
          cards={cards}
          config={config}
          models={models}
          isReadonly={isReadonly}
          routerHealthy={routerHealthy}
          activeKey={activeCard?.key ?? null}
          onOpen={(card, view) => openCard(card, view)}
          onEdit={startEdit}
          onDelete={(card) => {
            setDeleteError(null)
            setEntrypointPendingDelete({ entrypoint: card.entrypoint, index: card.index })
          }}
          onCreate={() => startCreate()}
        />
      ) : null}

      {tab === 'recipes' ? (
        <ConfigPageRecipesTable
          rows={rows}
          activeId={activeRow?.id ?? null}
          pendingRestart={pendingDigests}
          isReadonly={isReadonly}
          canDeploy={canDeploy}
          builtInsFailed={Boolean(builtInRecipes.error) && !builtInRecipes.refreshing}
          builtInsLoading={!builtInRecipes.data && !builtInRecipes.error}
          packagesError={packages.error}
          onRetryBuiltIns={() => void builtInRecipes.refresh()}
          onRetryPackages={packages.refresh}
          onOpen={(row) => openRecipe(row.id)}
          onCreateRecipe={() => openRecipeEditor('add')}
          onEdit={(row) => openRecipeEditor('edit', row.recipe)}
          onDelete={(row) => {
            setDeleteError(getRecipeDeleteBlocker(config, row.recipe.name))
            setRecipePendingDelete(row.recipe)
          }}
          onNewPublicModel={handleNewPublicModel}
          onToggle={(row, mode) => setToggle({ mode, recipe: row.summary })}
        />
      ) : null}

      {tab === 'probes' ? (
        <ConfigPageMoMProbesPanel
          onLaunch={launchProbe}
          recipeRevision={0}
          activeId={params.get('probe')}
          onOpen={(id) => update({ probe: id })}
          onClose={() => update({ probe: null })}
          onSummary={setProbeSummary}
          onOpenRecipes={() => update({ tab: 'recipes', probe: null }, { replace: true })}
        />
      ) : null}

      <ConfigPageEntrypointSheet
        card={shownCard}
        open={Boolean(activeCard)}
        view={entrypointView}
        onViewChange={(next) =>
          update({ view: next === 'overview' ? null : next }, { replace: true })
        }
        onOpenChange={(open) => (open ? null : update({ ep: null, view: null }))}
        config={config}
        models={models}
        isReadonly={isReadonly}
        routerHealthy={routerHealthy}
        onEdit={() => activeCard && startEdit(activeCard)}
        onDelete={() => {
          if (!activeCard) return
          setDeleteError(null)
          setEntrypointPendingDelete({ entrypoint: activeCard.entrypoint, index: activeCard.index })
        }}
        onOpenRecipe={(recipe) => openRecipe(recipe)}
      />

      <ConfigPageRecipeSheet
        row={shownRow}
        open={Boolean(activeRow)}
        view={recipeView}
        onViewChange={(next) =>
          update({ view: next === 'overview' ? null : next }, { replace: true })
        }
        onOpenChange={(open) => (open ? null : update({ recipe: null, view: null }))}
        models={models}
        pendingRestart={pendingDigests}
        isReadonly={isReadonly}
        canDeploy={canDeploy}
        onEdit={(row) => openRecipeEditor('edit', row.recipe)}
        onDelete={(row) => {
          setDeleteError(getRecipeDeleteBlocker(config, row.recipe.name))
          setRecipePendingDelete(row.recipe)
        }}
        onNewPublicModel={handleNewPublicModel}
        onOpenPublicModel={(name) => {
          const card = cards.find((candidate) => candidate.entrypoint.model_names.includes(name))
          if (card) update({ tab: null, recipe: null, ep: card.key, view: null })
        }}
        onToggle={(row, mode) => setToggle({ mode, recipe: row.summary })}
      />

      {mixture ? (
        <ConfigPageMixtureDialog
          config={config}
          models={models}
          builtInRecipes={builtInRecipes.data}
          initialRecipe={mixture.recipe}
          entrypoint={mixture.entrypoint}
          entrypointIndex={mixture.index}
          onClose={closeMixture}
          onSave={async (next) => {
            clearCreateLink()
            await saveConfig(next)
            toast.success(mixture.entrypoint ? 'Public model saved' : 'Public model created', {
              description: mixture.entrypoint
                ? undefined
                : 'Apps can call it as soon as the Router loads the change.',
            })
          }}
        />
      ) : null}

      <ConfigPageRecipeActivationDialog
        target={toggle}
        onOpenChange={(open) => (open ? null : setToggle(null))}
        onDone={onToggleDone}
      />

      <ConfirmDialog
        isOpen={entrypointPendingDelete !== null}
        title={`Delete ${entrypointPendingDelete?.entrypoint.model_names[0] ?? 'entrypoint'}?`}
        description="Remove these public model IDs from the router model catalog. Clients that call them get model_not_found; the recipe stays."
        confirmLabel="Delete entrypoint"
        pending={deletePending}
        details={
          entrypointPendingDelete ? (
            <span className="font-mono">
              {entrypointPendingDelete.entrypoint.model_names.join(', ')}
            </span>
          ) : undefined
        }
        errorMessage={deleteError}
        onCancel={() => {
          if (deletePending) return
          setEntrypointPendingDelete(null)
          setDeleteError(null)
        }}
        onConfirm={confirmDeleteEntrypoint}
      />

      <ConfirmDialog
        isOpen={recipePendingDelete !== null}
        title={`Delete recipe “${recipePendingDelete?.name ?? ''}”?`}
        description="Delete this named routing profile and all of its decisions."
        confirmLabel="Delete recipe"
        confirmationText={recipePendingDelete?.name}
        pending={deletePending}
        details={
          <div className="flex flex-col gap-0.5">
            <span>
              {collectRecipeTargetModels(recipePendingDelete).length} configured target models
            </span>
            <span>{recipePendingDelete?.routing.decisions?.length ?? 0} recipe decisions</span>
          </div>
        }
        errorMessage={deleteError}
        onCancel={() => {
          if (deletePending) return
          setRecipePendingDelete(null)
          setDeleteError(null)
        }}
        onConfirm={confirmDeleteRecipe}
      />
    </ConfigPageManagerLayout>
  )
}
