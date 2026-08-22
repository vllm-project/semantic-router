import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import ConfirmDialog from '../components/ConfirmDialog'
import ConfigPageMixtureDialog from './ConfigPageMixtureDialog'
import ConfigPageMoMTopologyDialog from './ConfigPageMoMTopologyDialog'
import { ConfigPageMoMEntrypointsList, ConfigPageMoMRecipesList } from './ConfigPageMoMRoutingLists'
import ConfigPageRecipeDialog from './ConfigPageRecipeDialog'
import { announceRecipeDraftChange, recipeDraftApi, type RecipeDraft } from '../utils/recipeDrafts'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import { cloneConfigData } from './configPageCanonicalization'
import {
  collectRecipeTargetModels,
  countRecipeEntrypoints,
  DEFAULT_RECIPE_NAME,
  getRecipeByName,
  getRecipeDeleteBlocker,
  getRecipeReadiness,
  validateEntrypointForm,
} from './configPageEntrypointsRecipesSupport'
import type {
  ConfigData,
  EntrypointConfig,
  NormalizedModel,
  RecipeConfig,
} from './configPageSupport'
import { DEFAULT_ROUTING_STRATEGY } from './configPageSupport'
import type { OpenViewModal } from './configPageRouterSectionSupport'
import type { MixtureWorkspaceView } from './ConfigPageEntrypointsRecipesSection'
import {
  countProjectionsInProfile,
  countSignalsInProfile,
  type RoutingProfileLike,
} from '../utils/routingScopes'

interface ConfigPageMoMRoutingPanelProps {
  activeView: MixtureWorkspaceView
  config: ConfigData
  isReadonly: boolean
  models: NormalizedModel[]
  saveConfig: (config: ConfigData) => Promise<void>
  openViewModal: OpenViewModal
}

interface PendingEntrypointDelete {
  entrypoint: EntrypointConfig
  index: number
}

export default function ConfigPageMoMRoutingPanel({
  activeView,
  config,
  isReadonly,
  models,
  saveConfig,
  openViewModal,
}: ConfigPageMoMRoutingPanelProps) {
  const navigate = useNavigate()
  const [entrypointPendingDelete, setEntrypointPendingDelete] =
    useState<PendingEntrypointDelete | null>(null)
  const [recipePendingDelete, setRecipePendingDelete] = useState<RecipeConfig | null>(null)
  const [deletePending, setDeletePending] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [topologyTarget, setTopologyTarget] = useState<{
    entrypoint: EntrypointConfig
    recipe: RecipeConfig
  } | null>(null)
  const [mixtureEditor, setMixtureEditor] = useState<{
    entrypoint?: EntrypointConfig
    index: number | null
  } | null>(null)
  const [recipeEditor, setRecipeEditor] = useState<RecipeConfig | null | undefined>(null)
  const [recipeEditorOpen, setRecipeEditorOpen] = useState(false)

  const [drafts, setDrafts] = useState<RecipeDraft[]>([])
  const loadDrafts = useCallback(async () => {
    try {
      setDrafts((await recipeDraftApi.list()).items)
    } catch {
      setDrafts([])
    }
  }, [])
  useEffect(() => {
    void loadDrafts()
    const refresh = () => void loadDrafts()
    window.addEventListener('recipe-drafts-changed', refresh)
    return () => window.removeEventListener('recipe-drafts-changed', refresh)
  }, [loadDrafts])
  const combinedConfig = useMemo<ConfigData>(() => {
    const live = new Set((config.recipes ?? []).map((recipe) => recipe.name))
    return {
      ...config,
      recipes: [...(config.recipes ?? []), ...drafts.filter((draft) => !live.has(draft.name))],
    }
  }, [config, drafts])
  const draftNames = useMemo(() => new Set(drafts.map((draft) => draft.name)), [drafts])

  const entrypoints = config.entrypoints ?? []

  const saveMixture = async (entrypoint: EntrypointConfig) => {
    const originalIndex = mixtureEditor?.index ?? null
    const identity = validateEntrypointForm(
      { modelNames: entrypoint.model_names.join('\n'), recipe: entrypoint.recipe },
      combinedConfig,
      models,
      originalIndex,
    )
    const nextConfig = cloneConfigData(config)
    const draft = drafts.find((candidate) => candidate.name === identity.recipe)
    if (draft && !(nextConfig.recipes ?? []).some((recipe) => recipe.name === draft.name)) {
      const readiness = getRecipeReadiness(draft, models)
      if (!readiness.ready) throw new Error(readiness.reason)
      nextConfig.recipes = [
        ...(nextConfig.recipes ?? []),
        { name: draft.name, description: draft.description, routing: draft.routing },
      ]
    }
    const nextEntrypoints = [...(nextConfig.entrypoints ?? [])]
    const normalized = { ...identity, model_bindings: entrypoint.model_bindings }
    if (originalIndex === null) nextEntrypoints.push(normalized)
    else nextEntrypoints[originalIndex] = normalized
    nextConfig.entrypoints = nextEntrypoints
    await saveConfig(nextConfig)
    if (draft) {
      await recipeDraftApi.remove(draft.name)
      announceRecipeDraftChange()
      await loadDrafts()
    }
  }

  const saveRecipe = async (
    identity: { name: string; description: string },
    originalName: string | null,
  ) => {
    const name = identity.name.trim()
    if (!name) throw new Error('Recipe name is required.')
    if (!/^[a-z0-9][a-z0-9_-]*$/.test(name)) {
      throw new Error('Use lowercase letters, numbers, hyphens, or underscores.')
    }
    if (name === DEFAULT_RECIPE_NAME) throw new Error('default is reserved for live routing.')
    if (
      (combinedConfig.recipes ?? []).some(
        (candidate) => candidate.name === name && candidate.name !== originalName,
      )
    ) {
      throw new Error(`Recipe “${name}” already exists.`)
    }
    const published = Boolean(
      originalName && (config.recipes ?? []).some((candidate) => candidate.name === originalName),
    )
    if (published) {
      if (name !== originalName) throw new Error('Published Recipe names cannot be changed.')
      const nextConfig = cloneConfigData(config)
      const target = (nextConfig.recipes ?? []).find((candidate) => candidate.name === originalName)
      if (!target) throw new Error('Recipe no longer exists.')
      target.description = identity.description.trim() || undefined
      await saveConfig(nextConfig)
      return
    }
    const current = originalName
      ? drafts.find((candidate) => candidate.name === originalName)
      : undefined
    await recipeDraftApi.save({
      name,
      description: identity.description.trim() || undefined,
      routing: current?.routing ?? {
        strategy: DEFAULT_ROUTING_STRATEGY,
        signals: {},
        projections: {},
        decisions: [],
      },
    })
    if (originalName && originalName !== name) await recipeDraftApi.remove(originalName)
    announceRecipeDraftChange()
    await loadDrafts()
  }

  const viewEntrypoint = (entrypoint: EntrypointConfig, index: number) => {
    const recipe = getRecipeByName(config, entrypoint.recipe)
    const targets = [
      ...new Set(
        Object.values(entrypoint.model_bindings ?? {})
          .flat()
          .map((reference) => reference.model),
      ),
    ]
    openViewModal(
      entrypoint.model_names.join(', '),
      [
        {
          title: 'Model composition',
          fields: [
            { label: 'Model names', value: entrypoint.model_names.join('\n'), fullWidth: true },
            { label: 'Recipe', value: entrypoint.recipe },
            { label: 'Decisions', value: recipe?.routing.decisions?.length ?? 0 },
            {
              label: 'Models',
              value: targets.join('\n') || 'No models assigned',
              fullWidth: true,
            },
          ],
        },
      ],
      isReadonly ? undefined : () => setMixtureEditor({ entrypoint, index }),
      [
        ...(recipe
          ? [
              {
                label: 'View topology',
                onClick: () => setTopologyTarget({ entrypoint, recipe }),
              },
            ]
          : []),
        ...(!isReadonly
          ? [
              {
                label: 'Delete model',
                tone: 'destructive' as const,
                onClick: () => {
                  setDeleteError(null)
                  setEntrypointPendingDelete({ entrypoint, index })
                },
              },
            ]
          : []),
      ],
    )
  }

  const viewRecipe = (recipe: RecipeConfig) => {
    const targets = collectRecipeTargetModels(recipe)
    const readiness = getRecipeReadiness(recipe, models)
    openViewModal(
      recipe.name,
      [
        {
          title: 'Recipe profile',
          fields: [
            {
              label: 'Status',
              value: draftNames.has(recipe.name) ? readiness.reason : 'Published',
            },
            {
              label: 'Description',
              value: recipe.description || 'No description',
              fullWidth: true,
            },
            { label: 'Entrypoint models', value: countRecipeEntrypoints(entrypoints, recipe.name) },
            {
              label: 'Decision strategy',
              value: recipe.routing.strategy ?? DEFAULT_ROUTING_STRATEGY,
            },
            { label: 'Decisions', value: recipe.routing.decisions?.length ?? 0 },
            {
              label: 'Signals',
              value: countSignalsInProfile(recipe.routing as RoutingProfileLike).total,
            },
            {
              label: 'Projections',
              value: countProjectionsInProfile(recipe.routing as RoutingProfileLike),
            },
            {
              label: 'Configured targets',
              value: targets.join('\n') || 'No target models',
              fullWidth: true,
            },
          ],
        },
      ],
      isReadonly
        ? undefined
        : () => {
            setRecipeEditor(recipe)
            setRecipeEditorOpen(true)
          },
      [
        ...(!isReadonly && recipe.name !== DEFAULT_RECIPE_NAME
          ? [
              {
                label: 'Build recipe',
                tone: 'primary' as const,
                onClick: () => navigate(`/config/signals?recipe=${encodeURIComponent(recipe.name)}`),
              },
              {
                label: 'Delete recipe',
                tone: 'destructive' as const,
                onClick: () => {
                  setDeleteError(getRecipeDeleteBlocker(config, recipe.name))
                  setRecipePendingDelete(recipe)
                },
              },
            ]
          : []),
      ],
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
      setEntrypointPendingDelete(null)
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete entrypoint.')
    } finally {
      setDeletePending(false)
    }
  }

  const confirmDeleteRecipe = async () => {
    if (!recipePendingDelete) return
    const isDraft = draftNames.has(recipePendingDelete.name)
    const blocker = isDraft ? null : getRecipeDeleteBlocker(config, recipePendingDelete.name)
    if (blocker) {
      setDeleteError(blocker)
      return
    }
    setDeletePending(true)
    setDeleteError(null)
    try {
      if (isDraft) {
        await recipeDraftApi.remove(recipePendingDelete.name)
        announceRecipeDraftChange()
        await loadDrafts()
      } else {
        const nextConfig = cloneConfigData(config)
        nextConfig.recipes = (nextConfig.recipes ?? []).filter(
          (recipe) => recipe.name !== recipePendingDelete.name,
        )
        await saveConfig(nextConfig)
      }
      setRecipePendingDelete(null)
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : 'Failed to delete recipe.')
    } finally {
      setDeletePending(false)
    }
  }

  return (
    <>
      <div className={pageStyles.tablesGrid}>
        {activeView === 'models' ? (
          <ConfigPageMoMEntrypointsList
            config={config}
            isReadonly={isReadonly}
            onAdd={() => setMixtureEditor({ index: null })}
            onView={viewEntrypoint}
          />
        ) : null}
        {activeView === 'recipes' ? (
          <ConfigPageMoMRecipesList
            config={combinedConfig}
            draftNames={draftNames}
            models={models}
            isReadonly={isReadonly}
            onAdd={() => {
              setRecipeEditor(undefined)
              setRecipeEditorOpen(true)
            }}
            onView={viewRecipe}
          />
        ) : null}
      </div>

      {recipeEditorOpen ? (
        <ConfigPageRecipeDialog
          recipe={recipeEditor ?? undefined}
          published={Boolean(
            recipeEditor && (config.recipes ?? []).some((item) => item.name === recipeEditor.name),
          )}
          onClose={() => setRecipeEditorOpen(false)}
          onSave={saveRecipe}
        />
      ) : null}

      {mixtureEditor ? (
        <ConfigPageMixtureDialog
          config={combinedConfig}
          models={models}
          entrypoint={mixtureEditor.entrypoint}
          onClose={() => setMixtureEditor(null)}
          onSave={saveMixture}
        />
      ) : null}

      {topologyTarget ? (
        <ConfigPageMoMTopologyDialog
          entrypoint={topologyTarget.entrypoint}
          recipe={topologyTarget.recipe}
          onClose={() => setTopologyTarget(null)}
        />
      ) : null}

      <ConfirmDialog
        isOpen={entrypointPendingDelete !== null}
        title="Delete entrypoint mapping?"
        description="Remove these public model IDs from the router model catalog."
        eyebrow="Public model namespace change"
        confirmLabel="Delete entrypoint"
        pending={deletePending}
        details={deleteError ? <span role="alert">{deleteError}</span> : undefined}
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
        eyebrow="Destructive routing change"
        confirmLabel="Delete recipe"
        confirmationText={recipePendingDelete?.name}
        pending={deletePending}
        details={
          <div className={pageStyles.deleteDetails}>
            <span>
              {collectRecipeTargetModels(recipePendingDelete).length} configured target models
            </span>
            <span>{recipePendingDelete?.routing.decisions?.length ?? 0} recipe decisions</span>
            {deleteError ? <span role="alert">{deleteError}</span> : null}
          </div>
        }
        onCancel={() => {
          if (deletePending) return
          setRecipePendingDelete(null)
          setDeleteError(null)
        }}
        onConfirm={confirmDeleteRecipe}
      />
    </>
  )
}
