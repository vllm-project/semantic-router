import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import ConfirmDialog from '../components/ConfirmDialog'
import type { FieldConfig } from '../components/EditModal'
import ConfigPageMixtureDialog from './ConfigPageMixtureDialog'
import ConfigPageModelUsageDialog from './ConfigPageModelUsageDialog'
import ConfigPageMoMTopologyDialog from './ConfigPageMoMTopologyDialog'
import { ConfigPageMoMEntrypointsList, ConfigPageMoMRecipesList } from './ConfigPageMoMRoutingLists'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import { cloneConfigData } from './configPageCanonicalization'
import {
  collectRecipeTargetModels,
  countRecipeEntrypoints,
  getRecipeDeleteBlocker,
  normalizeRecipeStrategy,
  type RecipeFormState,
  validateRecipeForm,
} from './configPageEntrypointsRecipesSupport'
import type {
  ConfigData,
  EntrypointConfig,
  NormalizedModel,
  RecipeConfig,
} from './configPageSupport'
import { DEFAULT_ROUTING_STRATEGY, ROUTING_STRATEGIES } from './configPageSupport'
import type { OpenEditModal, OpenViewModal } from './configPageRouterSectionSupport'
import type { MixtureWorkspaceView } from './ConfigPageEntrypointsRecipesSection'
import {
  countProjectionsInProfile,
  countSignalsInProfile,
  type RoutingProfileLike,
} from '../utils/routingScopes'

interface ConfigPageMoMRoutingPanelProps {
  activeView: Exclude<MixtureWorkspaceView, 'probes'>
  config: ConfigData
  isReadonly: boolean
  models: NormalizedModel[]
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
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
  openEditModal,
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
    index?: number
  } | null>(null)
  const [usageTarget, setUsageTarget] = useState<EntrypointConfig | null>(null)

  const entrypoints = config.entrypoints ?? []

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
      },
      mode,
    )
  }

  const viewRecipe = (recipe: RecipeConfig) => {
    const targets = collectRecipeTargetModels(recipe)
    openViewModal(
      recipe.name,
      [
        {
          title: 'Recipe profile',
          fields: [
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
      isReadonly ? undefined : () => openRecipeEditor('edit', recipe),
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
            onAdd={() => setMixtureEditor({})}
            onUsage={setUsageTarget}
            onEvaluate={(entrypoint) => {
              const model = entrypoint.model_names[0]
              if (!model) return
              navigate(`/evaluation?view=new&entrypoint=${encodeURIComponent(model)}`)
            }}
            onEdit={(entrypoint, index) => setMixtureEditor({ entrypoint, index })}
            onDelete={(entrypoint, index) => {
              setDeleteError(null)
              setEntrypointPendingDelete({ entrypoint, index })
            }}
            onTopology={(entrypoint, recipe) => setTopologyTarget({ entrypoint, recipe })}
          />
        ) : (
          <ConfigPageMoMRecipesList
            config={config}
            isReadonly={isReadonly}
            onAdd={() => openRecipeEditor('add')}
            onView={viewRecipe}
            onEdit={(recipe) => openRecipeEditor('edit', recipe)}
            onDelete={(recipe) => {
              setDeleteError(getRecipeDeleteBlocker(config, recipe.name))
              setRecipePendingDelete(recipe)
            }}
          />
        )}
      </div>

      {mixtureEditor ? (
        <ConfigPageMixtureDialog
          config={config}
          models={models}
          entrypoint={mixtureEditor.entrypoint}
          entrypointIndex={mixtureEditor.index}
          onClose={() => setMixtureEditor(null)}
          onSave={saveConfig}
        />
      ) : null}

      {topologyTarget ? (
        <ConfigPageMoMTopologyDialog
          entrypoint={topologyTarget.entrypoint}
          recipe={topologyTarget.recipe}
          onClose={() => setTopologyTarget(null)}
        />
      ) : null}

      {usageTarget ? (
        <ConfigPageModelUsageDialog entrypoint={usageTarget} onClose={() => setUsageTarget(null)} />
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
