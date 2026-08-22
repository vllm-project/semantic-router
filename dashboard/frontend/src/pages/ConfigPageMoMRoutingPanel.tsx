import { useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import type { FieldConfig } from '../components/EditModal'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import ConfigPageMoMTopologyDialog from './ConfigPageMoMTopologyDialog'
import { ConfigPageMoMEntrypointsList, ConfigPageMoMRecipesList } from './ConfigPageMoMRoutingLists'
import ConfigPageRecipeDecisionsEditor from './ConfigPageRecipeDecisionsEditor'
import ConfigPageRecipePolicyEditor from './ConfigPageRecipePolicyEditor'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import { cloneConfigData } from './configPageCanonicalization'
import {
  collectRecipeTargetModels,
  countRecipeEntrypoints,
  DEFAULT_RECIPE_NAME,
  getRecipeByName,
  getRecipeDeleteBlocker,
  getRecipeNames,
  normalizeRecipeStrategy,
  type EntrypointFormState,
  type RecipeFormState,
  validateEntrypointForm,
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
import {
  countProjectionsInProfile,
  countSignalsInProfile,
  type RoutingProfileLike,
} from '../utils/routingScopes'

interface ConfigPageMoMRoutingPanelProps {
  config: ConfigData
  isReadonly: boolean
  models: NormalizedModel[]
  saveConfig: (config: ConfigData) => Promise<void>
  openEditModal: OpenEditModal
  openViewModal: OpenViewModal
  catalog: BuiltInModelCatalog | null
  catalogLoading: boolean
  catalogError: string | null
  onCatalogRetry: () => void
}

interface PendingEntrypointDelete {
  entrypoint: EntrypointConfig
  index: number
}

const cloneDecisions = (recipe: RecipeConfig): NonNullable<RecipeConfig['routing']['decisions']> =>
  JSON.parse(JSON.stringify(recipe.routing.decisions ?? []))

const cloneSignals = (recipe?: RecipeConfig): NonNullable<RecipeConfig['routing']['signals']> =>
  JSON.parse(JSON.stringify(recipe?.routing.signals ?? {}))

export default function ConfigPageMoMRoutingPanel({
  config,
  isReadonly,
  models,
  saveConfig,
  openEditModal,
  openViewModal,
  catalog,
  catalogLoading,
  catalogError,
  onCatalogRetry,
}: ConfigPageMoMRoutingPanelProps) {
  const [entrypointPendingDelete, setEntrypointPendingDelete] =
    useState<PendingEntrypointDelete | null>(null)
  const [recipePendingDelete, setRecipePendingDelete] = useState<RecipeConfig | null>(null)
  const [deletePending, setDeletePending] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [topologyTarget, setTopologyTarget] = useState<{
    entrypoint: EntrypointConfig
    recipe: RecipeConfig
  } | null>(null)

  const entrypoints = config.entrypoints ?? []

  const openEntrypointEditor = (
    mode: 'add' | 'edit',
    entrypoint?: EntrypointConfig,
    originalIndex: number | null = null,
  ) => {
    const form: EntrypointFormState = {
      modelNames: entrypoint?.model_names.join('\n') ?? '',
      recipe: entrypoint?.recipe ?? DEFAULT_RECIPE_NAME,
    }
    const fields: FieldConfig<EntrypointFormState>[] = [
      {
        name: 'modelNames',
        label: 'Public model names',
        type: 'textarea',
        required: true,
        placeholder: 'team/custom-model',
        description: 'One virtual model ID per line. These names appear in /v1/models.',
      },
      {
        name: 'recipe',
        label: 'Routing recipe',
        type: 'select',
        required: true,
        options: getRecipeNames(config),
        description: 'Requests using any model above evaluate only this recipe.',
      },
    ]
    openEditModal(
      mode === 'add' ? 'Add Entrypoint' : 'Edit Entrypoint',
      form,
      fields,
      async (data) => {
        const normalized = validateEntrypointForm(data, config, models, originalIndex)
        const nextConfig = cloneConfigData(config)
        const nextEntrypoints = [...(nextConfig.entrypoints ?? [])]
        if (originalIndex === null) nextEntrypoints.push(normalized)
        else nextEntrypoints[originalIndex] = normalized
        nextConfig.entrypoints = nextEntrypoints
        await saveConfig(nextConfig)
      },
      mode,
    )
  }

  const openRecipeEditor = (mode: 'add' | 'edit', recipe?: RecipeConfig) => {
    const originalName = recipe?.name ?? null
    const form: RecipeFormState = {
      name: recipe?.name ?? '',
      description: recipe?.description ?? '',
      strategy: normalizeRecipeStrategy(
        recipe?.routing.strategy ?? config.global?.router?.strategy,
      ),
      signals: cloneSignals(recipe),
      decisions: recipe ? cloneDecisions(recipe) : [],
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
      {
        name: 'signals',
        label: 'Recipe policy signals',
        type: 'custom',
        description: 'Author metadata and classifier signals inside this isolated recipe.',
        customRender: (value, onChange) => (
          <ConfigPageRecipePolicyEditor
            value={value && typeof value === 'object' ? value : {}}
            onChange={(nextValue) => onChange(nextValue)}
          />
        ),
      },
      {
        name: 'decisions',
        label: 'Decision model routes',
        type: 'custom',
        description: 'Manage every decision and its complete target model reference data.',
        customRender: (value, onChange) => (
          <ConfigPageRecipeDecisionsEditor
            value={Array.isArray(value) ? value : []}
            models={models}
            onChange={(nextValue) => onChange(nextValue)}
          />
        ),
      },
    ]
    openEditModal(
      mode === 'add' ? 'Add Routing Recipe' : `Edit Recipe · ${recipe?.name ?? ''}`,
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

  const viewEntrypoint = (entrypoint: EntrypointConfig, index: number) => {
    const recipe = getRecipeByName(config, entrypoint.recipe)
    const targets = collectRecipeTargetModels(recipe)
    openViewModal(
      entrypoint.model_names.join(', '),
      [
        {
          title: 'Entrypoint mapping',
          fields: [
            { label: 'Public models', value: entrypoint.model_names.join('\n'), fullWidth: true },
            { label: 'Recipe', value: entrypoint.recipe },
            { label: 'Recipe decisions', value: recipe?.routing.decisions?.length ?? 0 },
            {
              label: 'Configured targets',
              value: targets.join('\n') || 'No target models',
              fullWidth: true,
            },
          ],
        },
      ],
      isReadonly ? undefined : () => openEntrypointEditor('edit', entrypoint, index),
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
        <ConfigPageMoMEntrypointsList
          config={config}
          isReadonly={isReadonly}
          onAdd={() => openEntrypointEditor('add')}
          onView={viewEntrypoint}
          onEdit={(entrypoint, index) => openEntrypointEditor('edit', entrypoint, index)}
          onDelete={(entrypoint, index) => {
            setDeleteError(null)
            setEntrypointPendingDelete({ entrypoint, index })
          }}
          onTopology={(entrypoint, recipe) => setTopologyTarget({ entrypoint, recipe })}
          catalog={catalog}
          catalogLoading={catalogLoading}
          catalogError={catalogError}
          onCatalogRetry={onCatalogRetry}
        />
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
      </div>

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
