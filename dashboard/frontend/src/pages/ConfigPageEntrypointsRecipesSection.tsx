import { useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import type { FieldConfig } from '../components/EditModal'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import ConfigPageMoMTopologyDialog from './ConfigPageMoMTopologyDialog'
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

interface ConfigPageEntrypointsRecipesSectionProps {
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

const cloneDecisions = (recipe: RecipeConfig): NonNullable<RecipeConfig['routing']['decisions']> =>
  JSON.parse(JSON.stringify(recipe.routing.decisions ?? []))

const cloneSignals = (recipe?: RecipeConfig): NonNullable<RecipeConfig['routing']['signals']> =>
  JSON.parse(JSON.stringify(recipe?.routing.signals ?? {}))

export default function ConfigPageEntrypointsRecipesSection({
  config,
  isReadonly,
  models,
  saveConfig,
  openEditModal,
  openViewModal,
}: ConfigPageEntrypointsRecipesSectionProps) {
  const [entrypointsSearch, setEntrypointsSearch] = useState('')
  const [recipesSearch, setRecipesSearch] = useState('')
  const [expandedEntrypoints, setExpandedEntrypoints] = useState<Set<string>>(new Set())
  const [expandedRecipes, setExpandedRecipes] = useState<Set<string>>(new Set())
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
  const recipes = config.recipes ?? []

  const filteredEntrypoints = entrypoints.filter((entrypoint) => {
    const query = entrypointsSearch.trim().toLowerCase()
    return (
      !query ||
      entrypoint.recipe.toLowerCase().includes(query) ||
      entrypoint.model_names.some((modelName) => modelName.toLowerCase().includes(query))
    )
  })
  const filteredRecipes = recipes.filter((recipe) => {
    const query = recipesSearch.trim().toLowerCase()
    return (
      !query ||
      recipe.name.toLowerCase().includes(query) ||
      recipe.description?.toLowerCase().includes(query) ||
      collectRecipeTargetModels(recipe).some((modelName) => modelName.toLowerCase().includes(query))
    )
  })

  const openEntrypointEditor = (
    mode: 'add' | 'edit',
    entrypoint?: EntrypointConfig,
    originalIndex: number | null = null,
  ) => {
    const form: EntrypointFormState = {
      modelNames: entrypoint?.model_names.join('\n') ?? 'vllm-sr/mom-',
      recipe: entrypoint?.recipe ?? DEFAULT_RECIPE_NAME,
    }
    const fields: FieldConfig<EntrypointFormState>[] = [
      {
        name: 'modelNames',
        label: 'Public model names',
        type: 'textarea',
        required: true,
        placeholder: 'vllm-sr/mom-balanced-v1',
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
              label: 'Physical targets',
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
              label: 'Physical targets',
              value: targets.join('\n') || 'No target models',
              fullWidth: true,
            },
          ],
        },
      ],
      isReadonly ? undefined : () => openRecipeEditor('edit', recipe),
    )
  }

  const renderRecipeModelPool = (recipe: RecipeConfig | null) => {
    if (!recipe) {
      return <div className={pageStyles.poolDisclosure}>Recipe not found.</div>
    }
    const decisions = recipe.routing.decisions ?? []
    return (
      <div className={pageStyles.poolDisclosure}>
        <div className={pageStyles.poolDisclosureHeader}>
          <div>
            <span className={pageStyles.metricLabel}>Mixture composition</span>
            <strong>{recipe.name}</strong>
          </div>
          <span className={pageStyles.recipeBadge}>
            {collectRecipeTargetModels(recipe).length} physical models
          </span>
        </div>
        {decisions.length > 0 ? (
          <div className={pageStyles.poolDecisionGrid}>
            {decisions.map((decision) => (
              <article key={decision.name} className={pageStyles.poolDecisionCard}>
                <div className={pageStyles.poolDecisionHeader}>
                  <strong>{decision.name}</strong>
                  <span>P{decision.priority}</span>
                </div>
                <p>{decision.description || 'No decision description'}</p>
                <div className={pageStyles.poolModelList}>
                  {(decision.modelRefs ?? []).map((reference) => (
                    <div
                      key={`${decision.name}-${reference.model}`}
                      className={pageStyles.poolModel}
                    >
                      <span className={pageStyles.poolModelDot} aria-hidden="true" />
                      <div>
                        <code>{reference.model}</code>
                        <small>
                          {reference.use_reasoning ? 'Reasoning' : 'Standard'}
                          {reference.reasoning_effort ? ` · ${reference.reasoning_effort}` : ''}
                          {typeof reference.weight === 'number'
                            ? ` · weight ${reference.weight}`
                            : ''}
                        </small>
                      </div>
                    </div>
                  ))}
                </div>
              </article>
            ))}
          </div>
        ) : (
          <span className={pageStyles.muted}>No decisions configured for this recipe.</span>
        )}
      </div>
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
    <ConfigPageManagerLayout
      eyebrow="Dispatch"
      title="Mixture-of-Models"
      description="Compose branded AMD models from reusable routing recipes and purpose-built model pools."
      configArea="Multi-recipe dispatch"
      scope="Live model namespace"
      panelEyebrow="Your model portfolio"
      panelTitle="Models you design and own"
      panelDescription="Every public model is a deliberate mixture: a routing policy, a decision graph, and the AMD models behind it."
      pills={[
        { label: 'Models' },
        { label: 'Decisions' },
        { label: 'Mixture-of-Models', active: true },
      ]}
    >
      <div className={pageStyles.tablesGrid}>
        <section className={pageStyles.portfolioPanel}>
          <div className={pageStyles.portfolioHeader}>
            <div>
              <span className={pageStyles.sectionEyebrow}>Your models</span>
              <h2>Entrypoints</h2>
              <p>Customer-facing model IDs mapped to isolated routing recipes.</p>
            </div>
            <div className={pageStyles.portfolioActions}>
              <input
                type="search"
                value={entrypointsSearch}
                onChange={(event) => setEntrypointsSearch(event.target.value)}
                placeholder="Search model or recipe"
                aria-label="Search entrypoints"
              />
              {!isReadonly ? (
                <button type="button" onClick={() => openEntrypointEditor('add')}>
                  Add entrypoint
                </button>
              ) : null}
            </div>
          </div>
          <div className={pageStyles.portfolioList}>
            {filteredEntrypoints.map((entrypoint) => {
              const key = entrypoint.model_names.join('|')
              const expanded = expandedEntrypoints.has(key)
              const mappedRecipe = getRecipeByName(config, entrypoint.recipe)
              const targetCount = collectRecipeTargetModels(mappedRecipe).length
              return (
                <article key={key} className={pageStyles.portfolioItem}>
                  <div className={pageStyles.portfolioItemMain}>
                    <button
                      type="button"
                      className={pageStyles.disclosureButton}
                      aria-expanded={expanded}
                      aria-label={`${expanded ? 'Collapse' : 'Expand'} ${entrypoint.model_names.join(', ')}`}
                      onClick={() => {
                        setExpandedEntrypoints((current) => {
                          const next = new Set(current)
                          if (next.has(key)) next.delete(key)
                          else next.add(key)
                          return next
                        })
                      }}
                    >
                      <span aria-hidden="true">{expanded ? '−' : '+'}</span>
                    </button>
                    <div className={pageStyles.portfolioIdentity}>
                      {entrypoint.model_names.map((modelName) => (
                        <code key={modelName}>{modelName}</code>
                      ))}
                      <span>Routes through {entrypoint.recipe}</span>
                    </div>
                    <div className={pageStyles.portfolioMeta}>
                      <span>{targetCount} models</span>
                      <span>{entrypoint.recipe}</span>
                    </div>
                    <div className={pageStyles.rowActions}>
                      {mappedRecipe ? (
                        <button
                          type="button"
                          onClick={() => setTopologyTarget({ entrypoint, recipe: mappedRecipe })}
                        >
                          Topology
                        </button>
                      ) : null}
                      <button
                        type="button"
                        onClick={() => viewEntrypoint(entrypoint, entrypoints.indexOf(entrypoint))}
                      >
                        View
                      </button>
                      {!isReadonly ? (
                        <>
                          <button
                            type="button"
                            onClick={() =>
                              openEntrypointEditor(
                                'edit',
                                entrypoint,
                                entrypoints.indexOf(entrypoint),
                              )
                            }
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            className={pageStyles.deleteAction}
                            onClick={() => {
                              setDeleteError(null)
                              setEntrypointPendingDelete({
                                entrypoint,
                                index: entrypoints.indexOf(entrypoint),
                              })
                            }}
                          >
                            Delete
                          </button>
                        </>
                      ) : null}
                    </div>
                  </div>
                  {expanded ? renderRecipeModelPool(mappedRecipe) : null}
                </article>
              )
            })}
            {filteredEntrypoints.length === 0 ? (
              <div className={pageStyles.emptyState}>
                {entrypointsSearch
                  ? 'No entrypoints match your search.'
                  : 'No entrypoints configured.'}
              </div>
            ) : null}
          </div>
        </section>

        <section className={pageStyles.portfolioPanel}>
          <div className={pageStyles.portfolioHeader}>
            <div>
              <span className={pageStyles.sectionEyebrow}>Reusable routing</span>
              <h2>Recipes</h2>
              <p>Reusable decision graphs and the physical model pools behind them.</p>
            </div>
            <div className={pageStyles.portfolioActions}>
              <input
                type="search"
                value={recipesSearch}
                onChange={(event) => setRecipesSearch(event.target.value)}
                placeholder="Search recipe or model"
                aria-label="Search recipes"
              />
              {!isReadonly ? (
                <button type="button" onClick={() => openRecipeEditor('add')}>
                  Add recipe
                </button>
              ) : null}
            </div>
          </div>
          <div className={pageStyles.portfolioList}>
            {filteredRecipes.map((recipe) => {
              const expanded = expandedRecipes.has(recipe.name)
              return (
                <article key={recipe.name} className={pageStyles.portfolioItem}>
                  <div className={pageStyles.portfolioItemMain}>
                    <button
                      type="button"
                      className={pageStyles.disclosureButton}
                      aria-expanded={expanded}
                      aria-label={`${expanded ? 'Collapse' : 'Expand'} ${recipe.name}`}
                      onClick={() => {
                        setExpandedRecipes((current) => {
                          const next = new Set(current)
                          if (next.has(recipe.name)) next.delete(recipe.name)
                          else next.add(recipe.name)
                          return next
                        })
                      }}
                    >
                      <span aria-hidden="true">{expanded ? '−' : '+'}</span>
                    </button>
                    <div className={pageStyles.portfolioIdentity}>
                      <strong>{recipe.name}</strong>
                      <span>{recipe.description || 'No description'}</span>
                    </div>
                    <div className={pageStyles.portfolioMeta}>
                      <span>{countRecipeEntrypoints(entrypoints, recipe.name)} public models</span>
                      <span>{recipe.routing.decisions?.length ?? 0} decisions</span>
                      <span>{collectRecipeTargetModels(recipe).length} models</span>
                    </div>
                    <div className={pageStyles.rowActions}>
                      <button type="button" onClick={() => viewRecipe(recipe)}>
                        View
                      </button>
                      {!isReadonly ? (
                        <>
                          <button type="button" onClick={() => openRecipeEditor('edit', recipe)}>
                            Edit
                          </button>
                          <button
                            type="button"
                            className={pageStyles.deleteAction}
                            onClick={() => {
                              setDeleteError(getRecipeDeleteBlocker(config, recipe.name))
                              setRecipePendingDelete(recipe)
                            }}
                          >
                            Delete
                          </button>
                        </>
                      ) : null}
                    </div>
                  </div>
                  {expanded ? renderRecipeModelPool(recipe) : null}
                </article>
              )
            })}
            {filteredRecipes.length === 0 ? (
              <div className={pageStyles.emptyState}>
                {recipesSearch ? 'No recipes match your search.' : 'No recipes configured.'}
              </div>
            ) : null}
          </div>
        </section>
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
              {collectRecipeTargetModels(recipePendingDelete).length} physical target models
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
    </ConfigPageManagerLayout>
  )
}
