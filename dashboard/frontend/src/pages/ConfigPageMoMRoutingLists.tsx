import { useEffect, useMemo, useState } from 'react'

import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import {
  countRecipeEntrypoints,
  DEFAULT_RECIPE_NAME,
  getRecipeByName,
  getRecipeReadiness,
} from './configPageEntrypointsRecipesSupport'
import type {
  ConfigData,
  EntrypointConfig,
  NormalizedModel,
  RecipeConfig,
} from './configPageSupport'

const PAGE_SIZE = 8

interface EntrypointsListProps {
  config: ConfigData
  isReadonly: boolean
  onAdd: () => void
  onView: (entrypoint: EntrypointConfig, index: number) => void
  onEdit: (entrypoint: EntrypointConfig, index: number) => void
  onDelete: (entrypoint: EntrypointConfig, index: number) => void
  onTopology: (entrypoint: EntrypointConfig, recipe: RecipeConfig) => void
}

interface RecipesListProps {
  config: ConfigData
  draftNames?: Set<string>
  models: NormalizedModel[]
  isReadonly: boolean
  onAdd: () => void
  onView: (recipe: RecipeConfig) => void
  onEdit: (recipe: RecipeConfig) => void
  onDelete: (recipe: RecipeConfig) => void
  onBuild: (recipe: RecipeConfig) => void
}

function Pager({
  page,
  count,
  onChange,
}: {
  page: number
  count: number
  onChange: (page: number) => void
}) {
  const pages = Math.max(1, Math.ceil(count / PAGE_SIZE))
  if (pages <= 1) return null
  return (
    <div className={pageStyles.pager} aria-label="Pagination">
      <span>
        {page + 1} / {pages}
      </span>
      <button type="button" disabled={page === 0} onClick={() => onChange(page - 1)}>
        Previous
      </button>
      <button type="button" disabled={page >= pages - 1} onClick={() => onChange(page + 1)}>
        Next
      </button>
    </div>
  )
}

const uniqueBoundModels = (entrypoint: EntrypointConfig) =>
  new Set(
    Object.values(entrypoint.model_bindings ?? {})
      .flat()
      .map((reference) => reference.model),
  ).size

export function ConfigPageMoMEntrypointsList({
  config,
  isReadonly,
  onAdd,
  onView,
  onEdit,
  onDelete,
  onTopology,
}: EntrypointsListProps) {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const entrypoints = config.entrypoints ?? []
  const query = search.trim().toLowerCase()
  const filtered = entrypoints.filter(
    (entrypoint) =>
      !query ||
      entrypoint.recipe.toLowerCase().includes(query) ||
      entrypoint.model_names.some((name) => name.toLowerCase().includes(query)) ||
      Object.values(entrypoint.model_bindings ?? {}).some((refs) =>
        refs.some((reference) => reference.model.toLowerCase().includes(query)),
      ),
  )
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  useEffect(() => setPage(0), [search])

  return (
    <section className={pageStyles.portfolioPanel}>
      <div className={pageStyles.portfolioHeader}>
        <div>
          <span className={pageStyles.sectionEyebrow}>Ready to call</span>
          <h2>Models</h2>
          <p>One model name. One recipe. Your choice of models.</p>
        </div>
        <div className={pageStyles.portfolioActions}>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search"
            aria-label="Search Mixture-of-Models"
          />
          {!isReadonly ? (
            <button type="button" onClick={onAdd}>
              Create model
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {visible.map((entrypoint) => {
          const key = entrypoint.model_names.join('|')
          const recipe = getRecipeByName(config, entrypoint.recipe)
          const originalIndex = entrypoints.indexOf(entrypoint)
          const boundModelCount = uniqueBoundModels(entrypoint)
          const decisions =
            Object.keys(entrypoint.model_bindings ?? {}).length ||
            recipe?.routing.decisions?.length ||
            0
          return (
            <article key={key} className={pageStyles.portfolioItem}>
              <div
                className={`${pageStyles.portfolioItemMain} ${pageStyles.staticPortfolioItemMain}`}
              >
                <div className={pageStyles.portfolioIdentity}>
                  <strong>{entrypoint.model_names[0]}</strong>
                  <span>
                    {entrypoint.model_names.length > 1
                      ? `+${entrypoint.model_names.length - 1} aliases`
                      : 'OpenAI-compatible model'}
                  </span>
                </div>
                <div className={pageStyles.portfolioMeta}>
                  <span>{entrypoint.recipe}</span>
                  <span>{decisions} decisions</span>
                  <span>
                    {boundModelCount} model{boundModelCount === 1 ? '' : 's'}
                  </span>
                </div>
                <div className={pageStyles.rowActions}>
                  {recipe ? (
                    <button type="button" onClick={() => onTopology(entrypoint, recipe)}>
                      Topology
                    </button>
                  ) : null}
                  <button type="button" onClick={() => onView(entrypoint, originalIndex)}>
                    View
                  </button>
                  {!isReadonly ? (
                    <>
                      <button type="button" onClick={() => onEdit(entrypoint, originalIndex)}>
                        Edit
                      </button>
                      <button
                        type="button"
                        className={pageStyles.deleteAction}
                        onClick={() => onDelete(entrypoint, originalIndex)}
                      >
                        Delete
                      </button>
                    </>
                  ) : null}
                </div>
              </div>
              {entrypoint.model_bindings && Object.keys(entrypoint.model_bindings).length > 0 ? (
                <div className={pageStyles.bindingStrip}>
                  {Object.entries(entrypoint.model_bindings).map(([decision, refs]) => (
                    <span key={decision}>
                      <strong>{decision}</strong>
                      {refs.length} model{refs.length === 1 ? '' : 's'}
                    </span>
                  ))}
                </div>
              ) : null}
            </article>
          )
        })}
        {visible.length === 0 ? (
          <div className={pageStyles.emptyState}>
            {search ? 'No matches.' : 'Create your first model.'}
          </div>
        ) : null}
      </div>
      <Pager page={page} count={filtered.length} onChange={setPage} />
    </section>
  )
}

export function ConfigPageMoMRecipesList({
  config,
  draftNames = new Set(),
  models,
  isReadonly,
  onAdd,
  onView,
  onEdit,
  onDelete,
  onBuild,
}: RecipesListProps) {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const recipes = useMemo(() => {
    const inferredDefault = getRecipeByName(config, DEFAULT_RECIPE_NAME)
    return [
      ...(inferredDefault ? [inferredDefault] : []),
      ...(config.recipes ?? []).filter((recipe) => recipe.name !== DEFAULT_RECIPE_NAME),
    ]
  }, [config])
  const query = search.trim().toLowerCase()
  const filtered = recipes.filter(
    (recipe) =>
      !query ||
      recipe.name.toLowerCase().includes(query) ||
      recipe.description?.toLowerCase().includes(query),
  )
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  useEffect(() => setPage(0), [search])

  return (
    <section className={pageStyles.portfolioPanel}>
      <div className={pageStyles.portfolioHeader}>
        <div>
          <span className={pageStyles.sectionEyebrow}>How models work together</span>
          <h2>Recipes</h2>
          <p>Start with a built-in recipe or create your own.</p>
        </div>
        <div className={pageStyles.portfolioActions}>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search"
            aria-label="Search recipes"
          />
          {!isReadonly ? (
            <button type="button" onClick={onAdd}>
              Create recipe
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {visible.map((recipe) => {
          const isDraft = draftNames.has(recipe.name)
          const readiness = getRecipeReadiness(recipe, models)
          const modelCount = countRecipeEntrypoints(config.entrypoints ?? [], recipe.name)
          const signalCount = Object.values(recipe.routing.signals ?? {}).reduce(
            (count, value) => count + (Array.isArray(value) ? value.length : 0),
            0,
          )
          return (
            <article key={recipe.name} className={pageStyles.portfolioItem}>
              <div
                className={`${pageStyles.portfolioItemMain} ${pageStyles.staticPortfolioItemMain}`}
              >
                <div className={pageStyles.portfolioIdentity}>
                  <div className={pageStyles.recipeTitle}>
                    <strong>{recipe.name}</strong>
                    <span
                      className={
                        recipe.name === DEFAULT_RECIPE_NAME
                          ? pageStyles.defaultBadge
                          : pageStyles.recipeBadge
                      }
                    >
                      {recipe.name === DEFAULT_RECIPE_NAME ? 'Default' : 'Recipe'}
                    </span>
                  </div>
                  <span>{recipe.description || 'Custom model composition'}</span>
                </div>
                <div className={pageStyles.portfolioMeta}>
                  <span>{isDraft ? (readiness.ready ? 'Ready' : 'Draft') : 'Live'}</span>
                  <span>
                    {modelCount} model{modelCount === 1 ? '' : 's'}
                  </span>
                  <span>{signalCount} signals</span>
                  <span>{recipe.routing.decisions?.length ?? 0} decisions</span>
                </div>
                <div className={pageStyles.rowActions}>
                  <button type="button" onClick={() => onView(recipe)}>
                    View
                  </button>
                  {!isReadonly ? (
                    <>
                      {recipe.name !== DEFAULT_RECIPE_NAME ? (
                        <button type="button" onClick={() => onBuild(recipe)}>
                          Build
                        </button>
                      ) : null}
                      <button type="button" onClick={() => onEdit(recipe)}>
                        Details
                      </button>
                      {recipe.name !== DEFAULT_RECIPE_NAME ? (
                        <button
                          type="button"
                          className={pageStyles.deleteAction}
                          onClick={() => onDelete(recipe)}
                        >
                          Delete
                        </button>
                      ) : null}
                    </>
                  ) : null}
                </div>
              </div>
            </article>
          )
        })}
        {visible.length === 0 ? (
          <div className={pageStyles.emptyState}>{search ? 'No matches.' : 'No recipes yet.'}</div>
        ) : null}
      </div>
      <Pager page={page} count={filtered.length} onChange={setPage} />
    </section>
  )
}
