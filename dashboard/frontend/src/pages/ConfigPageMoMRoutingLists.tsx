import { useState } from 'react'

import type { BuiltInModelCatalog } from '../types/modelCatalog'
import ConfigPageMoMEntrypointCatalogMetadata from './ConfigPageMoMEntrypointCatalogMetadata'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import {
  collectRecipeTargetModels,
  countRecipeEntrypoints,
  getRecipeByName,
} from './configPageEntrypointsRecipesSupport'
import type { ConfigData, EntrypointConfig, RecipeConfig } from './configPageSupport'

interface EntrypointsListProps {
  config: ConfigData
  isReadonly: boolean
  onAdd: () => void
  onView: (entrypoint: EntrypointConfig, index: number) => void
  onEdit: (entrypoint: EntrypointConfig, index: number) => void
  onDelete: (entrypoint: EntrypointConfig, index: number) => void
  onTopology: (entrypoint: EntrypointConfig, recipe: RecipeConfig) => void
  catalog: BuiltInModelCatalog | null
  catalogLoading: boolean
  catalogError: string | null
  onCatalogRetry: () => void
}

interface RecipesListProps {
  config: ConfigData
  isReadonly: boolean
  onAdd: () => void
  onView: (recipe: RecipeConfig) => void
  onEdit: (recipe: RecipeConfig) => void
  onDelete: (recipe: RecipeConfig) => void
}

export function ConfigPageMoMEntrypointsList({
  config,
  isReadonly,
  onAdd,
  onView,
  onEdit,
  onDelete,
  onTopology,
  catalog,
  catalogLoading,
  catalogError,
  onCatalogRetry,
}: EntrypointsListProps) {
  const [search, setSearch] = useState('')
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const entrypoints = config.entrypoints ?? []
  const query = search.trim().toLowerCase()
  const filtered = entrypoints.filter(
    (entrypoint) =>
      !query ||
      entrypoint.recipe.toLowerCase().includes(query) ||
      entrypoint.model_names.some((name) => name.toLowerCase().includes(query)),
  )

  return (
    <section className={pageStyles.portfolioPanel}>
      <div className={pageStyles.portfolioHeader}>
        <div>
          <span className={pageStyles.sectionEyebrow}>Request-facing models</span>
          <h2>Unified Models</h2>
          <p>Public model IDs mapped to isolated routing recipes.</p>
        </div>
        <div className={pageStyles.portfolioActions}>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search model or recipe"
            aria-label="Search unified models"
          />
          {!isReadonly ? (
            <button type="button" onClick={onAdd}>
              Add unified model
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {filtered.map((entrypoint) => {
          const key = entrypoint.model_names.join('|')
          const isExpanded = expanded.has(key)
          const recipe = getRecipeByName(config, entrypoint.recipe)
          const targetCount = collectRecipeTargetModels(recipe).length
          const originalIndex = entrypoints.indexOf(entrypoint)
          return (
            <article key={key} className={pageStyles.portfolioItem}>
              <div className={pageStyles.portfolioItemMain}>
                <button
                  type="button"
                  className={pageStyles.disclosureButton}
                  aria-expanded={isExpanded}
                  aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${entrypoint.model_names.join(', ')}`}
                  onClick={() =>
                    setExpanded((current) => {
                      const next = new Set(current)
                      if (next.has(key)) next.delete(key)
                      else next.add(key)
                      return next
                    })
                  }
                >
                  <span aria-hidden="true">{isExpanded ? '−' : '+'}</span>
                </button>
                <div className={pageStyles.portfolioIdentity}>
                  {entrypoint.model_names.map((name) => (
                    <code key={name}>{name}</code>
                  ))}
                  <span>Routes through {entrypoint.recipe}</span>
                </div>
                <div className={pageStyles.portfolioMeta}>
                  <span>{targetCount} models</span>
                  <span>{entrypoint.recipe}</span>
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
              {isExpanded ? (
                <ConfigPageMoMEntrypointCatalogMetadata
                  catalog={catalog}
                  catalogLoading={catalogLoading}
                  catalogError={catalogError}
                  config={config}
                  entrypoint={entrypoint}
                  onCatalogRetry={onCatalogRetry}
                />
              ) : null}
            </article>
          )
        })}
        {filtered.length === 0 ? (
          <div className={pageStyles.emptyState}>
            {search ? 'No unified models match your search.' : 'No unified models configured.'}
          </div>
        ) : null}
      </div>
    </section>
  )
}

export function ConfigPageMoMRecipesList({
  config,
  isReadonly,
  onAdd,
  onView,
  onEdit,
  onDelete,
}: RecipesListProps) {
  const [search, setSearch] = useState('')
  const recipes = config.recipes ?? []
  const query = search.trim().toLowerCase()
  const filtered = recipes.filter(
    (recipe) =>
      !query ||
      recipe.name.toLowerCase().includes(query) ||
      recipe.description?.toLowerCase().includes(query) ||
      collectRecipeTargetModels(recipe).some((name) => name.toLowerCase().includes(query)),
  )

  return (
    <section className={pageStyles.portfolioPanel}>
      <div className={pageStyles.portfolioHeader}>
        <div>
          <span className={pageStyles.sectionEyebrow}>Reusable routing</span>
          <h2>Recipes</h2>
          <p>Reusable decision graphs and the configured model pools behind them.</p>
        </div>
        <div className={pageStyles.portfolioActions}>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search recipe or model"
            aria-label="Search recipes"
          />
          {!isReadonly ? (
            <button type="button" onClick={onAdd}>
              Add recipe
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {filtered.map((recipe) => {
          return (
            <article key={recipe.name} className={pageStyles.portfolioItem}>
              <div
                className={`${pageStyles.portfolioItemMain} ${pageStyles.staticPortfolioItemMain}`}
              >
                <div className={pageStyles.portfolioIdentity}>
                  <strong>{recipe.name}</strong>
                  <span>{recipe.description || 'No description'}</span>
                </div>
                <div className={pageStyles.portfolioMeta}>
                  <span>
                    {countRecipeEntrypoints(config.entrypoints ?? [], recipe.name)} public models
                  </span>
                  <span>{recipe.routing.decisions?.length ?? 0} decisions</span>
                  <span>{collectRecipeTargetModels(recipe).length} models</span>
                </div>
                <div className={pageStyles.rowActions}>
                  <button type="button" onClick={() => onView(recipe)}>
                    View
                  </button>
                  {!isReadonly ? (
                    <>
                      <button type="button" onClick={() => onEdit(recipe)}>
                        Edit
                      </button>
                      <button
                        type="button"
                        className={pageStyles.deleteAction}
                        onClick={() => onDelete(recipe)}
                      >
                        Delete
                      </button>
                    </>
                  ) : null}
                </div>
              </div>
            </article>
          )
        })}
        {filtered.length === 0 ? (
          <div className={pageStyles.emptyState}>
            {search ? 'No recipes match your search.' : 'No recipes configured.'}
          </div>
        ) : null}
      </div>
    </section>
  )
}
