import { useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import {
  collectRecipeTargetModels,
  countRecipeEntrypoints,
  getRecipeByName,
  isSyntheticDefaultRecipe,
  listRecipeProfiles,
} from './configPageEntrypointsRecipesSupport'
import type { ConfigData, EntrypointConfig, RecipeConfig } from './configPageSupport'

interface EntrypointsListProps {
  config: ConfigData
  isReadonly: boolean
  onAdd: () => void
  onUsage: (entrypoint: EntrypointConfig) => void
  onEvaluate: (entrypoint: EntrypointConfig) => void
  onEdit: (entrypoint: EntrypointConfig, index: number) => void
  onDelete: (entrypoint: EntrypointConfig, index: number) => void
  onTopology: (entrypoint: EntrypointConfig, recipe: RecipeConfig) => void
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
  onUsage,
  onEvaluate,
  onEdit,
  onDelete,
  onTopology,
}: EntrypointsListProps) {
  const [search, setSearch] = useState('')
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
          <span className={pageStyles.sectionEyebrow}>Ready to call</span>
          <h2>Models</h2>
          <p>One public model name. One complete recipe.</p>
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
              <ProductIcon name="plus" />
              Create model
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {filtered.map((entrypoint) => {
          const key = entrypoint.model_names.join('|')
          const recipe = getRecipeByName(config, entrypoint.recipe)
          const targetCount = collectRecipeTargetModels(recipe).length
          const originalIndex = entrypoints.indexOf(entrypoint)
          return (
            <article key={key} className={pageStyles.portfolioItem}>
              <div
                className={`${pageStyles.portfolioItemMain} ${pageStyles.entrypointPortfolioItemMain}`}
              >
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
                    <button
                      type="button"
                      aria-label={`View topology for ${entrypoint.model_names.join(', ')}`}
                      onClick={() => onTopology(entrypoint, recipe)}
                    >
                      <ProductIcon name="topology" />
                      Topology
                    </button>
                  ) : null}
                  <button
                    type="button"
                    aria-label={`View usage for ${entrypoint.model_names.join(', ')}`}
                    onClick={() => onUsage(entrypoint)}
                  >
                    <ProductIcon name="link" />
                    Usage
                  </button>
                  <button
                    type="button"
                    aria-label={`Evaluate ${entrypoint.model_names.join(', ')}`}
                    onClick={() => onEvaluate(entrypoint)}
                  >
                    <ProductIcon name="chart" />
                    Evaluate
                  </button>
                  {!isReadonly ? (
                    <>
                      <button
                        type="button"
                        aria-label={`Edit ${entrypoint.model_names.join(', ')}`}
                        onClick={() => onEdit(entrypoint, originalIndex)}
                      >
                        <ProductIcon name="edit" />
                        Edit
                      </button>
                      <button
                        type="button"
                        className={pageStyles.deleteAction}
                        aria-label={`Delete ${entrypoint.model_names.join(', ')}`}
                        onClick={() => onDelete(entrypoint, originalIndex)}
                      >
                        <ProductIcon name="trash" />
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
            {search
              ? 'No matches.'
              : isReadonly
                ? 'No models configured.'
                : 'Create your first model.'}
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
  const recipes = listRecipeProfiles(config)
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
          <span className={pageStyles.sectionEyebrow}>How models work together</span>
          <h2>Recipes</h2>
          <p>Reusable routing intelligence, ready to become a model.</p>
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
              <ProductIcon name="plus" />
              Create recipe
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {filtered.map((recipe) => {
          const syntheticDefault = isSyntheticDefaultRecipe(config, recipe)
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
                  {syntheticDefault ? <span>Built in</span> : null}
                </div>
                <div className={pageStyles.rowActions}>
                  <button
                    type="button"
                    aria-label={`View recipe ${recipe.name}`}
                    onClick={() => onView(recipe)}
                  >
                    <ProductIcon name="eye" />
                    View
                  </button>
                  {!isReadonly && !syntheticDefault ? (
                    <>
                      <button
                        type="button"
                        aria-label={`Edit recipe ${recipe.name}`}
                        onClick={() => onEdit(recipe)}
                      >
                        <ProductIcon name="edit" />
                        Edit
                      </button>
                      <button
                        type="button"
                        className={pageStyles.deleteAction}
                        aria-label={`Delete recipe ${recipe.name}`}
                        onClick={() => onDelete(recipe)}
                      >
                        <ProductIcon name="trash" />
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
          <div className={pageStyles.emptyState}>{search ? 'No matches.' : 'No recipes yet.'}</div>
        ) : null}
      </div>
    </section>
  )
}
