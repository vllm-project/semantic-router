import { useEffect, useMemo, useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import type { RoutingEntrypoint, RoutingRecipe } from '../utils/routingManagementApi'
import pageStyles from './ConfigPageEntrypointsRecipesSection.module.css'
import { assignedModelCount } from './configPageMoMRoutingListSupport'
import { recipeDocumentSummary } from './configPageRecipeDialogSupport'

const PAGE_SIZE = 8

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
        <ProductIcon name="chevron-left" />
        Previous
      </button>
      <button type="button" disabled={page >= pages - 1} onClick={() => onChange(page + 1)}>
        Next
        <ProductIcon name="chevron-right" />
      </button>
    </div>
  )
}

function OpenCue({ topology = false }: { topology?: boolean }) {
  return (
    <span className={pageStyles.portfolioOpenCue} aria-hidden="true">
      <ProductIcon name={topology ? 'topology' : 'chevron-right'} />
      <span>{topology ? 'Topology' : 'Details'}</span>
    </span>
  )
}

export function ConfigPageMoMEntrypointsList({
  entrypoints,
  canManage,
  onAdd,
  onView,
}: {
  entrypoints: RoutingEntrypoint[]
  canManage: boolean
  onAdd: () => void
  onView: (entrypoint: RoutingEntrypoint) => void
}) {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const query = search.trim().toLowerCase()
  const filtered = entrypoints.filter(
    (entrypoint) =>
      !query ||
      entrypoint.name.toLowerCase().includes(query) ||
      entrypoint.aliases.some((alias) => alias.toLowerCase().includes(query)),
  )
  const visible = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  useEffect(() => setPage(0), [search])

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
          {canManage ? (
            <button type="button" onClick={onAdd}>
              <ProductIcon name="plus" />
              Create model
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {visible.map((entrypoint) => {
          const modelCount = assignedModelCount(entrypoint)
          return (
            <article key={entrypoint.id} className={pageStyles.portfolioItem}>
              <div
                className={`${pageStyles.portfolioItemMain} ${pageStyles.staticPortfolioItemMain} ${pageStyles.portfolioItemOpenable}`}
                role="button"
                tabIndex={0}
                aria-label={`Open ${entrypoint.name}`}
                onClick={() => onView(entrypoint)}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return
                  event.preventDefault()
                  onView(entrypoint)
                }}
              >
                <div className={pageStyles.portfolioIdentity}>
                  <strong>{entrypoint.aliases[0] ?? entrypoint.name}</strong>
                  <span>
                    {entrypoint.aliases.length > 1
                      ? `+${entrypoint.aliases.length - 1} aliases`
                      : 'OpenAI-compatible model'}
                  </span>
                </div>
                <div className={pageStyles.portfolioMeta}>
                  <span>{entrypoint.status === 'active' ? 'Live' : 'Draft'}</span>
                  <span>
                    {modelCount} model{modelCount === 1 ? '' : 's'}
                  </span>
                  <span>
                    {entrypoint.ruleCount} rule
                    {entrypoint.ruleCount === 1 ? '' : 's'}
                  </span>
                  <span>Revision {entrypoint.entrypointRevision}</span>
                </div>
                <OpenCue topology />
              </div>
            </article>
          )
        })}
        {visible.length === 0 ? (
          <div className={pageStyles.emptyState}>
            {search
              ? 'No matches.'
              : canManage
                ? 'Create your first model.'
                : 'No models configured.'}
          </div>
        ) : null}
      </div>
      <Pager page={page} count={filtered.length} onChange={setPage} />
    </section>
  )
}

export function ConfigPageMoMRecipesList({
  recipes,
  canManage,
  onAdd,
  onView,
}: {
  recipes: RoutingRecipe[]
  canManage: boolean
  onAdd: () => void
  onView: (recipe: RoutingRecipe) => void
}) {
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const ordered = useMemo(
    () => [...recipes].sort((a, b) => a.name.localeCompare(b.name)),
    [recipes],
  )
  const query = search.trim().toLowerCase()
  const filtered = ordered.filter(
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
          {canManage ? (
            <button type="button" onClick={onAdd}>
              <ProductIcon name="plus" />
              Create recipe
            </button>
          ) : null}
        </div>
      </div>
      <div className={pageStyles.portfolioList}>
        {visible.map((recipe) => {
          const summary = recipeDocumentSummary(recipe)
          return (
            <article key={recipe.id} className={pageStyles.portfolioItem}>
              <div
                className={`${pageStyles.portfolioItemMain} ${pageStyles.staticPortfolioItemMain} ${pageStyles.portfolioItemOpenable}`}
                role="button"
                tabIndex={0}
                aria-label={`Open recipe ${recipe.name}`}
                onClick={() => onView(recipe)}
                onKeyDown={(event) => {
                  if (event.key !== 'Enter' && event.key !== ' ') return
                  event.preventDefault()
                  onView(recipe)
                }}
              >
                <div className={pageStyles.portfolioIdentity}>
                  <div className={pageStyles.recipeTitle}>
                    <strong>{recipe.name}</strong>
                    <span className={pageStyles.recipeBadge}>
                      {recipe.immutable ? 'Built-in' : 'Recipe'}
                    </span>
                  </div>
                  <span>{recipe.description || 'Custom model composition'}</span>
                </div>
                <div className={pageStyles.portfolioMeta}>
                  <span>{recipe.status === 'active' ? 'Live' : 'Draft'}</span>
                  <span>{summary.signals} signals</span>
                  <span>{summary.decisions} decisions</span>
                </div>
                <OpenCue />
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
