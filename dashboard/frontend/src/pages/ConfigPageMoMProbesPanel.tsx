import { useEffect, useMemo, useRef, useState } from 'react'

import type {
  RecipeProbeDetail,
  RecipeProbePage,
  RecipeProbeRunPlan,
  RecipeProbeSummary,
  RecipeProbeValidationResult,
} from '../types/recipe'
import {
  createRecipeProbeRunPlan,
  getRecipeProbe,
  listRecipeProbes,
  validateRecipeProbe,
} from '../utils/recipeApi'
import { RecipeProbeDetailPanel, RecipeProbeValidation } from './ConfigPageMoMProbeResults'
import styles from './ConfigPageMoMProbesPanel.module.css'
import {
  mapRecipeProbeFacetOptions,
  RECIPE_PROBE_PAGE_SIZE,
  type RecipeProbeKey,
} from './configPageMoMProbesSupport'

interface ConfigPageMoMProbesPanelProps {
  onLaunch: (intent: 'run' | 'edit', plan: RecipeProbeRunPlan) => void
  recipeRevision: number
}

export default function ConfigPageMoMProbesPanel({
  onLaunch,
  recipeRevision,
}: ConfigPageMoMProbesPanelProps) {
  const [page, setPage] = useState(1)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [recipe, setRecipe] = useState('')
  const [decision, setDecision] = useState('')
  const [tag, setTag] = useState('')
  const [model, setModel] = useState('')
  const [shape, setShape] = useState('')
  const [result, setResult] = useState<RecipeProbePage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [selected, setSelected] = useState<RecipeProbeKey | null>(null)
  const [detail, setDetail] = useState<RecipeProbeDetail | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [validation, setValidation] = useState<Record<string, RecipeProbeValidationResult>>({})
  const [activeAction, setActiveAction] = useState<string | null>(null)
  const [actionError, setActionError] = useState<Record<string, string>>({})
  const actionControllerRef = useRef<AbortController | null>(null)

  useEffect(
    () => () => {
      actionControllerRef.current?.abort()
    },
    [],
  )

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const nextQuery = query.trim()
      if (nextQuery === debouncedQuery) return
      setPage(1)
      setDebouncedQuery(nextQuery)
    }, 250)
    return () => window.clearTimeout(timer)
  }, [debouncedQuery, query])

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    void listRecipeProbes(
      {
        page,
        pageSize: RECIPE_PROBE_PAGE_SIZE,
        query: debouncedQuery,
        recipe,
        decision,
        tag,
        model,
        requestShape: shape,
      },
      controller.signal,
    )
      .then(setResult)
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setResult(null)
        setError(reason instanceof Error ? reason.message : 'Failed to load Recipe probes.')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
  }, [debouncedQuery, decision, model, page, recipe, recipeRevision, refreshKey, shape, tag])

  useEffect(() => {
    if (!selected) {
      setDetail(null)
      setDetailError(null)
      return
    }
    const controller = new AbortController()
    setDetail(null)
    setDetailLoading(true)
    setDetailError(null)
    void getRecipeProbe(selected.decision, selected.variant, controller.signal)
      .then(setDetail)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setDetailError(reason instanceof Error ? reason.message : 'Failed to load probe detail.')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false)
      })
    return () => controller.abort()
  }, [selected])

  useEffect(() => {
    setSelected(null)
    setValidation({})
    setActionError({})
  }, [recipeRevision, result?.recipe_digest])

  const facets = result?.facets
  const hasFilters = Boolean(debouncedQuery || recipe || decision || tag || model || shape)
  const pageLabel = useMemo(
    () => `Page ${result?.page ?? page} of ${Math.max(result?.total_pages ?? 1, 1)}`,
    [page, result?.page, result?.total_pages],
  )

  const setFilter = (setter: (value: string) => void, value: string) => {
    setPage(1)
    setter(value)
  }

  const launchProbe = async (
    intent: 'run' | 'edit',
    probe: RecipeProbeSummary,
    recipeDigest: string,
  ) => {
    const actionKey = `${intent}:${probe.id}`
    const controller = new AbortController()
    actionControllerRef.current?.abort()
    actionControllerRef.current = controller
    setActiveAction(actionKey)
    setActionError((current) => ({ ...current, [probe.id]: '' }))
    try {
      const plan = await createRecipeProbeRunPlan(
        probe.decision_id,
        probe.variant_id,
        recipeDigest,
        controller.signal,
      )
      if (controller.signal.aborted) return
      onLaunch(intent, plan)
    } catch (reason) {
      if (reason instanceof Error && reason.name === 'AbortError') return
      setActionError((current) => ({
        ...current,
        [probe.id]: reason instanceof Error ? reason.message : `Failed to ${intent} probe.`,
      }))
    } finally {
      if (actionControllerRef.current === controller) {
        actionControllerRef.current = null
        setActiveAction(null)
      }
    }
  }

  const validateProbe = async (probe: RecipeProbeSummary, recipeDigest: string) => {
    const actionKey = `validate:${probe.id}`
    const controller = new AbortController()
    actionControllerRef.current?.abort()
    actionControllerRef.current = controller
    setActiveAction(actionKey)
    setActionError((current) => ({ ...current, [probe.id]: '' }))
    setValidation((current) => {
      const next = { ...current }
      delete next[probe.id]
      return next
    })
    try {
      const next = await validateRecipeProbe(
        probe.decision_id,
        probe.variant_id,
        recipeDigest,
        controller.signal,
      )
      if (controller.signal.aborted) return
      setValidation((current) => ({ ...current, [probe.id]: next }))
    } catch (reason) {
      if (reason instanceof Error && reason.name === 'AbortError') return
      setActionError((current) => ({
        ...current,
        [probe.id]: reason instanceof Error ? reason.message : 'Failed to validate probe.',
      }))
    } finally {
      if (actionControllerRef.current === controller) {
        actionControllerRef.current = null
        setActiveAction(null)
      }
    }
  }

  return (
    <section className={styles.panel}>
      <div className={styles.header}>
        <div>
          <span className={styles.eyebrow}>Offline routing fixtures</span>
          <h2>Probes</h2>
          <p>
            Inspect verified requests, validate routing only, or launch an exact request in
            Playground.
          </p>
        </div>
        <span className={styles.total}>{result?.total ?? 0} probes</span>
      </div>

      <div className={styles.filters}>
        <label className={styles.searchField}>
          <span>Search</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Query, decision, variant, or tag"
          />
        </label>
        <label>
          <span>Recipe</span>
          <select value={recipe} onChange={(event) => setFilter(setRecipe, event.target.value)}>
            <option value="">All recipes</option>
            {mapRecipeProbeFacetOptions(facets?.recipes ?? {}, recipe).map(([value, count]) => (
              <option key={value} value={value}>
                {value} ({count})
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Decision</span>
          <select value={decision} onChange={(event) => setFilter(setDecision, event.target.value)}>
            <option value="">All decisions</option>
            {mapRecipeProbeFacetOptions(facets?.decisions ?? {}, decision).map(([value, count]) => (
              <option key={value} value={value}>
                {value} ({count})
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Tag</span>
          <select value={tag} onChange={(event) => setFilter(setTag, event.target.value)}>
            <option value="">All tags</option>
            {mapRecipeProbeFacetOptions(facets?.tags ?? {}, tag).map(([value, count]) => (
              <option key={value} value={value}>
                {value} ({count})
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Model</span>
          <select value={model} onChange={(event) => setFilter(setModel, event.target.value)}>
            <option value="">All models</option>
            {mapRecipeProbeFacetOptions(facets?.models ?? {}, model).map(([value, count]) => (
              <option key={value} value={value}>
                {value} ({count})
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Shape</span>
          <select value={shape} onChange={(event) => setFilter(setShape, event.target.value)}>
            <option value="">All shapes</option>
            {mapRecipeProbeFacetOptions(facets?.shapes ?? {}, shape).map(([value, count]) => (
              <option key={value} value={value}>
                {value} ({count})
              </option>
            ))}
          </select>
        </label>
        {hasFilters ? (
          <button
            type="button"
            className={styles.clearButton}
            onClick={() => {
              setQuery('')
              setDebouncedQuery('')
              setRecipe('')
              setDecision('')
              setTag('')
              setModel('')
              setShape('')
              setPage(1)
            }}
          >
            Clear
          </button>
        ) : null}
      </div>

      {error ? (
        <div className={styles.asyncState} role="alert">
          <strong>Probes are unavailable</strong>
          <span>{error}</span>
          <button type="button" onClick={() => setRefreshKey((current) => current + 1)}>
            Retry
          </button>
        </div>
      ) : null}

      {!error && loading && !result ? (
        <div className={styles.asyncState} role="status">
          Loading probes…
        </div>
      ) : null}

      {!error && result ? (
        <>
          <div className={`${styles.list} ${loading ? styles.refreshing : ''}`} aria-busy={loading}>
            {result.items.map((probe) => {
              const isSelected = selected?.id === probe.id
              const validationResult = validation[probe.id]
              const titleID = `probe-${probe.id.replace(/[^A-Za-z0-9_-]/g, '-')}`
              return (
                <article key={probe.id} className={styles.probeCard} aria-labelledby={titleID}>
                  <div className={styles.probeMain}>
                    <button
                      type="button"
                      className={styles.disclosure}
                      aria-expanded={isSelected}
                      aria-label={`${isSelected ? 'Hide' : 'View'} ${probe.id} details`}
                      onClick={() =>
                        setSelected(
                          isSelected
                            ? null
                            : {
                                id: probe.id,
                                decision: probe.decision_id,
                                variant: probe.variant_id,
                              },
                        )
                      }
                    >
                      {isSelected ? '−' : '+'}
                    </button>
                    <div className={styles.identity}>
                      <div>
                        <strong id={titleID}>{probe.variant_id}</strong>
                        <code>{probe.decision_id}</code>
                      </div>
                      <p>
                        {probe.display_prompt ||
                          probe.query_preview ||
                          'Structured message fixture'}
                      </p>
                      <div className={styles.tagList}>
                        <span className={styles.recipeTag}>
                          {probe.expected.recipe || 'default'}
                        </span>
                        {!probe.playground.enabled ? (
                          <span className={styles.evalOnlyTag}>Eval only</span>
                        ) : null}
                        {probe.request_shapes.map((item) => (
                          <span key={`shape-${item}`}>{item}</span>
                        ))}
                        {probe.tags.slice(0, 6).map((item) => (
                          <span key={`tag-${item}`}>{item}</span>
                        ))}
                        {probe.tags.length > 6 ? <span>+{probe.tags.length - 6}</span> : null}
                      </div>
                    </div>
                    <div className={styles.expectation}>
                      <span>Expected</span>
                      <strong>{probe.expected.decision}</strong>
                      <small>
                        {probe.expected.algorithm || probe.expected.alias || 'route only'}
                      </small>
                    </div>
                    <div className={styles.rowActions}>
                      <button
                        type="button"
                        disabled={activeAction !== null || !probe.playground.enabled}
                        aria-label={`Run probe ${probe.id}`}
                        title={
                          probe.playground.enabled
                            ? 'Run this exact probe in Playground'
                            : probe.playground.reason
                        }
                        onClick={() => void launchProbe('run', probe, result.recipe_digest)}
                      >
                        {activeAction === `run:${probe.id}` ? 'Preparing…' : 'Run'}
                      </button>
                      <button
                        type="button"
                        disabled={
                          activeAction !== null || !probe.editable || !probe.playground.enabled
                        }
                        aria-label={`Edit probe ${probe.id}`}
                        title={
                          !probe.playground.enabled
                            ? probe.playground.reason
                            : probe.editable
                              ? 'Edit this probe in Playground'
                              : 'This structured probe does not end in a user message'
                        }
                        onClick={() => void launchProbe('edit', probe, result.recipe_digest)}
                      >
                        {activeAction === `edit:${probe.id}` ? 'Preparing…' : 'Edit'}
                      </button>
                      <button
                        type="button"
                        disabled={activeAction !== null}
                        aria-label={`Validate probe ${probe.id}`}
                        onClick={() => void validateProbe(probe, result.recipe_digest)}
                      >
                        {activeAction === `validate:${probe.id}` ? 'Validating…' : 'Validate'}
                      </button>
                    </div>
                  </div>
                  {actionError[probe.id] ? (
                    <div className={styles.rowError} role="alert">
                      {actionError[probe.id]}
                    </div>
                  ) : null}
                  {!probe.playground.enabled ? (
                    <div className={styles.executionNotice} role="note">
                      <strong>Eval-only probe.</strong> {probe.playground.reason}
                    </div>
                  ) : null}
                  {validationResult ? <RecipeProbeValidation result={validationResult} /> : null}
                  {isSelected ? (
                    detailLoading ? (
                      <div className={styles.detailState} role="status">
                        Loading probe detail…
                      </div>
                    ) : detailError ? (
                      <div className={styles.detailState} role="alert">
                        {detailError}
                      </div>
                    ) : detail ? (
                      <RecipeProbeDetailPanel detail={detail} />
                    ) : null
                  ) : null}
                </article>
              )
            })}
            {result.items.length === 0 ? (
              <div className={styles.asyncState}>No probes match the current filters.</div>
            ) : null}
          </div>

          <div className={styles.pager}>
            <span>
              {result.total.toLocaleString()} results · {pageLabel}
            </span>
            <div>
              <button
                type="button"
                disabled={result.page <= 1 || loading}
                onClick={() => setPage(result.page - 1)}
              >
                Previous
              </button>
              <button
                type="button"
                disabled={result.page >= result.total_pages || loading}
                onClick={() => setPage(result.page + 1)}
              >
                Next
              </button>
            </div>
          </div>
        </>
      ) : null}
    </section>
  )
}
