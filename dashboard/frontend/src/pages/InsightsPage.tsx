import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import { DataTable } from '../components/DataTable'
import InsightsCharts from '../components/InsightsCharts'
import ProductIcon from '../components/ProductIcon'
import TableHeader from '../components/TableHeader'

import configStyles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import styles from './InsightsPage.module.css'
import {
  accessUsageEventToInsightsRecord,
  isInsightsDataUnavailableError,
  usageSummaryToInsightsAggregate,
} from './insightsPageApi'
import {
  createInsightsTableColumns,
  formatInsightsDecisionName,
  getInsightsRecordPath,
} from './insightsPageSupport'
import type {
  InsightsAggregateResponse,
  InsightsFilterType,
  InsightsRecord,
} from './insightsPageTypes'
import { inferenceAccessApi } from '../utils/inferenceAccessApi'

const insightsPageSize = 25
const insightsSearchDebounceMs = 300
const EMPTY_AGGREGATE: InsightsAggregateResponse = {
  object: 'management.usage.aggregate',
  record_count: 0,
  lifecycle: { completed: 0, failed: 0, aborted: 0, in_progress: 0, unknown: 0 },
  summary: {
    total_saved: 0,
    baseline_spend: 0,
    actual_spend: 0,
    cost_record_count: 0,
    excluded_record_count: 0,
  },
  model_selection: [],
  decision_distribution: [],
  signal_distribution: [],
  token_volume: {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    excluded_record_count: 0,
  },
  token_breakdown: { by_decision: [], by_selected_model: [] },
  available_recipes: [],
  available_decisions: [],
  available_models: [],
}

export default function InsightsPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const initialSearch = searchParams.get('search') || ''
  const [records, setRecords] = useState<InsightsRecord[]>([])
  const [aggregate, setAggregate] = useState<InsightsAggregateResponse | null>(null)
  const [totalRecords, setTotalRecords] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [historyUnavailable, setHistoryUnavailable] = useState(false)
  const [searchTerm, setSearchTerm] = useState(initialSearch)
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState(initialSearch)
  const [filter, setFilter] = useState<InsightsFilterType>('all')
  const [recipeFilter, setRecipeFilter] = useState('all')
  const [decisionFilter, setDecisionFilter] = useState('all')
  const [modelFilter, setModelFilter] = useState('all')
  const [currentPage, setCurrentPage] = useState(1)
  const [cursorByPage, setCursorByPage] = useState<Record<number, string | undefined>>({
    1: undefined,
  })
  const [hasMore, setHasMore] = useState(false)
  const requestSequenceRef = useRef(0)
  const tableColumns = useMemo(() => createInsightsTableColumns(), [])

  const activeFilters = useMemo(
    () => ({
      searchTerm: debouncedSearchTerm,
      filter,
      recipeFilter,
      decisionFilter,
      modelFilter,
    }),
    [debouncedSearchTerm, filter, recipeFilter, decisionFilter, modelFilter],
  )
  const currentCursor = cursorByPage[currentPage]

  const fetchRecords = useCallback(async () => {
    const requestSequence = requestSequenceRef.current + 1
    requestSequenceRef.current = requestSequence
    setLoading(true)

    try {
      const [logPage, usage] = await Promise.all([
        inferenceAccessApi.requestLogs({
          cursor: currentCursor,
          limit: insightsPageSize,
          q: activeFilters.searchTerm.trim() || undefined,
          model: activeFilters.modelFilter === 'all' ? undefined : activeFilters.modelFilter,
        }),
        inferenceAccessApi.usage({
          model: activeFilters.modelFilter === 'all' ? undefined : activeFilters.modelFilter,
        }),
      ])
      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      const mapped = logPage.items.map(accessUsageEventToInsightsRecord).filter((record) => {
        if (activeFilters.recipeFilter !== 'all' && record.recipe !== activeFilters.recipeFilter) {
          return false
        }
        if (
          activeFilters.decisionFilter !== 'all' &&
          record.decision !== activeFilters.decisionFilter
        ) {
          return false
        }
        if (activeFilters.filter === 'cached' && !record.from_cache) return false
        if (activeFilters.filter === 'streamed' && !record.streaming) return false
        return true
      })
      setRecords(mapped)
      setHasMore(logPage.hasMore)
      setTotalRecords(
        (currentPage - 1) * insightsPageSize + mapped.length + (logPage.hasMore ? 1 : 0),
      )
      if (logPage.nextCursor) {
        setCursorByPage((current) =>
          current[currentPage + 1] === logPage.nextCursor
            ? current
            : { ...current, [currentPage + 1]: logPage.nextCursor },
        )
      }
      setAggregate(usageSummaryToInsightsAggregate(usage, mapped))
      setError(null)
      setHistoryUnavailable(false)
    } catch (err) {
      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      const unavailable = isInsightsDataUnavailableError(err)
      setRecords([])
      setTotalRecords(0)
      setAggregate(null)
      setHasMore(false)
      setError(unavailable ? null : err instanceof Error ? err.message : 'Unknown error')
      setHistoryUnavailable(unavailable)
    } finally {
      if (requestSequenceRef.current === requestSequence) {
        setLoading(false)
      }
    }
  }, [activeFilters, currentCursor, currentPage])

  useEffect(() => {
    const debounceTimer = window.setTimeout(() => {
      setDebouncedSearchTerm(searchTerm)
    }, insightsSearchDebounceMs)

    return () => window.clearTimeout(debounceTimer)
  }, [searchTerm])

  useEffect(() => {
    setCurrentPage(1)
    setCursorByPage({ 1: undefined })
  }, [debouncedSearchTerm, filter, recipeFilter, decisionFilter, modelFilter])

  useEffect(
    () => () => {
      requestSequenceRef.current += 1
    },
    [],
  )

  useEffect(() => {
    void fetchRecords()
  }, [fetchRecords])

  const availableRecipes = aggregate?.available_recipes ?? []
  const availableDecisions = aggregate?.available_decisions ?? []
  const availableModels = aggregate?.available_models ?? []
  const hasInsightsData =
    totalRecords > 0 ||
    (aggregate?.record_count ?? 0) > 0 ||
    availableRecipes.length > 0 ||
    availableDecisions.length > 0 ||
    availableModels.length > 0

  const handleSearchChange = useCallback((value: string) => {
    setSearchTerm(value)
  }, [])

  const handleDecisionFilterChange = useCallback((value: string) => {
    setDecisionFilter(value)
    setCurrentPage(1)
  }, [])

  const handleRecipeFilterChange = useCallback((value: string) => {
    setRecipeFilter(value)
    setCurrentPage(1)
  }, [])

  const handleModelFilterChange = useCallback((value: string) => {
    setModelFilter(value)
    setCurrentPage(1)
  }, [])

  const handleCacheFilterChange = useCallback((value: InsightsFilterType) => {
    setFilter(value)
    setCurrentPage(1)
  }, [])

  const handleViewRecord = useCallback(
    (record: InsightsRecord) => {
      navigate(getInsightsRecordPath(record.id))
    },
    [navigate],
  )

  return (
    <div className={styles.page}>
      <ConfigPageManagerLayout
        eyebrow="Insights"
        title="Insights"
        description="See every routing decision, model pick, and saving."
      >
        {error ? (
          <div className={styles.error} role="alert">
            <span>{error}</span>
          </div>
        ) : null}

        {loading && !aggregate ? (
          <section className={styles.overviewLoading} aria-label="Loading insight overview">
            <div className={styles.overviewLoadingCards} aria-hidden="true">
              {Array.from({ length: 4 }, (_, index) => (
                <span key={index} />
              ))}
            </div>
            <p>Loading request intelligence…</p>
          </section>
        ) : (
          <InsightsCharts aggregate={aggregate || EMPTY_AGGREGATE} />
        )}

        <div className={configStyles.sectionPanel}>
          <section className={configStyles.sectionTableBlock}>
            <div className={styles.toolbar}>
              <div>
                <h2 className={styles.sectionTitle}>Requests</h2>
                <p className={styles.sectionSubtitle}>One row for every routed request.</p>
              </div>
              <div className={styles.toolbarActions}>
                <button
                  type="button"
                  onClick={() => void fetchRecords()}
                  className={styles.refreshButton}
                  disabled={loading}
                >
                  <ProductIcon name="refresh" aria-hidden="true" />
                  {loading ? 'Refreshing…' : 'Refresh'}
                </button>
              </div>
            </div>

            <TableHeader
              title="Requests"
              count={totalRecords}
              searchPlaceholder="Search request ID"
              searchValue={searchTerm}
              onSearchChange={handleSearchChange}
              variant="embedded"
            />

            <div className={styles.filterRow}>
              <select
                className={styles.filterSelect}
                aria-label="Filter by recipe"
                value={recipeFilter}
                onChange={(event) => handleRecipeFilterChange(event.target.value)}
                disabled={availableRecipes.length === 0}
              >
                <option value="all">All Recipes</option>
                {availableRecipes.map((recipe) => (
                  <option key={recipe} value={recipe}>
                    {recipe}
                  </option>
                ))}
              </select>

              <select
                className={styles.filterSelect}
                aria-label="Filter by decision"
                value={decisionFilter}
                onChange={(event) => handleDecisionFilterChange(event.target.value)}
                disabled={availableDecisions.length === 0}
              >
                <option value="all">All Decisions</option>
                {availableDecisions.map((decision) => (
                  <option key={decision} value={decision}>
                    {formatInsightsDecisionName(decision)}
                  </option>
                ))}
              </select>

              <select
                className={styles.filterSelect}
                aria-label="Filter by model"
                value={modelFilter}
                onChange={(event) => handleModelFilterChange(event.target.value)}
                disabled={availableModels.length === 0}
              >
                <option value="all">All Models</option>
                {availableModels.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>

              <select
                className={styles.filterSelect}
                aria-label="Filter by cache status"
                value={filter}
                onChange={(event) =>
                  handleCacheFilterChange(event.target.value as InsightsFilterType)
                }
              >
                <option value="all">Cache Status</option>
                <option value="cached">Cached Only</option>
                <option value="streamed">Streamed Only</option>
              </select>
            </div>

            {loading && records.length === 0 ? (
              <div className={styles.loadingInline}>
                <div className={styles.spinner} />
                <p>Loading requests…</p>
              </div>
            ) : !hasInsightsData ? (
              <div className={styles.emptyState}>
                {historyUnavailable ? (
                  <div className={styles.emptyHint}>
                    <p>Insights are unavailable.</p>
                    <p className={styles.emptySubtext}>Request history is not ready.</p>
                  </div>
                ) : error ? (
                  <div className={styles.emptyHint}>
                    <p>Insights are unavailable.</p>
                    <p className={styles.emptySubtext}>Try again in a moment.</p>
                  </div>
                ) : (
                  <div className={styles.emptyHint}>
                    <p>No requests yet.</p>
                    <p className={styles.emptySubtext}>Send a request to see its complete route.</p>
                  </div>
                )}
              </div>
            ) : (
              <DataTable
                columns={tableColumns}
                data={records}
                keyExtractor={(row) => row.id}
                onView={handleViewRecord}
                openOnRowClick
                emptyMessage="No requests match these filters"
                className={styles.insightsTable}
              />
            )}

            {currentPage > 1 || hasMore ? (
              <nav className={styles.pagination} aria-label="Request pages">
                <span className={styles.paginationRange}>
                  {((currentPage - 1) * insightsPageSize + 1).toLocaleString('en-US')}–
                  {((currentPage - 1) * insightsPageSize + records.length).toLocaleString('en-US')}
                </span>
                <div className={styles.paginationControls}>
                  <button
                    type="button"
                    className={styles.paginationButton}
                    onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                    disabled={currentPage === 1}
                  >
                    <ProductIcon name="arrow-left" aria-hidden="true" />
                    Previous
                  </button>
                  <span className={styles.paginationInfo}>Page {currentPage}</span>
                  <button
                    type="button"
                    className={styles.paginationButton}
                    onClick={() => setCurrentPage((page) => page + 1)}
                    disabled={!hasMore || !cursorByPage[currentPage + 1]}
                  >
                    Next
                    <ProductIcon name="arrow-right" aria-hidden="true" />
                  </button>
                </div>
              </nav>
            ) : null}
          </section>
        </div>
      </ConfigPageManagerLayout>
    </div>
  )
}
