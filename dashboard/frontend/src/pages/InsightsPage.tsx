import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import { DataTable } from '../components/DataTable'
import InsightsCharts from '../components/InsightsCharts'
import TableHeader from '../components/TableHeader'

import configStyles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import styles from './InsightsPage.module.css'
import { isInsightsReplayUnavailableError } from './insightsPageApi'
import { fetchAbortableInsightsJSON, isAbortError } from './insightsPageRequestSupport'
import {
  createInsightsTableColumns,
  formatInsightsDecisionName,
  getInsightsRecordPath,
} from './insightsPageSupport'
import type {
  InsightsAggregateResponse,
  InsightsFilterType,
  InsightsListResponse,
  InsightsRecord,
} from './insightsPageTypes'

const insightsPageSize = 25
const insightsSearchDebounceMs = 300
const EMPTY_AGGREGATE: InsightsAggregateResponse = {
  object: 'router_replay.aggregate',
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

interface ReplayQueryFilters {
  searchTerm: string
  filter: InsightsFilterType
  recipeFilter: string
  decisionFilter: string
  modelFilter: string
}

function buildReplayQueryString(
  filters: ReplayQueryFilters,
  pagination?: { limit: number; offset: number },
) {
  const params = new URLSearchParams()

  if (pagination) {
    params.set('limit', String(pagination.limit))
    params.set('offset', String(pagination.offset))
    params.set('showDetails', 'false')
  }

  const search = filters.searchTerm.trim()
  if (search) {
    params.set('search', search)
  }
  if (filters.filter !== 'all') {
    params.set('cache_status', filters.filter)
  }
  if (filters.recipeFilter !== 'all') {
    params.set('recipe', filters.recipeFilter)
  }
  if (filters.decisionFilter !== 'all') {
    params.set('decision', filters.decisionFilter)
  }
  if (filters.modelFilter !== 'all') {
    params.set('model', filters.modelFilter)
  }

  const query = params.toString()
  return query ? `?${query}` : ''
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
  const [replayUnavailable, setReplayUnavailable] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [searchTerm, setSearchTerm] = useState(initialSearch)
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState(initialSearch)
  const [filter, setFilter] = useState<InsightsFilterType>('all')
  const [recipeFilter, setRecipeFilter] = useState('all')
  const [decisionFilter, setDecisionFilter] = useState('all')
  const [modelFilter, setModelFilter] = useState('all')
  const [currentPage, setCurrentPage] = useState(1)
  const requestSequenceRef = useRef(0)
  const requestAbortControllerRef = useRef<AbortController | null>(null)
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

  const totalPages = Math.max(1, Math.ceil(totalRecords / insightsPageSize))
  const listQuery = useMemo(
    () =>
      buildReplayQueryString(activeFilters, {
        limit: insightsPageSize,
        offset: (currentPage - 1) * insightsPageSize,
      }),
    [activeFilters, currentPage],
  )
  const aggregateQuery = useMemo(() => buildReplayQueryString(activeFilters), [activeFilters])

  const fetchRecords = useCallback(async () => {
    const requestSequence = requestSequenceRef.current + 1
    requestSequenceRef.current = requestSequence
    requestAbortControllerRef.current?.abort()
    const abortController = new AbortController()
    requestAbortControllerRef.current = abortController
    setLoading(true)

    try {
      const [listResponse, aggregateResponse] = await Promise.all([
        fetchAbortableInsightsJSON<InsightsListResponse>(
          `/api/router/v1/router_replay${listQuery}`,
          'insight records',
          abortController.signal,
        ),
        fetchAbortableInsightsJSON<InsightsAggregateResponse>(
          `/api/router/v1/router_replay/aggregate${aggregateQuery}`,
          'insight aggregates',
          abortController.signal,
        ),
      ])
      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      setRecords(listResponse.data || [])
      setTotalRecords(
        typeof listResponse.total === 'number' ? listResponse.total : listResponse.count,
      )
      setAggregate(aggregateResponse)
      setError(null)
      setReplayUnavailable(false)
    } catch (err) {
      if (isAbortError(err)) {
        return
      }
      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      const unavailable = isInsightsReplayUnavailableError(err)
      setRecords([])
      setTotalRecords(0)
      setAggregate(null)
      setError(unavailable ? null : err instanceof Error ? err.message : 'Unknown error')
      setReplayUnavailable(unavailable)
    } finally {
      if (requestAbortControllerRef.current === abortController) {
        requestAbortControllerRef.current = null
      }
      if (requestSequenceRef.current === requestSequence) {
        setLoading(false)
      }
    }
  }, [aggregateQuery, listQuery])

  useEffect(() => {
    const debounceTimer = window.setTimeout(() => {
      setDebouncedSearchTerm(searchTerm)
    }, insightsSearchDebounceMs)

    return () => window.clearTimeout(debounceTimer)
  }, [searchTerm])

  useEffect(() => {
    setCurrentPage(1)
  }, [debouncedSearchTerm])

  useEffect(
    () => () => {
      requestSequenceRef.current += 1
      requestAbortControllerRef.current?.abort()
    },
    [],
  )

  useEffect(() => {
    void fetchRecords()

    if (!autoRefresh) {
      return undefined
    }

    const interval = window.setInterval(() => {
      void fetchRecords()
    }, 5000)

    return () => window.clearInterval(interval)
  }, [autoRefresh, fetchRecords])

  useEffect(() => {
    if (currentPage > totalPages) {
      setCurrentPage(totalPages)
    }
  }, [currentPage, totalPages])

  const availableRecipes = aggregate?.available_recipes ?? []
  const availableDecisions = aggregate?.available_decisions ?? []
  const availableModels = aggregate?.available_models ?? []
  const hasReplayData =
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
        configArea="Analysis"
        scope="Live request intelligence"
        panelTitle="Routing intelligence"
        panelDescription="Decisions, usage, and savings at a glance."
        pills={[
          { label: 'Cost Savings', active: true },
          { label: 'Selections' },
          { label: 'Signals' },
        ]}
      >
        {error ? (
          <div className={styles.error}>
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
                <label className={styles.toggle}>
                  <input
                    type="checkbox"
                    checked={autoRefresh}
                    onChange={(event) => setAutoRefresh(event.target.checked)}
                  />
                  <span>Auto-refresh</span>
                </label>
                <button
                  type="button"
                  onClick={() => void fetchRecords()}
                  className={styles.refreshButton}
                >
                  Refresh
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
            ) : !hasReplayData ? (
              <div className={styles.emptyState}>
                {replayUnavailable ? (
                  <div className={styles.emptyHint}>
                    <p>Insights are off.</p>
                    <p className={styles.emptySubtext}>
                      Enable Router Replay to start collecting requests.
                    </p>
                  </div>
                ) : error ? (
                  <div className={styles.emptyHint}>
                    <p>Insights are unavailable.</p>
                    <p className={styles.emptySubtext}>Check Router Replay, then try again.</p>
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
                emptyMessage="No requests match these filters"
                className={styles.insightsTable}
              />
            )}

            {totalRecords > insightsPageSize ? (
              <nav className={styles.pagination} aria-label="Request pages">
                <span className={styles.paginationRange}>
                  {((currentPage - 1) * insightsPageSize + 1).toLocaleString('en-US')}–
                  {Math.min(currentPage * insightsPageSize, totalRecords).toLocaleString('en-US')}{' '}
                  of {totalRecords.toLocaleString('en-US')}
                </span>
                <div className={styles.paginationControls}>
                  <button
                    type="button"
                    className={styles.paginationButton}
                    onClick={() => setCurrentPage(1)}
                    disabled={currentPage === 1}
                    aria-label="First page"
                  >
                    First
                  </button>
                  <button
                    type="button"
                    className={styles.paginationButton}
                    onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </button>
                  <span className={styles.paginationInfo}>
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    type="button"
                    className={styles.paginationButton}
                    onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}
                    disabled={currentPage === totalPages}
                  >
                    Next
                  </button>
                  <button
                    type="button"
                    className={styles.paginationButton}
                    onClick={() => setCurrentPage(totalPages)}
                    disabled={currentPage === totalPages}
                    aria-label="Last page"
                  >
                    Last
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
