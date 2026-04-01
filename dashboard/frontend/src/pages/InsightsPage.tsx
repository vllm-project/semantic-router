import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { DataTable } from '../components/DataTable'
import InsightsCharts from '../components/InsightsCharts'
import TableHeader from '../components/TableHeader'

import configStyles from './ConfigPage.module.css'
import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import styles from './InsightsPage.module.css'
import {
  fetchInsightsJSON,
  isInsightsReplayUnavailableError,
} from './insightsPageApi'
import {
  createInsightsTableColumns,
  getInsightsRecordPath,
} from './insightsPageSupport'
import type {
  InsightsAggregateResponse,
  InsightsFilterType,
  InsightsListResponse,
  InsightsRecord,
} from './insightsPageTypes'

const insightsPageSize = 25

interface ReplayQueryFilters {
  searchTerm: string
  filter: InsightsFilterType
  decisionFilter: string
  modelFilter: string
}

function buildReplayQueryString(filters: ReplayQueryFilters, pagination?: { limit: number; offset: number }) {
  const params = new URLSearchParams()

  if (pagination) {
    params.set('limit', String(pagination.limit))
    params.set('offset', String(pagination.offset))
  }

  const search = filters.searchTerm.trim()
  if (search) {
    params.set('search', search)
  }
  if (filters.filter !== 'all') {
    params.set('cache_status', filters.filter)
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
  const [records, setRecords] = useState<InsightsRecord[]>([])
  const [aggregate, setAggregate] = useState<InsightsAggregateResponse | null>(null)
  const [totalRecords, setTotalRecords] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [replayUnavailable, setReplayUnavailable] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [filter, setFilter] = useState<InsightsFilterType>('all')
  const [decisionFilter, setDecisionFilter] = useState('all')
  const [modelFilter, setModelFilter] = useState('all')
  const [currentPage, setCurrentPage] = useState(1)
  const requestSequenceRef = useRef(0)
  const tableColumns = useMemo(() => createInsightsTableColumns(), [])

  const activeFilters = useMemo(
    () => ({
      searchTerm,
      filter,
      decisionFilter,
      modelFilter,
    }),
    [searchTerm, filter, decisionFilter, modelFilter],
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
    setLoading(true)

    try {
      const [listResponse, aggregateResponse] = await Promise.all([
        fetchInsightsJSON<InsightsListResponse>(`/api/router/v1/router_replay${listQuery}`, 'insight records'),
        fetchInsightsJSON<InsightsAggregateResponse>(
          `/api/router/v1/router_replay/aggregate${aggregateQuery}`,
          'insight aggregates',
        ),
      ])
      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      setRecords(listResponse.data || [])
      setTotalRecords(typeof listResponse.total === 'number' ? listResponse.total : listResponse.count)
      setAggregate(aggregateResponse)
      setError(null)
      setReplayUnavailable(false)
    } catch (err) {
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
      if (requestSequenceRef.current === requestSequence) {
        setLoading(false)
      }
    }
  }, [aggregateQuery, listQuery])

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

  const availableDecisions = aggregate?.available_decisions ?? []
  const availableModels = aggregate?.available_models ?? []
  const hasReplayData =
    totalRecords > 0 ||
    (aggregate?.record_count ?? 0) > 0 ||
    availableDecisions.length > 0 ||
    availableModels.length > 0

  const handleSearchChange = useCallback((value: string) => {
    setSearchTerm(value)
    setCurrentPage(1)
  }, [])

  const handleDecisionFilterChange = useCallback((value: string) => {
    setDecisionFilter(value)
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

  const handleViewRecord = useCallback((record: InsightsRecord) => {
    navigate(getInsightsRecordPath(record.id))
  }, [navigate])

  if (loading && !hasReplayData && records.length === 0) {
    return (
      <div className={styles.loading}>
        <div className={styles.spinner} />
        <p>Loading insight records...</p>
      </div>
    )
  }

  return (
    <ConfigPageManagerLayout
      eyebrow="Insights"
      title="Insights"
      description="See what the router picked, what signals fired, and how much it saved."
      configArea="Analysis"
      scope="Filtered replay intelligence"
      panelTitle="Semantic Router Insights"
      panelDescription="Decisions, model picks, token usage, and savings in one view."
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

      {aggregate ? <InsightsCharts aggregate={aggregate} /> : null}

      <div className={configStyles.sectionPanel}>
        <section className={configStyles.sectionTableBlock}>
          <div className={styles.toolbar}>
            <div>
              <h2 className={styles.sectionTitle}>Insight Records</h2>
              <p className={styles.sectionSubtitle}>
                Replay-backed routing records with spend, savings, and token details per request.
              </p>
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
              <button type="button" onClick={() => void fetchRecords()} className={styles.refreshButton}>
                Refresh
              </button>
            </div>
          </div>

          <TableHeader
            title="Routing Insights"
            count={totalRecords}
            searchPlaceholder="Search by Request ID..."
            searchValue={searchTerm}
            onSearchChange={handleSearchChange}
            variant="embedded"
          />

          <div className={styles.filterRow}>
            <select
              className={styles.filterSelect}
              value={decisionFilter}
              onChange={(event) => handleDecisionFilterChange(event.target.value)}
              disabled={availableDecisions.length === 0}
            >
              <option value="all">All Decisions</option>
              {availableDecisions.map((decision) => (
                <option key={decision} value={decision}>
                  {decision}
                </option>
              ))}
            </select>

            <select
              className={styles.filterSelect}
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
              value={filter}
              onChange={(event) => handleCacheFilterChange(event.target.value as InsightsFilterType)}
            >
              <option value="all">Cache Status</option>
              <option value="cached">Cached Only</option>
              <option value="streamed">Streamed Only</option>
            </select>
          </div>

          {!hasReplayData && !loading ? (
            <div className={styles.emptyState}>
              {replayUnavailable ? (
                <div className={styles.emptyHint}>
                  <p>Insights stay empty until router replay is enabled and requests flow through the router.</p>
                  <p className={styles.emptySubtext}>
                    Enable `global.services.router_replay.enabled`, or override a specific decision with `router_replay.enabled: true`. Use `enabled: false` on a decision only when you need to turn replay off for that route.
                  </p>
                </div>
              ) : error ? (
                <div className={styles.emptyHint}>
                  <p>Unable to load insights. If replay is disabled, enable router replay globally or on the affected decision, then send traffic through the router.</p>
                  <pre className={styles.configHint}>{`global:
  services:
    router_replay:
      enabled: true
      store_backend: memory  # or redis, postgres, milvus

routing:
  decisions:
    - name: some-route
      plugins:
        - type: router_replay
          configuration:
            enabled: false  # optional per-decision opt-out`}</pre>
                  <p className={styles.emptySubtext}>Then restart the router and send some requests.</p>
                </div>
              ) : (
                <div className={styles.emptyHint}>
                  <p>Insights records will appear here once requests are processed.</p>
                  <p className={styles.emptySubtext}>Send chat completion traffic through the router to populate this view.</p>
                </div>
              )}
            </div>
          ) : (
            <DataTable
              columns={tableColumns}
              data={records}
              keyExtractor={(row) => row.id}
              onView={handleViewRecord}
              emptyMessage="No insight records match your current filters"
              className={styles.insightsTable}
            />
          )}
        </section>
      </div>

      {totalRecords > insightsPageSize ? (
        <div className={styles.pagination}>
          <button type="button" className={styles.paginationButton} onClick={() => setCurrentPage(1)} disabled={currentPage === 1}>
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
            Page {currentPage} of {totalPages} ({totalRecords} records)
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
          >
            Last
          </button>
        </div>
      ) : null}
    </ConfigPageManagerLayout>
  )
}
