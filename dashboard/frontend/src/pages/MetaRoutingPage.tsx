import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import { DataTable } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import ViewModal from '../components/ViewModal'

import ConfigPageManagerLayout from './ConfigPageManagerLayout'
import styles from './MetaRoutingPage.module.css'
import {
  buildMetaRoutingQueryString,
  buildMetaRoutingRecordSections,
  createMetaRoutingTableColumns,
  distributionMax,
  fetchMetaRoutingJSON,
  formatLatencyDelta,
  formatPercent,
} from './metaRoutingPageSupport'
import type {
  MetaRoutingAggregateResponse,
  MetaRoutingAggregateValue,
  MetaRoutingFeedbackDetailResponse,
  MetaRoutingFeedbackListResponse,
  MetaRoutingFeedbackSummary,
  MetaRoutingQueryFilters,
} from './metaRoutingPageTypes'

const metaRoutingPageSize = 25

function DistributionCard({
  title,
  data,
}: {
  title: string
  data: MetaRoutingAggregateValue[]
}) {
  const maxValue = distributionMax(data)

  return (
    <article className={styles.chartCard}>
      <h3 className={styles.chartTitle}>{title}</h3>
      {data.length === 0 ? (
        <span className={styles.chartEmpty}>No records in the current filter window.</span>
      ) : (
        <div className={styles.distributionList}>
          {data.slice(0, 8).map((entry) => (
            <div key={`${title}-${entry.name}`} className={styles.distributionRow}>
              <span className={styles.distributionLabel}>{entry.name}</span>
              <div className={styles.distributionBarTrack}>
                <div
                  className={styles.distributionBar}
                  style={{ width: `${maxValue > 0 ? (entry.value / maxValue) * 100 : 0}%` }}
                />
              </div>
              <span className={styles.distributionValue}>{entry.value}</span>
            </div>
          ))}
        </div>
      )}
    </article>
  )
}

export default function MetaRoutingPage() {
  const [searchParams] = useSearchParams()
  const initialSearch = searchParams.get('search') || ''

  const [records, setRecords] = useState<MetaRoutingFeedbackSummary[]>([])
  const [aggregate, setAggregate] = useState<MetaRoutingAggregateResponse | null>(null)
  const [totalRecords, setTotalRecords] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [currentPage, setCurrentPage] = useState(1)
  const [searchTerm, setSearchTerm] = useState(initialSearch)
  const [mode, setMode] = useState(searchParams.get('mode') || 'all')
  const [trigger, setTrigger] = useState(searchParams.get('trigger') || 'all')
  const [rootCause, setRootCause] = useState(searchParams.get('root_cause') || 'all')
  const [actionType, setActionType] = useState(searchParams.get('action_type') || 'all')
  const [signalFamily, setSignalFamily] = useState(searchParams.get('signal_family') || 'all')
  const [overturned, setOverturned] = useState<'all' | 'true' | 'false'>(
    (searchParams.get('overturned') as 'all' | 'true' | 'false') || 'all',
  )
  const [decision, setDecision] = useState(searchParams.get('decision') || 'all')
  const [model, setModel] = useState(searchParams.get('model') || 'all')
  const [responseStatus, setResponseStatus] = useState(searchParams.get('response_status') || 'all')
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [modalLoading, setModalLoading] = useState(false)
  const [selectedDetail, setSelectedDetail] = useState<MetaRoutingFeedbackDetailResponse | null>(null)
  const requestSequenceRef = useRef(0)
  const tableColumns = useMemo(() => createMetaRoutingTableColumns(), [])

  useEffect(() => {
    setSearchTerm(searchParams.get('search') || '')
    setMode(searchParams.get('mode') || 'all')
    setTrigger(searchParams.get('trigger') || 'all')
    setRootCause(searchParams.get('root_cause') || 'all')
    setActionType(searchParams.get('action_type') || 'all')
    setSignalFamily(searchParams.get('signal_family') || 'all')
    setOverturned((searchParams.get('overturned') as 'all' | 'true' | 'false') || 'all')
    setDecision(searchParams.get('decision') || 'all')
    setModel(searchParams.get('model') || 'all')
    setResponseStatus(searchParams.get('response_status') || 'all')
    setCurrentPage(1)
  }, [searchParams])

  const activeFilters = useMemo<MetaRoutingQueryFilters>(
    () => ({
      searchTerm,
      mode,
      trigger,
      rootCause,
      actionType,
      signalFamily,
      overturned,
      decision,
      model,
      responseStatus,
    }),
    [actionType, decision, mode, model, overturned, responseStatus, rootCause, searchTerm, signalFamily, trigger],
  )

  const totalPages = Math.max(1, Math.ceil(totalRecords / metaRoutingPageSize))
  const listQuery = useMemo(
    () => buildMetaRoutingQueryString(activeFilters, { limit: metaRoutingPageSize, offset: (currentPage - 1) * metaRoutingPageSize }),
    [activeFilters, currentPage],
  )
  const aggregateQuery = useMemo(() => buildMetaRoutingQueryString(activeFilters), [activeFilters])

  const fetchRecords = useCallback(async () => {
    const requestSequence = requestSequenceRef.current + 1
    requestSequenceRef.current = requestSequence
    setLoading(true)

    try {
      const [listResponse, aggregateResponse] = await Promise.all([
        fetchMetaRoutingJSON<MetaRoutingFeedbackListResponse>(
          `/api/router/v1/meta_routing_feedback${listQuery}`,
          'meta-routing feedback records',
        ),
        fetchMetaRoutingJSON<MetaRoutingAggregateResponse>(
          `/api/router/v1/meta_routing_feedback/aggregate${aggregateQuery}`,
          'meta-routing aggregates',
        ),
      ])

      if (requestSequenceRef.current !== requestSequence) {
        return
      }

      setRecords(listResponse.data || [])
      setTotalRecords(typeof listResponse.total === 'number' ? listResponse.total : listResponse.count)
      setAggregate(aggregateResponse)
      setError(null)
    } catch (err) {
      if (requestSequenceRef.current !== requestSequence) {
        return
      }
      setRecords([])
      setTotalRecords(0)
      setAggregate(null)
      setError(err instanceof Error ? err.message : 'Unknown error')
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

  const openRecord = useCallback(async (record: MetaRoutingFeedbackSummary) => {
    setModalLoading(true)
    setViewModalOpen(true)
    try {
      const detail = await fetchMetaRoutingJSON<MetaRoutingFeedbackDetailResponse>(
        `/api/router/v1/meta_routing_feedback/${record.id}`,
        'meta-routing feedback record',
      )
      setSelectedDetail(detail)
    } catch {
      setSelectedDetail(null)
    } finally {
      setModalLoading(false)
    }
  }, [])

  const closeModal = useCallback(() => {
    setViewModalOpen(false)
    setSelectedDetail(null)
  }, [])

  const availableModes = aggregate?.available_modes ?? []
  const availableTriggers = aggregate?.available_triggers ?? []
  const availableRootCauses = aggregate?.available_root_causes ?? []
  const availableActionTypes = aggregate?.available_action_types ?? []
  const availableSignalFamilies = aggregate?.available_signal_families ?? []
  const availableDecisions = aggregate?.available_decisions ?? []
  const availableModels = aggregate?.available_models ?? []
  const availableStatuses = aggregate?.available_response_statuses ?? []
  const hasData =
    totalRecords > 0 ||
    (aggregate?.record_count ?? 0) > 0 ||
    availableModes.length > 0 ||
    availableTriggers.length > 0

  const resetFilters = () => {
    setSearchTerm('')
    setMode('all')
    setTrigger('all')
    setRootCause('all')
    setActionType('all')
    setSignalFamily('all')
    setOverturned('all')
    setDecision('all')
    setModel('all')
    setResponseStatus('all')
    setCurrentPage(1)
  }

  if (loading && !hasData && records.length === 0) {
    return (
      <div className={styles.loading}>
        <div className={styles.spinner} />
        <p>Loading meta-routing feedback...</p>
      </div>
    )
  }

  const summary = aggregate?.summary

  return (
    <ConfigPageManagerLayout
      eyebrow="Analysis"
      title="Meta Routing"
      description="Inspect assess-and-refine traces, bounded refinement plans, and whether a second pass actually changed the route."
      configArea="Analysis"
      scope="Pass-level routing traces and feedback outcomes"
      panelTitle="Meta Routing Feedback"
      panelDescription="Use persisted feedback records to compare rollout modes, identify fragile decisions, and drill into the exact pass that triggered refinement."
      pills={[
        { label: 'Observe' },
        { label: 'Shadow' },
        { label: 'Active' },
        { label: 'Feedback Records', active: true },
      ]}
    >
      {error ? (
        <div className={styles.error}>
          <span>{error}</span>
        </div>
      ) : null}

      {aggregate && aggregate.record_count > 0 ? (
        <>
          <section className={styles.summaryGrid}>
            <article className={styles.summaryCard}>
              <span className={styles.summaryLabel}>Planned Refinement</span>
              <strong className={`${styles.summaryValue} ${styles.summaryAccent}`}>{formatPercent(summary?.planned_refinement_rate)}</strong>
              <span className={styles.summarySubtle}>How often a pass crossed the trigger threshold and produced a plan.</span>
            </article>
            <article className={styles.summaryCard}>
              <span className={styles.summaryLabel}>Executed Refinement</span>
              <strong className={styles.summaryValue}>{formatPercent(summary?.executed_refinement_rate)}</strong>
              <span className={styles.summarySubtle}>Only shadow and active modes execute the bounded refinement plan.</span>
            </article>
            <article className={styles.summaryCard}>
              <span className={styles.summaryLabel}>Decision Overturn Rate</span>
              <strong className={`${styles.summaryValue} ${styles.summaryAccent}`}>{formatPercent(summary?.overturn_rate)}</strong>
              <span className={styles.summarySubtle}>How often the chosen route or final model changed after refinement.</span>
            </article>
            <article className={styles.summaryCard}>
              <span className={styles.summaryLabel}>Average Latency Delta</span>
              <strong className={styles.summaryValue}>{formatLatencyDelta(summary?.average_latency_delta_ms)}</strong>
              <span className={styles.summarySubtle}>p95 {formatLatencyDelta(summary?.p95_latency_delta_ms)} · Top trigger {summary?.top_trigger || 'n/a'}</span>
            </article>
          </section>

          <section className={styles.chartsGrid}>
            <DistributionCard title="Rollout Modes" data={aggregate.mode_distribution} />
            <DistributionCard title="Triggers" data={aggregate.trigger_distribution} />
            <DistributionCard title="Root Causes" data={aggregate.root_cause_distribution} />
            <DistributionCard title="Allowed Actions" data={aggregate.action_type_distribution} />
            <DistributionCard title="Refined Families" data={aggregate.signal_family_distribution} />
            <DistributionCard title="Decision Changes" data={aggregate.decision_change_distribution} />
          </section>
        </>
      ) : null}

      <div className={styles.toolbar}>
        <TableHeader
          title="Feedback Records"
          count={totalRecords}
          searchPlaceholder="Search request id, model, decision, or query"
          searchValue={searchTerm}
          onSearchChange={(value) => {
            setSearchTerm(value)
            setCurrentPage(1)
          }}
          onSecondaryAction={() => void fetchRecords()}
          secondaryActionText="Refresh"
          variant="embedded"
        />
        <div className={styles.toolbarActions}>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => setAutoRefresh(event.target.checked)}
            />
            <span>Auto refresh</span>
          </label>
          <Link className={styles.linkButton} to="/config/meta-routing">
            Edit Policy
          </Link>
          <button type="button" className={styles.refreshButton} onClick={resetFilters}>
            Reset Filters
          </button>
        </div>
      </div>

      <div className={styles.filterRow}>
        <select className={styles.filterSelect} value={mode} onChange={(event) => { setMode(event.target.value); setCurrentPage(1) }}>
          <option value="all">All modes</option>
          {availableModes.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        <select className={styles.filterSelect} value={trigger} onChange={(event) => { setTrigger(event.target.value); setCurrentPage(1) }}>
          <option value="all">All triggers</option>
          {availableTriggers.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        <select className={styles.filterSelect} value={rootCause} onChange={(event) => { setRootCause(event.target.value); setCurrentPage(1) }}>
          <option value="all">All root causes</option>
          {availableRootCauses.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        <select className={styles.filterSelect} value={actionType} onChange={(event) => { setActionType(event.target.value); setCurrentPage(1) }}>
          <option value="all">All actions</option>
          {availableActionTypes.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        <select className={styles.filterSelect} value={signalFamily} onChange={(event) => { setSignalFamily(event.target.value); setCurrentPage(1) }}>
          <option value="all">All families</option>
          {availableSignalFamilies.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        <select className={styles.filterSelect} value={overturned} onChange={(event) => { setOverturned(event.target.value as 'all' | 'true' | 'false'); setCurrentPage(1) }}>
          <option value="all">All outcomes</option>
          <option value="true">Overturned</option>
          <option value="false">Stable</option>
        </select>
        <select className={styles.filterSelect} value={decision} onChange={(event) => { setDecision(event.target.value); setCurrentPage(1) }}>
          <option value="all">All decisions</option>
          {availableDecisions.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        <select className={styles.filterSelect} value={model} onChange={(event) => { setModel(event.target.value); setCurrentPage(1) }}>
          <option value="all">All models</option>
          {availableModels.map((option) => <option key={option} value={option}>{option}</option>)}
        </select>
        <select className={styles.filterSelect} value={responseStatus} onChange={(event) => { setResponseStatus(event.target.value); setCurrentPage(1) }}>
          <option value="all">All statuses</option>
          {availableStatuses.map((option) => <option key={option} value={String(option)}>{option}</option>)}
        </select>
      </div>

      {records.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyHint}>
            <h3>No meta-routing feedback records</h3>
            <p className={styles.emptySubtext}>
              Enable `routing.meta`, let traffic pass through the router, and the runtime will emit one `FeedbackRecord` per request in observe, shadow, and active modes.
            </p>
            <pre className={styles.configHint}>{`routing:\n  meta:\n    mode: observe\n    max_passes: 2\n    trigger_policy:\n      decision_margin_below: 0.18\n      projection_boundary_within: 0.07\n      partition_conflict: true`}</pre>
            <Link className={styles.linkButton} to="/config/meta-routing">
              Configure Meta Routing
            </Link>
          </div>
        </div>
      ) : (
        <>
          <DataTable
            columns={tableColumns}
            data={records}
            keyExtractor={(row) => row.id}
            onView={(row) => { void openRecord(row) }}
            emptyMessage="No meta-routing feedback records match the current filters."
            readonly={true}
          />
          <div className={styles.pagination}>
            <button
              type="button"
              className={styles.paginationButton}
              disabled={currentPage <= 1}
              onClick={() => setCurrentPage((page) => Math.max(1, page - 1))}
            >
              Previous
            </button>
            <span className={styles.paginationInfo}>
              Page {currentPage} of {totalPages}
            </span>
            <button
              type="button"
              className={styles.paginationButton}
              disabled={currentPage >= totalPages}
              onClick={() => setCurrentPage((page) => Math.min(totalPages, page + 1))}
            >
              Next
            </button>
          </div>
        </>
      )}

      <ViewModal
        isOpen={viewModalOpen}
        onClose={closeModal}
        title={selectedDetail ? `Meta Routing Record: ${selectedDetail.id}` : 'Meta Routing Record'}
        sections={selectedDetail
          ? buildMetaRoutingRecordSections(selectedDetail)
          : [
            {
              title: 'Loading',
              fields: [
                {
                  label: 'Status',
                  value: modalLoading ? 'Loading record details...' : 'Unable to load record details.',
                  fullWidth: true,
                },
              ],
            },
          ]}
      />
    </ConfigPageManagerLayout>
  )
}
