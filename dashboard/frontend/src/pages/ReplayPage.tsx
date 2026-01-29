import React, { useEffect, useState, useCallback, useMemo } from 'react'
import { DataTable, Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import ViewModal, { ViewSection, ViewField } from '../components/ViewModal'
import CollapsibleSection from '../components/CollapsibleSection'
import ReplayCharts from '../components/ReplayCharts'
import { formatDate } from '../types/evaluation'
import { useReadonly } from '../contexts/ReadonlyContext'
import styles from './ReplayPage.module.css'

// Types based on the backend store/store.go
interface Signal {
  keyword?: string[]
  embedding?: string[]
  domain?: string[]
  fact_check?: string[]
  user_feedback?: string[]
  preference?: string[]
  language?: string[]
  latency?: string[]
  context?: string[]
}

interface ReplayRecord {
  id: string
  timestamp: string
  request_id?: string
  decision?: string
  category?: string
  original_model?: string
  selected_model?: string
  reasoning_mode?: string
  confidence_score?: number
  selection_method?: string
  signals: Signal
  request_body?: string
  response_body?: string
  response_status?: number
  from_cache?: boolean
  streaming?: boolean
  request_body_truncated?: boolean
  response_body_truncated?: boolean
}

interface ReplayListResponse {
  object: string
  count: number
  data: ReplayRecord[]
}

type FilterType = 'all' | 'cached' | 'streamed'

const ReplayPage: React.FC = () => {
  const { isReadonly } = useReadonly()
  const [records, setRecords] = useState<ReplayRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(false)
  
  // UI state
  const [searchTerm, setSearchTerm] = useState('')
  const [filter, setFilter] = useState<FilterType>('all')
  const [decisionFilter, setDecisionFilter] = useState<string>('all')
  const [modelFilter, setModelFilter] = useState<string>('all')
  
  // ViewModal state
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [selectedRecordForModal, setSelectedRecordForModal] = useState<ReplayRecord | null>(null)
  const [modalLoading, setModalLoading] = useState(false)

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(25)

  // Get unique decisions for filter dropdown
  const uniqueDecisions = useMemo(() => {
    const decisions = new Set<string>()
    records.forEach(r => {
      if (r.decision) decisions.add(r.decision)
    })
    return Array.from(decisions).sort()
  }, [records])

  // Get unique models for filter dropdown
  const uniqueModels = useMemo(() => {
    const models = new Set<string>()
    records.forEach(r => {
      if (r.selected_model) models.add(r.selected_model)
      if (r.original_model) models.add(r.original_model)
    })
    return Array.from(models).sort()
  }, [records])

  const fetchRecords = useCallback(async () => {
    try {
      const response = await fetch('/api/router/v1/router_replay')
      if (!response.ok) {
        throw new Error(`Failed to fetch records: ${response.statusText}`)
      }
      const data: ReplayListResponse = await response.json()
      // Sort by timestamp descending (newest first)
      const sortedData = (data.data || []).sort((a, b) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )
      setRecords(sortedData)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchRecords()
    
    if (autoRefresh) {
      const interval = setInterval(fetchRecords, 5000)
      return () => clearInterval(interval)
    }
  }, [fetchRecords, autoRefresh])

  // Filter and search records
  const filteredRecords = useMemo(() => {
    return records.filter(record => {
      // Apply cache/stream filter
      if (filter === 'cached' && !record.from_cache) return false
      if (filter === 'streamed' && !record.streaming) return false
      
      // Apply decision filter
      if (decisionFilter !== 'all' && record.decision !== decisionFilter) return false
      
      // Apply model filter
      if (modelFilter !== 'all') {
        const matchesModel = record.selected_model === modelFilter || record.original_model === modelFilter
        if (!matchesModel) return false
      }
      
      // Apply search (by request ID only, per requirements)
      if (searchTerm) {
        const term = searchTerm.toLowerCase()
        const matchesRequestId = record.request_id?.toLowerCase().includes(term)
        if (!matchesRequestId) return false
      }
      
      return true
    })
  }, [records, filter, decisionFilter, modelFilter, searchTerm])

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1)
  }, [filter, decisionFilter, modelFilter, searchTerm])

  // Pagination calculations
  const totalPages = Math.ceil(filteredRecords.length / pageSize)
  const paginatedRecords = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize
    return filteredRecords.slice(startIndex, startIndex + pageSize)
  }, [filteredRecords, currentPage, pageSize])

  const formatJson = (jsonStr: string | undefined) => {
    if (!jsonStr) return null
    try {
      return JSON.stringify(JSON.parse(jsonStr), null, 2)
    } catch {
      return jsonStr
    }
  }


  // Build ViewModal sections from record (according to requirements)
  const buildRecordSections = (record: ReplayRecord): ViewSection[] => {
    const sections: ViewSection[] = []

    // 1. Decision Information
    sections.push({
      title: 'Decision Information',
      fields: [
        { label: 'Decision name', value: record.decision || '-' },
        {
          label: 'Category',
          value: record.signals?.domain?.length
            ? record.signals.domain.join(', ')
            : record.category || '-'
        },
        {
          label: 'Confidence score',
          value: record.confidence_score !== undefined
            ? `${(record.confidence_score * 100).toFixed(1)}%`
            : '-'
        },
        { label: 'Reasoning mode', value: record.reasoning_mode || '-' }
      ]
    })

    // 2. Model Selection
    sections.push({
      title: 'Model Selection',
      fields: [
        { label: 'Original model', value: record.original_model || '-' },
        { label: 'Selected model', value: record.selected_model || '-' },
        { label: 'Selection method/strategy', value: record.selection_method || '-' }
      ]
    })

    // 3. Signals
    const signalFields: ViewField[] = []
    
    if (record.signals?.keyword?.length) {
      signalFields.push({
        label: 'Keyword matches',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.keyword.map((kw, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(118, 185, 0, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem',
                fontFamily: 'var(--font-mono)'
              }}>
                {kw}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }
    
    if (record.signals?.embedding?.length) {
      signalFields.push({
        label: 'Embedding matches',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.embedding.map((emb, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(0, 212, 255, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem',
                fontFamily: 'var(--font-mono)'
              }}>
                {emb}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }
    
    if (record.signals?.domain?.length) {
      signalFields.push({
        label: 'Domain matches',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.domain.map((dom, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(147, 51, 234, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem'
              }}>
                {dom}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }
    
    if (record.signals?.fact_check?.length) {
      signalFields.push({
        label: 'Fact check results',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.fact_check.map((fc, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(34, 197, 94, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem'
              }}>
                {fc}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }
    
    if (record.signals?.user_feedback?.length) {
      signalFields.push({
        label: 'User feedback',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.user_feedback.map((uf, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(236, 72, 153, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem'
              }}>
                {uf}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }
    
    if (record.signals?.preference?.length) {
      signalFields.push({
        label: 'Preference signals',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.preference.map((pref, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(234, 179, 8, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem'
              }}>
                {pref}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }

    if (record.signals?.language?.length) {
      signalFields.push({
        label: 'Language signals',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.language.map((lang, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(59, 130, 246, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem'
              }}>
                {lang}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }

    if (record.signals?.latency?.length) {
      signalFields.push({
        label: 'Latency signals',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.latency.map((lat, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(245, 158, 11, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem'
              }}>
                {lat}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }

    if (record.signals?.context?.length) {
      signalFields.push({
        label: 'Context signals',
        value: (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {record.signals.context.map((ctx, i) => (
              <span key={i} style={{
                padding: '0.25rem 0.75rem',
                background: 'rgba(168, 85, 247, 0.1)',
                borderRadius: '4px',
                fontSize: '0.875rem'
              }}>
                {ctx}
              </span>
            ))}
          </div>
        ),
        fullWidth: true
      })
    }

    if (signalFields.length > 0) {
      sections.push({
        title: 'Signals',
        fields: signalFields
      })
    }

    // 4. Plugin Status
    sections.push({
      title: 'Plugin Status',
      fields: [
        {
          label: 'Cache',
          value: record.from_cache ? 'Hit' : 'Miss'
        },
        {
          label: 'Streaming',
          value: record.streaming ? 'On' : 'Off'
        },
        {
          label: 'Response status code',
          value: record.response_status ? (
            <span style={{
              padding: '0.25rem 0.75rem',
              background: record.response_status < 400 ? 'rgba(118, 185, 0, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              borderRadius: '4px',
              fontSize: '0.875rem',
              fontWeight: 600
            }}>
              {record.response_status}
            </span>
          ) : '-'
        }
      ]
    })

    // 5. Request/Response (Collapsible)
    const requestResponseFields: ViewField[] = []

    if (isReadonly) {
      // In readonly mode, show lock icon and message
      if (record.request_body || record.response_body) {
        requestResponseFields.push({
          label: 'Request body',
          value: (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 0.75rem',
              color: 'var(--color-text-tertiary)',
              fontSize: '0.875rem'
            }}>
              <span style={{ fontSize: '1rem' }}>🔒</span>
              <span>Not available in read-only mode</span>
            </div>
          ),
          fullWidth: false
        })

        requestResponseFields.push({
          label: 'Response body',
          value: (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 0.75rem',
              color: 'var(--color-text-tertiary)',
              fontSize: '0.875rem'
            }}>
              <span style={{ fontSize: '1rem' }}>🔒</span>
              <span>Not available in read-only mode</span>
            </div>
          ),
          fullWidth: false
        })
      }
    } else {
      // Normal mode: show request/response bodies
      if (record.request_body) {
        requestResponseFields.push({
          label: 'Request body',
          value: (
            <CollapsibleSection
              id={`request-${record.id}`}
              title="request body"
              isTruncated={record.request_body_truncated || false}
              defaultExpanded={false}
              content={
                <pre style={{
                  margin: 0,
                  padding: '0.75rem',
                  background: 'var(--color-bg-tertiary)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  fontFamily: 'var(--font-mono)',
                  maxHeight: '400px',
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {formatJson(record.request_body) || record.request_body}
                </pre>
              }
            />
          ),
          fullWidth: true
        })
      }

      if (record.response_body) {
        requestResponseFields.push({
          label: 'Response body',
          value: (
            <CollapsibleSection
              id={`response-${record.id}`}
              title="response body"
              isTruncated={record.response_body_truncated || false}
              defaultExpanded={false}
              content={
                <pre style={{
                  margin: 0,
                  padding: '0.75rem',
                  background: 'var(--color-bg-tertiary)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '4px',
                  fontSize: '0.8rem',
                  fontFamily: 'var(--font-mono)',
                  maxHeight: '400px',
                  overflow: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word'
                }}>
                  {formatJson(record.response_body) || record.response_body}
                </pre>
              }
            />
          ),
          fullWidth: true
        })
      }
    }

    if (requestResponseFields.length > 0) {
      sections.push({
        title: 'Request/Response',
        fields: requestResponseFields
      })
    }

    return sections
  }

  // Handle view record in modal - fetch fresh data from API
  const handleViewRecord = async (record: ReplayRecord) => {
    setModalLoading(true)
    setViewModalOpen(true)
    
    try {
      // Fetch fresh record data by ID to ensure we have latest data
      const response = await fetch(`/api/router/v1/router_replay/${record.id}`)
      if (response.ok) {
        const freshRecord = await response.json()
        setSelectedRecordForModal(freshRecord)
      } else {
        // Fallback to cached data if API call fails
        setSelectedRecordForModal(record)
      }
    } catch {
      // Fallback to cached data on error
      setSelectedRecordForModal(record)
    } finally {
      setModalLoading(false)
    }
  }
  
  // Handle close modal
  const handleCloseModal = () => {
    setViewModalOpen(false)
    setSelectedRecordForModal(null)
  }

  // Helper function to collect all signal names
  const collectSignals = (signals: Signal): string[] => {
    const allSignals: string[] = []
    if (signals.keyword?.length) allSignals.push(...signals.keyword)
    if (signals.embedding?.length) allSignals.push(...signals.embedding)
    if (signals.domain?.length) allSignals.push(...signals.domain)
    if (signals.fact_check?.length) allSignals.push(...signals.fact_check)
    if (signals.user_feedback?.length) allSignals.push(...signals.user_feedback)
    if (signals.preference?.length) allSignals.push(...signals.preference)
    if (signals.language?.length) allSignals.push(...signals.language)
    if (signals.latency?.length) allSignals.push(...signals.latency)
    if (signals.context?.length) allSignals.push(...signals.context)
    return allSignals
  }

  // Define table columns for DataTable component
  const tableColumns: Column<ReplayRecord>[] = [
    {
      key: 'request_id',
      header: 'Request ID',
      width: '280px',
      render: (row) => (
        <span className={styles.requestId} title={row.request_id || ''}>
          {row.request_id || '-'}
        </span>
      )
    },
    {
      key: 'timestamp',
      header: 'Created',
      width: '130px',
      sortable: true,
      render: (row) => (
        <span className={styles.timestamp}>{formatDate(row.timestamp)}</span>
      )
    },
    {
      key: 'decision',
      header: 'Decision',
      width: '180px',
      sortable: true,
      render: (row) => <span className={styles.decision}>{row.decision || '-'}</span>
    },
    {
      key: 'signals',
      header: 'Signals',
      width: '200px',
      render: (row) => {
        const allSignals = collectSignals(row.signals)
        if (allSignals.length === 0) return <span>-</span>

        return (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
            {allSignals.slice(0, 3).map((signal, i) => (
              <span key={i} style={{
                padding: '0.125rem 0.5rem',
                background: 'rgba(118, 185, 0, 0.15)',
                border: '1px solid rgba(118, 185, 0, 0.3)',
                borderRadius: '3px',
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                whiteSpace: 'nowrap'
              }}>
                {signal}
              </span>
            ))}
            {allSignals.length > 3 && (
              <span style={{
                padding: '0.125rem 0.5rem',
                fontSize: '0.75rem',
                color: 'var(--color-text-secondary)'
              }}>
                +{allSignals.length - 3}
              </span>
            )}
          </div>
        )
      }
    },
    {
      key: 'reasoning',
      header: 'Reasoning',
      width: '90px',
      align: 'center',
      render: (row) => (
        <span className={`${styles.reasoningBadge} ${
          row.reasoning_mode === 'on'
            ? styles.reasoningOn
            : styles.reasoningOff
        }`}>
          {row.reasoning_mode === 'on' ? 'On' : 'Off'}
        </span>
      )
    },
    {
      key: 'model',
      header: 'Model Change',
      width: '320px',
      render: (row) => (
        <div className={styles.modelChange}>
          <span className={styles.modelName}>{row.original_model || '-'}</span>
          <span className={styles.modelArrow}>→</span>
          <span className={styles.modelName}>{row.selected_model || '-'}</span>
        </div>
      )
    },
    {
      key: 'status',
      header: 'Status',
      width: '70px',
      align: 'center',
      render: (row) => (
        <span className={`${styles.statusBadge} ${
          row.response_status && row.response_status < 400
            ? styles.statusSuccess
            : styles.statusError
        }`}>
          {row.response_status || '-'}
        </span>
      )
    },
    {
      key: 'flags',
      header: 'Flags',
      width: '160px',
      render: (row) => (
        <div className={styles.indicators}>
          <span className={`${styles.indicator} ${row.from_cache ? styles.indicatorActive : ''}`}>
            <svg className={styles.indicatorIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            Cache
          </span>
          <span className={`${styles.indicator} ${row.streaming ? styles.indicatorActive : ''}`}>
            <svg className={styles.indicatorIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            Stream
          </span>
        </div>
      )
    }
  ]


  // Loading state
  if (loading && records.length === 0) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner}></div>
          <p>Loading router replay records...</p>
        </div>
      </div>
    )
  }

  // List View
  return ( 
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>
            <svg className={styles.titleIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Router Replay
          </h1>
          <p className={styles.subtitle}>
            View and analyze routing decision records for debugging and analysis.
          </p>
        </div>
        <div className={styles.headerRight}>
          <label className={styles.autoRefreshToggle}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            <span>Auto-refresh</span>
          </label>
          <button onClick={fetchRecords} className={styles.refreshButton}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M23 4v6h-6M1 20v-6h6" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Refresh
          </button>
        </div>
      </div>

      {/* Only show error banner if we have records (refresh failed) - not when router_replay is disabled */}
      {error && records.length > 0 && (
        <div className={styles.error}>
          <svg className={styles.errorIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* Statistics Charts */}
      {records.length > 0 && <ReplayCharts records={records} />}

      <TableHeader
        title="Routing Records"
        count={filteredRecords.length}
        searchPlaceholder="Search by Request ID..."
        searchValue={searchTerm}
        onSearchChange={setSearchTerm}
      />

      <div className={styles.filtersRow}>
        <select
          className={styles.filterSelect}
          value={decisionFilter}
          onChange={(e) => setDecisionFilter(e.target.value)}
          disabled={uniqueDecisions.length === 0}
        >
          <option value="all">All Decisions</option>
          {uniqueDecisions.map(decision => (
            <option key={decision} value={decision}>{decision}</option>
          ))}
        </select>
        <select
          className={styles.filterSelect}
          value={modelFilter}
          onChange={(e) => setModelFilter(e.target.value)}
          disabled={uniqueModels.length === 0}
        >
          <option value="all">All Models</option>
          {uniqueModels.map(model => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
        <select
          className={styles.filterSelect}
          value={filter}
          onChange={(e) => setFilter(e.target.value as FilterType)}
        >
          <option value="all">Cache Status</option>
          <option value="cached">Cached Only</option>
          <option value="streamed">Streamed Only</option>
        </select>
      </div>

      <div className={styles.recordsSection}>

        {/* Show config hint only when NO records exist at all (not from filtering) */}
        {records.length === 0 && !loading && (
          <div className={styles.emptyState}>
            {error ? (
              <div className={styles.emptyHint}>
                <p>Router Replay may not be enabled. To enable it, add this to your config.yaml:</p>
                <pre className={styles.configHint}>{`router_replay:
  enabled: true
  store_backend: memory  # or redis, postgres
  max_records: 200
  capture_request_body: true
  capture_response_body: true`}</pre>
                <p className={styles.emptySubtext}>Then restart the router and send some requests.</p>
              </div>
            ) : (
              <div className={styles.emptyHint}>
                <p>Router replay records will appear here once requests are processed.</p>
                <p className={styles.emptySubtext}>Send a chat completion request through the router to see it here.</p>
              </div>
            )}
          </div>
        )}

        {/* Always show DataTable - it handles empty state internally */}
        {(records.length > 0 || loading) && (
          <DataTable
            columns={tableColumns}
            data={paginatedRecords}
            keyExtractor={(row) => row.id}
            onView={handleViewRecord}
            emptyMessage="No records match your search"
          />
        )}
      </div>

      {/* Pagination Controls - outside scrollable area */}
      {totalPages > 1 && (
        <div className={styles.pagination}>
          <button
            className={styles.paginationButton}
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
          >
            First
          </button>
          <button
            className={styles.paginationButton}
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
          >
            Previous
          </button>
          
          <span className={styles.paginationInfo}>
            Page {currentPage} of {totalPages} ({filteredRecords.length} records)
          </span>
          
          <button
            className={styles.paginationButton}
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
          >
            Next
          </button>
          <button
            className={styles.paginationButton}
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage === totalPages}
          >
            Last
          </button>
        </div>
      )}


      {/* View Modal */}
      <ViewModal
        isOpen={viewModalOpen}
        onClose={handleCloseModal}
        title={modalLoading 
          ? 'Loading...'
          : selectedRecordForModal?.request_id 
            ? `Record: ${selectedRecordForModal.request_id.substring(0, 8)}...`
            : `Record: ${selectedRecordForModal?.decision || selectedRecordForModal?.id || ''}`
        }
        sections={selectedRecordForModal ? buildRecordSections(selectedRecordForModal) : []}
      />
    </div>
  )
}

export default ReplayPage
