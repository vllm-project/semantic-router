import React, { useState, useCallback, useMemo } from 'react'
import { DataTable, Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import ViewModal, { ViewSection } from '../components/ViewModal'
import { formatDate } from '../types/evaluation'
import { useReadonly } from '../contexts/ReadonlyContext'
import styles from './MemoryPage.module.css'

interface MemoryRecord {
  id: string
  type: 'semantic' | 'procedural' | 'episodic'
  content: string
  user_id: string
  source?: string
  importance: number
  access_count: number
  created_at: string
  updated_at?: string
}

interface MemoryListResponse {
  memories: MemoryRecord[]
  total: number
  limit: number
}

interface MemoryDeleteResponse {
  success: boolean
  message: string
}

type MemoryTypeFilter = 'all' | 'semantic' | 'procedural' | 'episodic'

const USER_ID_PATTERN = /^[a-zA-Z0-9._@:/$-]+$/

const MemoryPage: React.FC = () => {
  const { isReadonly } = useReadonly()

  // User ID state
  const [userId, setUserId] = useState('')
  const [activeUserId, setActiveUserId] = useState('')

  // Data state
  const [memories, setMemories] = useState<MemoryRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [memoryNotConfigured, setMemoryNotConfigured] = useState(false)

  // UI state
  const [searchTerm, setSearchTerm] = useState('')
  const [typeFilter, setTypeFilter] = useState<MemoryTypeFilter>('all')

  // ViewModal state
  const [viewModalOpen, setViewModalOpen] = useState(false)
  const [selectedMemory, setSelectedMemory] = useState<MemoryRecord | null>(null)
  const [modalLoading, setModalLoading] = useState(false)

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(25)

  const fetchMemories = useCallback(async (uid: string) => {
    if (!uid) return

    setLoading(true)
    setError(null)
    setMemoryNotConfigured(false)

    try {
      let url = `/api/router/v1/memory?user_id=${encodeURIComponent(uid)}&limit=100`
      if (typeFilter !== 'all') {
        url += `&type=${typeFilter}`
      }

      const response = await fetch(url)

      if (response.status === 503) {
        setMemoryNotConfigured(true)
        setMemories([])
        return
      }

      if (response.status === 401) {
        setError('Enter a User ID to explore memories')
        setMemories([])
        return
      }

      if (response.status === 400) {
        const body = await response.json().catch(() => ({}))
        if (body.code === 'INVALID_USER_ID') {
          setError('User ID contains invalid characters')
        } else {
          setError(body.message || 'Bad request')
        }
        setMemories([])
        return
      }

      if (!response.ok) {
        throw new Error(`Failed to fetch memories: ${response.statusText}`)
      }

      const data: MemoryListResponse = await response.json()
      setMemories(data.memories || [])
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Failed to connect to router API')
      } else {
        setError(err instanceof Error ? err.message : 'Unknown error')
      }
      setMemories([])
    } finally {
      setLoading(false)
    }
  }, [typeFilter])

  const handleLoadMemories = useCallback(() => {
    const trimmed = userId.trim()
    if (!trimmed) {
      setError('Enter a User ID to explore memories')
      return
    }
    if (!USER_ID_PATTERN.test(trimmed)) {
      setError('User ID contains invalid characters')
      return
    }
    setActiveUserId(trimmed)
    setCurrentPage(1)
    fetchMemories(trimmed)
  }, [userId, fetchMemories])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleLoadMemories()
    }
  }, [handleLoadMemories])

  // Refetch when type filter changes (only if we have an active user)
  const handleTypeFilterChange = useCallback((newFilter: MemoryTypeFilter) => {
    setTypeFilter(newFilter)
    setCurrentPage(1)
    if (activeUserId) {
      // We need to fetch with the new filter value directly since state hasn't updated yet
      const fetchWithFilter = async () => {
        setLoading(true)
        setError(null)
        setMemoryNotConfigured(false)
        try {
          let url = `/api/router/v1/memory?user_id=${encodeURIComponent(activeUserId)}&limit=100`
          if (newFilter !== 'all') {
            url += `&type=${newFilter}`
          }
          const response = await fetch(url)
          if (response.status === 503) {
            setMemoryNotConfigured(true)
            setMemories([])
            return
          }
          if (!response.ok) {
            throw new Error(`Failed to fetch memories: ${response.statusText}`)
          }
          const data: MemoryListResponse = await response.json()
          setMemories(data.memories || [])
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Unknown error')
          setMemories([])
        } finally {
          setLoading(false)
        }
      }
      fetchWithFilter()
    }
  }, [activeUserId])

  // Client-side content search
  const filteredMemories = useMemo(() => {
    if (!searchTerm) return memories
    const term = searchTerm.toLowerCase()
    return memories.filter(m => m.content.toLowerCase().includes(term))
  }, [memories, searchTerm])

  // Reset to page 1 when search changes
  React.useEffect(() => {
    setCurrentPage(1)
  }, [searchTerm])

  // Pagination calculations
  const totalPages = Math.ceil(filteredMemories.length / pageSize)
  const paginatedMemories = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize
    return filteredMemories.slice(startIndex, startIndex + pageSize)
  }, [filteredMemories, currentPage, pageSize])

  // Delete single memory
  const handleDelete = useCallback(async (record: MemoryRecord) => {
    if (!window.confirm(`Delete this ${record.type} memory?\n\n"${record.content.substring(0, 100)}${record.content.length > 100 ? '...' : ''}"`)) {
      return
    }
    try {
      const response = await fetch(`/api/router/v1/memory/${record.id}?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'DELETE',
      })
      if (response.status === 404) {
        setError('Memory not found (may have been deleted)')
        setMemories(prev => prev.filter(m => m.id !== record.id))
        return
      }
      if (!response.ok) {
        throw new Error(`Failed to delete memory: ${response.statusText}`)
      }
      const data: MemoryDeleteResponse = await response.json()
      if (data.success) {
        setMemories(prev => prev.filter(m => m.id !== record.id))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete memory')
    }
  }, [activeUserId])

  // Bulk delete all
  const handleDeleteAll = useCallback(async () => {
    if (!window.confirm(`Delete ALL memories for user "${activeUserId}"?\n\nThis action cannot be undone.`)) {
      return
    }
    try {
      const response = await fetch(`/api/router/v1/memory?user_id=${encodeURIComponent(activeUserId)}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        throw new Error(`Failed to delete memories: ${response.statusText}`)
      }
      const data: MemoryDeleteResponse = await response.json()
      if (data.success) {
        setMemories([])
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete memories')
    }
  }, [activeUserId])

  // Bulk delete by type
  const handleDeleteByType = useCallback(async () => {
    if (!window.confirm(`Delete all "${typeFilter}" memories for user "${activeUserId}"?\n\nThis action cannot be undone.`)) {
      return
    }
    try {
      const response = await fetch(`/api/router/v1/memory?user_id=${encodeURIComponent(activeUserId)}&type=${typeFilter}`, {
        method: 'DELETE',
      })
      if (!response.ok) {
        throw new Error(`Failed to delete memories: ${response.statusText}`)
      }
      const data: MemoryDeleteResponse = await response.json()
      if (data.success) {
        setMemories(prev => prev.filter(m => m.type !== typeFilter))
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete memories')
    }
  }, [activeUserId, typeFilter])

  // View detail modal
  const handleViewMemory = useCallback(async (record: MemoryRecord) => {
    setModalLoading(true)
    setViewModalOpen(true)

    try {
      const response = await fetch(`/api/router/v1/memory/${record.id}?user_id=${encodeURIComponent(activeUserId)}`)
      if (response.ok) {
        const freshRecord = await response.json()
        setSelectedMemory(freshRecord)
      } else {
        setSelectedMemory(record)
      }
    } catch {
      setSelectedMemory(record)
    } finally {
      setModalLoading(false)
    }
  }, [activeUserId])

  const handleCloseModal = () => {
    setViewModalOpen(false)
    setSelectedMemory(null)
  }

  // Build modal sections
  const buildMemorySections = (record: MemoryRecord): ViewSection[] => {
    const sections: ViewSection[] = []

    // Memory Info
    const typeLabel = record.type.charAt(0).toUpperCase() + record.type.slice(1)
    sections.push({
      title: 'Memory Info',
      fields: [
        {
          label: 'Type',
          value: (
            <span className={`${styles.typeBadge} ${
              record.type === 'semantic' ? styles.typeSemantic :
              record.type === 'procedural' ? styles.typeProcedural :
              styles.typeEpisodic
            }`}>
              {typeLabel}
            </span>
          ),
        },
        {
          label: 'Source',
          value: record.source || '\u2014',
        },
        {
          label: 'Importance',
          value: `${(record.importance * 100).toFixed(0)}%`,
        },
        {
          label: 'Access Count',
          value: String(record.access_count),
        },
      ],
    })

    // Content
    sections.push({
      title: 'Content',
      fields: [
        {
          label: 'Full Content',
          value: (
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
              wordBreak: 'break-word',
            }}>
              {record.content}
            </pre>
          ),
          fullWidth: true,
        },
      ],
    })

    // Timestamps
    sections.push({
      title: 'Timestamps',
      fields: [
        { label: 'Created', value: formatDate(record.created_at) },
        { label: 'Updated', value: formatDate(record.updated_at) },
      ],
    })

    return sections
  }

  // Importance color coding
  const getImportanceStyle = (importance: number): React.CSSProperties => {
    if (importance >= 0.7) {
      return { background: 'rgba(118, 185, 0, 0.15)', color: 'var(--color-primary)' }
    } else if (importance >= 0.4) {
      return { background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b' }
    }
    return { background: 'rgba(255, 255, 255, 0.05)', color: 'var(--color-text-secondary)' }
  }

  // Table columns
  const tableColumns: Column<MemoryRecord>[] = [
    {
      key: 'type',
      header: 'Type',
      width: '120px',
      sortable: true,
      render: (row) => {
        const label = row.type.charAt(0).toUpperCase() + row.type.slice(1)
        return (
          <span className={`${styles.typeBadge} ${
            row.type === 'semantic' ? styles.typeSemantic :
            row.type === 'procedural' ? styles.typeProcedural :
            styles.typeEpisodic
          }`}>
            {label}
          </span>
        )
      },
    },
    {
      key: 'content',
      header: 'Content',
      render: (row) => (
        <span className={styles.contentPreview} title={row.content}>
          {row.content.length > 100 ? row.content.substring(0, 100) + '...' : row.content}
        </span>
      ),
    },
    {
      key: 'source',
      header: 'Source',
      width: '120px',
      sortable: true,
      render: (row) => <span>{row.source || '\u2014'}</span>,
    },
    {
      key: 'importance',
      header: 'Importance',
      width: '100px',
      sortable: true,
      render: (row) => (
        <span className={styles.importanceBadge} style={getImportanceStyle(row.importance)}>
          {(row.importance * 100).toFixed(0)}%
        </span>
      ),
    },
    {
      key: 'access_count',
      header: 'Access Count',
      width: '100px',
      sortable: true,
      render: (row) => <span>{row.access_count}</span>,
    },
    {
      key: 'created_at',
      header: 'Created',
      width: '140px',
      sortable: true,
      render: (row) => <span style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>{formatDate(row.created_at)}</span>,
    },
  ]

  // Memory not configured state
  if (memoryNotConfigured) {
    return (
      <div className={styles.container}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            <h1 className={styles.title}>
              <svg className={styles.titleIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z" strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M10 21h4" strokeLinecap="round"/>
              </svg>
              Memory
            </h1>
            <p className={styles.subtitle}>Browse, inspect, and manage agentic memories.</p>
          </div>
        </div>
        <div className={styles.emptyState}>
          <div className={styles.emptyHint}>
            <p>Memory store is not configured. To enable it, add this to your config.yaml:</p>
            <pre className={styles.configHint}>{`memory:
  enabled: true
  backend: milvus
  milvus:
    address: localhost:19530
    database: default`}</pre>
            <p className={styles.emptySubtext}>Then restart the router.</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>
            <svg className={styles.titleIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2a7 7 0 0 1 7 7c0 2.38-1.19 4.47-3 5.74V17a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2v-2.26C6.19 13.47 5 11.38 5 9a7 7 0 0 1 7-7z" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M10 21h4" strokeLinecap="round"/>
            </svg>
            Memory
          </h1>
          <p className={styles.subtitle}>Browse, inspect, and manage agentic memories.</p>
        </div>
        <div className={styles.headerRight}>
          {!isReadonly && activeUserId && memories.length > 0 && (
            <div className={styles.bulkActions}>
              {typeFilter !== 'all' && (
                <button className={styles.dangerButton} onClick={handleDeleteByType}>
                  Delete {typeFilter.charAt(0).toUpperCase() + typeFilter.slice(1)}
                </button>
              )}
              <button className={styles.dangerButton} onClick={handleDeleteAll}>
                Delete All
              </button>
            </div>
          )}
        </div>
      </div>

      {/* User ID input bar */}
      <div className={styles.userIdSection}>
        <input
          type="text"
          className={styles.userIdInput}
          placeholder="Enter User ID (e.g. user@example.com)"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className={styles.loadButton}
          onClick={handleLoadMemories}
          disabled={loading || !userId.trim()}
        >
          {loading ? (
            <>
              <div className={styles.spinner} style={{ width: 16, height: 16, borderWidth: 2 }}></div>
              Loading...
            </>
          ) : (
            'Load Memories'
          )}
        </button>
      </div>

      {/* Error banner */}
      {error && (
        <div className={styles.error}>
          <svg className={styles.errorIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>{error}</span>
        </div>
      )}

      {/* Only show table/filters when we have an active user */}
      {activeUserId && !memoryNotConfigured && (
        <>
          <TableHeader
            title="Memories"
            count={filteredMemories.length}
            searchPlaceholder="Search by content..."
            searchValue={searchTerm}
            onSearchChange={setSearchTerm}
          />

          <div className={styles.filtersRow}>
            <select
              className={styles.filterSelect}
              value={typeFilter}
              onChange={(e) => handleTypeFilterChange(e.target.value as MemoryTypeFilter)}
            >
              <option value="all">All Types</option>
              <option value="semantic">Semantic</option>
              <option value="procedural">Procedural</option>
              <option value="episodic">Episodic</option>
            </select>
          </div>

          {loading && memories.length === 0 ? (
            <div className={styles.loading}>
              <div className={styles.spinner}></div>
              <p>Loading memories...</p>
            </div>
          ) : memories.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyHint}>
                <p>No memories found for user &ldquo;{activeUserId}&rdquo;.</p>
                <p className={styles.emptySubtext}>Memories are created during agentic conversations.</p>
              </div>
            </div>
          ) : (
            <DataTable
              columns={tableColumns}
              data={paginatedMemories}
              keyExtractor={(row) => row.id}
              onView={handleViewMemory}
              onDelete={handleDelete}
              readonly={isReadonly}
              emptyMessage="No memories match your search"
            />
          )}

          {/* Pagination */}
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
                Page {currentPage} of {totalPages} ({filteredMemories.length} memories)
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
        </>
      )}

      {/* Prompt to enter user ID when no active user */}
      {!activeUserId && !error && (
        <div className={styles.emptyState}>
          <div className={styles.emptyHint}>
            <p>Enter a User ID above to explore their memories.</p>
            <p className={styles.emptySubtext}>Memories are stored per user and require a user ID to access.</p>
          </div>
        </div>
      )}

      {/* View Modal */}
      <ViewModal
        isOpen={viewModalOpen}
        onClose={handleCloseModal}
        title={modalLoading
          ? 'Loading...'
          : selectedMemory
            ? `Memory: ${selectedMemory.type} - ${selectedMemory.content.substring(0, 40)}${selectedMemory.content.length > 40 ? '...' : ''}`
            : 'Memory'
        }
        sections={selectedMemory ? buildMemorySections(selectedMemory) : []}
      />
    </div>
  )
}

export default MemoryPage
