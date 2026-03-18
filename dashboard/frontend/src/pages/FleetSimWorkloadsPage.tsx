import { useEffect, useState } from 'react'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import FleetSimSurfaceLayout from './FleetSimSurfaceLayout'
import styles from './FleetSimPage.module.css'
import {
  deleteTrace,
  getTraceSample,
  getWorkloadStats,
  listTraces,
  listWorkloads,
  uploadTrace,
  type BuiltinWorkload,
  type FleetSimTraceFormat,
  type TraceInfo,
  type TraceSample,
} from '../utils/fleetSimApi'
import { formatDateTime, formatNumber } from './fleetSimPageSupport'

type BuiltinWorkloadWithStats = BuiltinWorkload & {
  statsSummary?: string
}

export default function FleetSimWorkloadsPage() {
  const [workloads, setWorkloads] = useState<BuiltinWorkloadWithStats[]>([])
  const [traces, setTraces] = useState<TraceInfo[]>([])
  const [selectedTrace, setSelectedTrace] = useState<TraceInfo | null>(null)
  const [selectedSample, setSelectedSample] = useState<TraceSample | null>(null)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadFormat, setUploadFormat] = useState<FleetSimTraceFormat>('jsonl')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')

  const load = async () => {
    const [workloadsData, tracesData] = await Promise.all([listWorkloads(), listTraces()])
    const enriched = await Promise.all(
      workloadsData.map(async (workload) => {
        const stats = await getWorkloadStats(workload.name)
        return {
          ...workload,
          stats,
          statsSummary: `P99 ${formatNumber(stats.p99_total_tokens)} tokens · ${formatNumber(stats.n_requests)} synthetic samples`,
        }
      })
    )
    setWorkloads(enriched)
    setTraces(tracesData)
  }

  useEffect(() => {
    void load().catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load workloads')
    })
  }, [])

  const handleUpload = async () => {
    if (!uploadFile) {
      setError('Choose a trace file before uploading.')
      return
    }
    try {
      await uploadTrace(uploadFile, uploadFormat)
      setUploadFile(null)
      setMessage(`Uploaded ${uploadFile.name}`)
      setError('')
      await load()
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Failed to upload trace')
    }
  }

  const handleDeleteTrace = async (trace: TraceInfo) => {
    try {
      await deleteTrace(trace.id)
      if (selectedTrace?.id === trace.id) {
        setSelectedTrace(null)
        setSelectedSample(null)
      }
      setMessage(`Deleted ${trace.name}`)
      setError('')
      await load()
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Failed to delete trace')
    }
  }

  const handleViewTrace = async (trace: TraceInfo) => {
    try {
      const sample = await getTraceSample(trace.id)
      setSelectedTrace(trace)
      setSelectedSample(sample)
      setError('')
    } catch (sampleError) {
      setError(sampleError instanceof Error ? sampleError.message : 'Failed to fetch trace sample')
    }
  }

  const workloadColumns: Column<BuiltinWorkloadWithStats>[] = [
    {
      key: 'name',
      header: 'Workload',
      sortable: true,
      render: (row) => (
        <div className={styles.compactListMeta}>
          <span className={styles.compactListTitle}>{row.name}</span>
          <span className={styles.compactListText}>{row.description}</span>
        </div>
      ),
    },
    {
      key: 'statsSummary',
      header: 'Profile',
      render: (row) => row.statsSummary || 'Loading…',
    },
    {
      key: 'path',
      header: 'Bundle Path',
      render: (row) => <span className={styles.mono}>{row.path.split('/').slice(-1)[0]}</span>,
    },
  ]

  const traceColumns: Column<TraceInfo>[] = [
    {
      key: 'name',
      header: 'Trace',
      sortable: true,
      render: (row) => (
        <div className={styles.compactListMeta}>
          <span className={styles.compactListTitle}>{row.name}</span>
          <span className={styles.compactListText}>{formatNumber(row.n_requests)} requests</span>
        </div>
      ),
    },
    {
      key: 'format',
      header: 'Format',
      sortable: true,
      render: (row) => <span className={styles.mono}>{row.format}</span>,
    },
    {
      key: 'upload_time',
      header: 'Uploaded',
      sortable: true,
      render: (row) => formatDateTime(row.upload_time),
    },
  ]

  const filteredTraces = traces.filter((trace) =>
    trace.name.toLowerCase().includes(search.toLowerCase()) || trace.format.includes(search.toLowerCase())
  )

  return (
    <FleetSimSurfaceLayout
      title="Workloads"
      description="Curate the trace library that powers fleet sizing. Built-in workload CDFs stay bundled with the package, while uploaded request traces persist in the shared simulator state."
      currentPath="/fleet-sim/workloads"
      meta={[
        { label: 'Built-ins', value: formatNumber(workloads.length) },
        { label: 'Uploaded traces', value: formatNumber(traces.length) },
        { label: 'Sample preview', value: selectedSample ? formatNumber(selectedSample.total) : 'Idle' },
      ]}
    >
      <div className={styles.splitGrid}>
        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Trace Intake</h2>
              <p className={styles.sectionDescription}>Upload semantic-router JSONL, generic JSONL, or CSV traces and keep them available to every dashboard session on this workspace.</p>
            </div>
          </div>
          <div className={styles.formGrid}>
            <label className={`${styles.field} ${styles.fieldWide}`}>
              <span className={styles.fieldLabel}>Trace File</span>
              <input
                className={styles.fileInput}
                type="file"
                onChange={(event) => setUploadFile(event.target.files?.[0] || null)}
              />
            </label>
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Format</span>
              <select
                className={styles.select}
                value={uploadFormat}
                onChange={(event) => setUploadFormat(event.target.value as FleetSimTraceFormat)}
              >
                <option value="jsonl">jsonl</option>
                <option value="semantic_router">semantic_router</option>
                <option value="csv">csv</option>
              </select>
            </label>
          </div>
          <div className={styles.buttonRow} style={{ marginTop: '0.9rem' }}>
            <button type="button" className={styles.primaryButton} onClick={() => void handleUpload()}>
              Upload Trace
            </button>
            <span className={styles.inlineHint}>The backend proxy forwards uploads straight to the sidecar API.</span>
          </div>
          {message ? <p className={`${styles.message} ${styles.messageSuccess}`}>{message}</p> : null}
          {error ? <p className={`${styles.message} ${styles.messageError}`}>{error}</p> : null}
        </section>

        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Trace Sample</h2>
              <p className={styles.sectionDescription}>Use a lightweight sample to verify that uploaded fields match the simulator parser before submitting runs.</p>
            </div>
          </div>
          {selectedTrace && selectedSample ? (
            <>
              <p className={styles.message}>
                {selectedTrace.name} · showing {Math.min(selectedSample.records.length, selectedSample.total)} of {selectedSample.total} rows
              </p>
              <pre className={styles.jsonPreview}>{JSON.stringify(selectedSample.records, null, 2)}</pre>
            </>
          ) : (
            <div className={styles.emptyState}>Select “View” on a trace row to inspect a request sample.</div>
          )}
        </section>
      </div>

      <section className={styles.sectionCard}>
        <TableHeader title="Built-in Workloads" count={workloads.length} variant="embedded" />
        <DataTable
          columns={workloadColumns}
          data={workloads}
          keyExtractor={(row) => row.name}
          emptyMessage="No built-in workloads are available."
        />
      </section>

      <section className={styles.sectionCard}>
        <TableHeader
          title="Uploaded Traces"
          count={filteredTraces.length}
          searchPlaceholder="Search traces..."
          searchValue={search}
          onSearchChange={setSearch}
          variant="embedded"
        />
        <DataTable
          columns={traceColumns}
          data={filteredTraces}
          keyExtractor={(row) => row.id}
          onView={(row) => void handleViewTrace(row)}
          onDelete={(row) => void handleDeleteTrace(row)}
          emptyMessage="No uploaded traces yet."
        />
      </section>
    </FleetSimSurfaceLayout>
  )
}
