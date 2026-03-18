import { useEffect, useRef, useState, type ChangeEvent } from 'react'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import FleetSimSurfaceLayout from './FleetSimSurfaceLayout'
import FleetSimTracePreviewDialog from './FleetSimTracePreviewDialog'
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
import {
  describeBuiltinWorkload,
  formatBuiltinWorkloadName,
  formatDateTime,
  formatNumber,
  formatTraceFormat,
} from './fleetSimPageSupport'

type BuiltinWorkloadWithStats = BuiltinWorkload & {
  statsSummary?: string
}

export default function FleetSimWorkloadsPage() {
  const [workloads, setWorkloads] = useState<BuiltinWorkloadWithStats[]>([])
  const [traces, setTraces] = useState<TraceInfo[]>([])
  const [selectedTrace, setSelectedTrace] = useState<TraceInfo | null>(null)
  const [selectedSample, setSelectedSample] = useState<TraceSample | null>(null)
  const [uploadFormat, setUploadFormat] = useState<FleetSimTraceFormat>('jsonl')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const fileInputRef = useRef<HTMLInputElement | null>(null)

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

  const handleUpload = async (file: File) => {
    if (!file) {
      setError('Choose a trace file before uploading.')
      return
    }
    try {
      await uploadTrace(file, uploadFormat)
      setMessage(`Uploaded ${file.name}`)
      setError('')
      await load()
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Failed to upload trace')
    }
  }

  const handleUploadButtonClick = () => {
    fileInputRef.current?.click()
  }

  const handleUploadSelection = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    await handleUpload(file)
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

  const handleClosePreview = () => {
    setSelectedTrace(null)
    setSelectedSample(null)
  }

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
      render: (row) => <span className={styles.mono}>{formatTraceFormat(row.format)}</span>,
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
      description="Curate the traffic inputs behind every planning run, from reusable library profiles to uploaded production traces."
      currentPath="/fleet-sim/workloads"
      meta={[
        { label: 'Library profiles', value: formatNumber(workloads.length) },
        { label: 'Uploaded traces', value: formatNumber(traces.length) },
        { label: 'Ingest formats', value: 'JSONL · Router JSONL · CSV' },
      ]}
    >
      <section className={styles.sectionCard}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.sectionKicker}>Upload</span>
            <h2 className={styles.sectionTitle}>Trace Intake</h2>
            <p className={styles.sectionDescription}>Add replayable traces for traffic you want to compare against the built-in planning library.</p>
          </div>
        </div>
        <div className={styles.formStack}>
          <div className={styles.fieldSection}>
            <div>
              <span className={styles.sectionKicker}>Trace source</span>
              <p className={styles.sectionDescription}>Use the same intake formats the simulator already understands and keep them searchable from the dashboard.</p>
            </div>
            <div className={styles.formGrid}>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Format</span>
                <select
                  className={styles.select}
                  value={uploadFormat}
                  onChange={(event) => setUploadFormat(event.target.value as FleetSimTraceFormat)}
                >
                  <option value="jsonl">JSONL</option>
                  <option value="semantic_router">Router JSONL</option>
                  <option value="csv">CSV</option>
                </select>
              </label>
            </div>
          </div>
        </div>
        <div className={styles.buttonRow} style={{ marginTop: '0.9rem' }}>
          <input
            ref={fileInputRef}
            className={styles.hiddenFileInput}
            type="file"
            onChange={(event) => void handleUploadSelection(event)}
          />
          <button type="button" className={styles.primaryButton} onClick={handleUploadButtonClick}>
            Upload Trace
          </button>
          <span className={styles.inlineHint}>Pick the format first, then choose the trace file from the system upload dialog.</span>
        </div>
        {message ? <p className={`${styles.message} ${styles.messageSuccess}`}>{message}</p> : null}
        {error ? <p className={`${styles.message} ${styles.messageError}`}>{error}</p> : null}
      </section>

      <section className={styles.sectionCard}>
        <div className={styles.sectionHeader}>
          <div>
            <span className={styles.sectionKicker}>Library</span>
            <h2 className={styles.sectionTitle}>Built-in workloads</h2>
            <p className={styles.sectionDescription}>Reusable traffic patterns for quick sizing passes before you pull in a custom trace.</p>
          </div>
        </div>
        <div className={styles.libraryGrid}>
          {workloads.map((workload) => (
            <article key={workload.name} className={styles.libraryCard}>
              <div className={styles.libraryCardHeader}>
                <div>
                  <span className={styles.libraryCardKey}>Built-in profile</span>
                  <h3 className={styles.libraryCardTitle}>{formatBuiltinWorkloadName(workload.name)}</h3>
                </div>
                <span className={styles.metricPill}>
                  {workload.stats ? `P99 ${formatNumber(workload.stats.p99_total_tokens)} tokens` : 'Loading profile'}
                </span>
              </div>
              <p className={styles.libraryCardText}>{describeBuiltinWorkload(workload.name, workload.description)}</p>
              <div className={styles.metricPillRow}>
                <span className={styles.metricPill}>{workload.stats ? `${formatNumber(workload.stats.n_requests)} samples` : 'Sampling'}</span>
                <span className={styles.metricPill}>{workload.stats ? `${formatNumber(workload.stats.arrival_rate_rps, 1)} rps` : 'Profiling'}</span>
                <span className={styles.metricPill}>{workload.stats ? `P95 ${formatNumber(workload.stats.p95_prompt_tokens)} prompt` : 'Waiting'}</span>
              </div>
              <p className={styles.cardFootnote}>{workload.path.split('/').slice(-1)[0]}</p>
            </article>
          ))}
        </div>
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
        <p className={styles.inlineHint} style={{ marginTop: '0.9rem' }}>
          Open “View” to inspect a replay sample in a focused preview dialog.
        </p>
      </section>
      <FleetSimTracePreviewDialog
        trace={selectedTrace}
        sample={selectedSample}
        onClose={handleClosePreview}
      />
    </FleetSimSurfaceLayout>
  )
}
