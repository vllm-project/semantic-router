import { useEffect, useState } from 'react'
import TableHeader from '../components/TableHeader'
import { DataTable, type Column } from '../components/DataTable'
import FleetSimSurfaceLayout from './FleetSimSurfaceLayout'
import styles from './FleetSimPage.module.css'
import {
  createJob,
  deleteJob,
  listFleets,
  listJobs,
  listTraces,
  listWorkloads,
  type FleetConfig,
  type FleetSimJob,
  type FleetSimJobType,
  type TraceInfo,
  type WorkloadRef,
  type BuiltinWorkload,
} from '../utils/fleetSimApi'
import {
  extractJobFleetID,
  extractJobWorkload,
  formatDateTime,
  formatNumber,
  JobStatusBadge,
  renderJobResultSummary,
} from './fleetSimPageSupport'

function buildWorkloadRef(workloadMode: 'builtin' | 'trace', builtinName: string, traceID: string): WorkloadRef {
  if (workloadMode === 'trace') {
    return { type: 'trace', trace_id: traceID }
  }
  return { type: 'builtin', name: builtinName }
}

export default function FleetSimRunsPage() {
  const [jobType, setJobType] = useState<FleetSimJobType>('optimize')
  const [workloads, setWorkloads] = useState<BuiltinWorkload[]>([])
  const [traces, setTraces] = useState<TraceInfo[]>([])
  const [fleets, setFleets] = useState<FleetConfig[]>([])
  const [jobs, setJobs] = useState<FleetSimJob[]>([])
  const [expandedJobIDs, setExpandedJobIDs] = useState<Set<string>>(new Set())
  const [workloadMode, setWorkloadMode] = useState<'builtin' | 'trace'>('builtin')
  const [builtinName, setBuiltinName] = useState('azure')
  const [traceID, setTraceID] = useState('')
  const [fleetID, setFleetID] = useState('')
  const [lam, setLam] = useState('200')
  const [sloMs, setSloMs] = useState('500')
  const [nRequests, setNRequests] = useState('20000')
  const [lamRange, setLamRange] = useState('100, 200, 300, 500')
  const [bShort, setBShort] = useState('4096')
  const [gpuShort, setGpuShort] = useState('a100')
  const [gpuLong, setGpuLong] = useState('h100')
  const [longMaxCtx, setLongMaxCtx] = useState('65536')
  const [gammaMin, setGammaMin] = useState('1.0')
  const [gammaMax, setGammaMax] = useState('2.0')
  const [gammaStep, setGammaStep] = useState('0.1')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const load = async () => {
    const [workloadsData, tracesData, fleetsData, jobsData] = await Promise.all([
      listWorkloads(),
      listTraces(),
      listFleets(),
      listJobs(),
    ])
    setWorkloads(workloadsData)
    setTraces(tracesData)
    setFleets(fleetsData)
    setJobs(jobsData.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)))
    if (!fleetID && fleetsData[0]) {
      setFleetID(fleetsData[0].id)
    }
    if (!traceID && tracesData[0]) {
      setTraceID(tracesData[0].id)
    }
  }

  useEffect(() => {
    let cancelled = false

    const loadWithErrors = async () => {
      try {
        await load()
        if (!cancelled) setError('')
      } catch (loadError) {
        if (!cancelled) setError(loadError instanceof Error ? loadError.message : 'Failed to load simulator runs')
      }
    }

    void loadWithErrors()
    const intervalID = window.setInterval(() => {
      void loadWithErrors()
    }, 5000)

    return () => {
      cancelled = true
      window.clearInterval(intervalID)
    }
  }, [fleetID, traceID])

  const handleSubmit = async () => {
    try {
      const workload = buildWorkloadRef(workloadMode, builtinName, traceID)
      let payload: Record<string, unknown>

      if (jobType === 'optimize') {
        payload = {
          type: 'optimize',
          optimize: {
            workload,
            lam: Number(lam),
            slo_ms: Number(sloMs),
            b_short: Number(bShort),
            gpu_short: gpuShort,
            gpu_long: gpuLong,
            long_max_ctx: Number(longMaxCtx),
            gamma_min: Number(gammaMin),
            gamma_max: Number(gammaMax),
            gamma_step: Number(gammaStep),
            n_sim_requests: Number(nRequests),
          },
        }
      } else if (jobType === 'simulate') {
        if (!fleetID) {
          throw new Error('Save or select a fleet before running a simulation.')
        }
        payload = {
          type: 'simulate',
          simulate: {
            workload,
            fleet_id: fleetID,
            lam: Number(lam),
            slo_ms: Number(sloMs),
            n_requests: Number(nRequests),
          },
        }
      } else {
        if (!fleetID) {
          throw new Error('Save or select a fleet before running a what-if sweep.')
        }
        payload = {
          type: 'whatif',
          whatif: {
            workload,
            fleet_id: fleetID,
            lam_range: lamRange.split(',').map((value) => Number(value.trim())).filter((value) => !Number.isNaN(value)),
            slo_ms: Number(sloMs),
            n_requests: Number(nRequests),
          },
        }
      }

      const created = await createJob(payload)
      setMessage(`Submitted ${created.type} job ${created.id}`)
      setError('')
      await load()
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Failed to submit job')
    }
  }

  const handleDeleteJob = async (job: FleetSimJob) => {
    try {
      await deleteJob(job.id)
      setMessage(`Deleted job ${job.id}`)
      setError('')
      await load()
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Failed to delete job')
    }
  }

  const jobColumns: Column<FleetSimJob>[] = [
    {
      key: 'type',
      header: 'Run',
      sortable: true,
      render: (row) => (
        <div className={styles.compactListMeta}>
          <span className={styles.compactListTitle}>{row.type.toUpperCase()}</span>
          <span className={styles.compactListText}>{formatDateTime(row.created_at)}</span>
        </div>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      sortable: true,
      render: (row) => <JobStatusBadge status={row.status} />,
    },
    {
      key: 'workload',
      header: 'Workload',
      render: (row) => {
        const workload = extractJobWorkload(row)
        return workload?.type === 'trace' ? workload.trace_id || 'Trace' : workload?.name || 'Built-in'
      },
    },
    {
      key: 'fleet',
      header: 'Fleet',
      render: (row) => extractJobFleetID(row),
    },
  ]

  return (
    <FleetSimSurfaceLayout
      title="Runs"
      description="Submit optimize, simulate, and what-if jobs through the dashboard while the backend proxies every request into the `vllm-sr-sim` sidecar. Results persist with the workspace and refresh in place."
      currentPath="/fleet-sim/runs"
      meta={[
        { label: 'Saved fleets', value: formatNumber(fleets.length) },
        { label: 'Queued jobs', value: formatNumber(jobs.length) },
        { label: 'Trace options', value: formatNumber(traces.length) },
      ]}
    >
      <section className={styles.sectionCard}>
        <div className={styles.sectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Submit Run</h2>
            <p className={styles.sectionDescription}>Optimize searches for the cheapest viable split fleet. Simulate and what-if reuse a saved fleet so repeated analyses stay comparable.</p>
          </div>
        </div>
        <div className={styles.formGrid}>
          <label className={styles.field}>
            <span className={styles.fieldLabel}>Run Type</span>
            <select className={styles.select} value={jobType} onChange={(event) => setJobType(event.target.value as FleetSimJobType)}>
              <option value="optimize">optimize</option>
              <option value="simulate">simulate</option>
              <option value="whatif">whatif</option>
            </select>
          </label>
          <label className={styles.field}>
            <span className={styles.fieldLabel}>Workload Source</span>
            <select className={styles.select} value={workloadMode} onChange={(event) => setWorkloadMode(event.target.value as 'builtin' | 'trace')}>
              <option value="builtin">Built-in workload</option>
              <option value="trace">Uploaded trace</option>
            </select>
          </label>
          {workloadMode === 'builtin' ? (
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Built-in</span>
              <select className={styles.select} value={builtinName} onChange={(event) => setBuiltinName(event.target.value)}>
                {workloads.map((workload) => (
                  <option key={workload.name} value={workload.name}>
                    {workload.name}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Trace</span>
              <select className={styles.select} value={traceID} onChange={(event) => setTraceID(event.target.value)}>
                <option value="">Select trace</option>
                {traces.map((trace) => (
                  <option key={trace.id} value={trace.id}>
                    {trace.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className={styles.field}>
            <span className={styles.fieldLabel}>Lambda</span>
            <input className={styles.input} value={lam} onChange={(event) => setLam(event.target.value)} />
          </label>
          <label className={styles.field}>
            <span className={styles.fieldLabel}>SLO (ms)</span>
            <input className={styles.input} value={sloMs} onChange={(event) => setSloMs(event.target.value)} />
          </label>
          <label className={styles.field}>
            <span className={styles.fieldLabel}>Requests</span>
            <input className={styles.input} value={nRequests} onChange={(event) => setNRequests(event.target.value)} />
          </label>
          {jobType !== 'optimize' ? (
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Fleet</span>
              <select className={styles.select} value={fleetID} onChange={(event) => setFleetID(event.target.value)}>
                <option value="">Select fleet</option>
                {fleets.map((fleet) => (
                  <option key={fleet.id} value={fleet.id}>
                    {fleet.name}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {jobType === 'whatif' ? (
            <label className={`${styles.field} ${styles.fieldWide}`}>
              <span className={styles.fieldLabel}>Lambda Range</span>
              <input className={styles.input} value={lamRange} onChange={(event) => setLamRange(event.target.value)} />
            </label>
          ) : null}
          {jobType === 'optimize' ? (
            <>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>B Short</span>
                <input className={styles.input} value={bShort} onChange={(event) => setBShort(event.target.value)} />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Short GPU</span>
                <input className={styles.input} value={gpuShort} onChange={(event) => setGpuShort(event.target.value)} />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Long GPU</span>
                <input className={styles.input} value={gpuLong} onChange={(event) => setGpuLong(event.target.value)} />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Long Max Context</span>
                <input className={styles.input} value={longMaxCtx} onChange={(event) => setLongMaxCtx(event.target.value)} />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Gamma Min</span>
                <input className={styles.input} value={gammaMin} onChange={(event) => setGammaMin(event.target.value)} />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Gamma Max</span>
                <input className={styles.input} value={gammaMax} onChange={(event) => setGammaMax(event.target.value)} />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Gamma Step</span>
                <input className={styles.input} value={gammaStep} onChange={(event) => setGammaStep(event.target.value)} />
              </label>
            </>
          ) : null}
        </div>
        <div className={styles.buttonRow} style={{ marginTop: '1rem' }}>
          <button type="button" className={styles.primaryButton} onClick={() => void handleSubmit()}>
            Submit Run
          </button>
          <span className={styles.inlineHint}>
            {jobType === 'optimize'
              ? 'Optimize submits a fresh search over gamma and pool split.'
              : 'Simulate and what-if reuse a saved fleet definition.'}
          </span>
        </div>
        {message ? <p className={`${styles.message} ${styles.messageSuccess}`}>{message}</p> : null}
        {error ? <p className={`${styles.message} ${styles.messageError}`}>{error}</p> : null}
      </section>

      <section className={styles.sectionCard}>
        <TableHeader title="Run History" count={jobs.length} variant="embedded" />
        <DataTable
          columns={jobColumns}
          data={jobs}
          keyExtractor={(row) => row.id}
          onDelete={(row) => void handleDeleteJob(row)}
          expandable
          isRowExpanded={(row) => expandedJobIDs.has(row.id)}
          onToggleExpand={(row) => {
            setExpandedJobIDs((current) => {
              const next = new Set(current)
              if (next.has(row.id)) next.delete(row.id)
              else next.add(row.id)
              return next
            })
          }}
          renderExpandedRow={(row) => renderJobResultSummary(row)}
          emptyMessage="No simulator jobs yet."
        />
      </section>
    </FleetSimSurfaceLayout>
  )
}
