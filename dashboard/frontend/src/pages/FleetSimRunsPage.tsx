import { useEffect, useState, type Dispatch, type SetStateAction } from 'react'
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
  describeJobType,
  formatBuiltinWorkloadName,
  formatDateTime,
  formatGpuLabel,
  formatJobType,
  formatNumber,
  JobStatusBadge,
  renderJobResultRows,
} from './fleetSimPageSupport'

const GPU_OPTIONS = ['a100', 'h100', 'a10g']
const RUN_TYPE_OPTIONS = [
  { value: 'optimize', title: 'Optimize' },
  { value: 'simulate', title: 'Simulate' },
  { value: 'whatif', title: 'What-if' },
] as const
const WORKLOAD_SOURCE_OPTIONS = [
  { value: 'builtin', title: 'Built-in library', description: 'Use a reusable planning profile.' },
  { value: 'trace', title: 'Uploaded trace', description: 'Replay a saved traffic slice from this workspace.' },
] as const

function buildWorkloadRef(workloadMode: 'builtin' | 'trace', builtinName: string, traceID: string): WorkloadRef {
  if (workloadMode === 'trace') {
    return { type: 'trace', trace_id: traceID }
  }
  return { type: 'builtin', name: builtinName }
}

function resolveWorkloadLabel(
  workload: WorkloadRef | null | undefined,
  workloads: BuiltinWorkload[],
  traces: TraceInfo[]
): string {
  if (!workload) return 'Traffic input pending'
  if (workload.type === 'trace') {
    return traces.find((trace) => trace.id === workload.trace_id)?.name || 'Uploaded trace'
  }
  const builtinName = workload.name || workloads[0]?.name || 'builtin'
  return formatBuiltinWorkloadName(builtinName)
}

function resolveFleetLabel(fleetID: string, fleets: FleetConfig[]): string {
  if (fleetID === 'Search mode') return fleetID
  return fleets.find((fleet) => fleet.id === fleetID)?.name || fleetID
}

async function fetchRunsPageData() {
  const [workloadsData, tracesData, fleetsData, jobsData] = await Promise.all([
    listWorkloads(),
    listTraces(),
    listFleets(),
    listJobs(),
  ])
  return {
    workloadsData,
    tracesData,
    fleetsData,
    jobsData: jobsData.sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)),
  }
}

function applyRunsPageData(
  data: Awaited<ReturnType<typeof fetchRunsPageData>>,
  state: {
    fleetID: string
    traceID: string
    setWorkloads: Dispatch<SetStateAction<BuiltinWorkload[]>>
    setTraces: Dispatch<SetStateAction<TraceInfo[]>>
    setFleets: Dispatch<SetStateAction<FleetConfig[]>>
    setJobs: Dispatch<SetStateAction<FleetSimJob[]>>
    setFleetID: Dispatch<SetStateAction<string>>
    setTraceID: Dispatch<SetStateAction<string>>
  }
) {
  const { workloadsData, tracesData, fleetsData, jobsData } = data
  const {
    fleetID,
    traceID,
    setWorkloads,
    setTraces,
    setFleets,
    setJobs,
    setFleetID,
    setTraceID,
  } = state

  setWorkloads(workloadsData)
  setTraces(tracesData)
  setFleets(fleetsData)
  setJobs(jobsData)
  if (!fleetID && fleetsData[0]) {
    setFleetID(fleetsData[0].id)
  }
  if (!traceID && tracesData[0]) {
    setTraceID(tracesData[0].id)
  }
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

  useEffect(() => {
    let cancelled = false

    const loadWithErrors = async () => {
      try {
        const data = await fetchRunsPageData()
        if (cancelled) return
        applyRunsPageData(data, {
          fleetID,
          traceID,
          setWorkloads,
          setTraces,
          setFleets,
          setJobs,
          setFleetID,
          setTraceID,
        })
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

  const lambdaValues = lamRange
    .split(',')
    .map((value) => Number(value.trim()))
    .filter((value) => !Number.isNaN(value))
  const selectedTrace = traces.find((trace) => trace.id === traceID)
  const selectedFleet = fleets.find((fleet) => fleet.id === fleetID)
  const selectedWorkloadLabel = resolveWorkloadLabel(
    workloadMode === 'trace'
      ? { type: 'trace', trace_id: traceID }
      : { type: 'builtin', name: builtinName },
    workloads,
    traces
  )

  const handleSubmit = async () => {
    try {
      if (workloadMode === 'trace' && !traceID) {
        throw new Error('Select an uploaded trace before submitting this run.')
      }
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
        if (lambdaValues.length === 0) {
          throw new Error('Add at least one arrival-rate checkpoint for the what-if sweep.')
        }
        payload = {
          type: 'whatif',
          whatif: {
            workload,
            fleet_id: fleetID,
            lam_range: lambdaValues,
            slo_ms: Number(sloMs),
            n_requests: Number(nRequests),
          },
        }
      }

      const created = await createJob(payload)
      setMessage(`Submitted ${created.type} job ${created.id}`)
      setError('')
      applyRunsPageData(await fetchRunsPageData(), {
        fleetID,
        traceID,
        setWorkloads,
        setTraces,
        setFleets,
        setJobs,
        setFleetID,
        setTraceID,
      })
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : 'Failed to submit job')
    }
  }

  const handleDeleteJob = async (job: FleetSimJob) => {
    try {
      await deleteJob(job.id)
      setMessage(`Deleted job ${job.id}`)
      setError('')
      applyRunsPageData(await fetchRunsPageData(), {
        fleetID,
        traceID,
        setWorkloads,
        setTraces,
        setFleets,
        setJobs,
        setFleetID,
        setTraceID,
      })
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
          <span className={styles.compactListTitle}>{formatJobType(row.type)}</span>
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
        return resolveWorkloadLabel(workload, workloads, traces)
      },
    },
    {
      key: 'fleet',
      header: 'Fleet',
      render: (row) => resolveFleetLabel(extractJobFleetID(row), fleets),
    },
  ]

  return (
    <FleetSimSurfaceLayout
      title="Runs"
      description="Launch planning scenarios, compare saved fleet behavior, and keep a readable history of the decisions behind each run."
      currentPath="/fleet-sim/runs"
      meta={[
        { label: 'Saved fleets', value: formatNumber(fleets.length) },
        { label: 'Run history', value: formatNumber(jobs.length) },
        { label: 'Trace options', value: formatNumber(traces.length) },
      ]}
    >
      <div className={styles.composerSplit}>
        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <span className={styles.sectionKicker}>Scenario builder</span>
              <h2 className={styles.sectionTitle}>Submit run</h2>
              <p className={styles.sectionDescription}>Build a scenario in dashboard terms first, then let the simulator fill in the numbers behind it.</p>
            </div>
          </div>
          <div className={styles.formStack}>
            <div className={styles.fieldSection}>
              <div>
                <span className={styles.sectionKicker}>Run type</span>
                <p className={styles.sectionDescription}>Choose whether you want a fresh recommendation, a replay, or a saturation check.</p>
              </div>
              <div className={styles.choiceGrid}>
                {RUN_TYPE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={`${styles.choiceButton} ${jobType === option.value ? styles.choiceButtonActive : ''}`}
                    onClick={() => setJobType(option.value)}
                  >
                    <span className={styles.choiceButtonTitle}>{option.title}</span>
                    <span className={styles.choiceButtonDescription}>{describeJobType(option.value)}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className={styles.fieldSection}>
              <div>
                <span className={styles.sectionKicker}>Traffic input</span>
                <p className={styles.sectionDescription}>Pick the source you want the run to reflect, then set the core demand target.</p>
              </div>
              <div className={styles.choiceGrid}>
                {WORKLOAD_SOURCE_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={`${styles.choiceButton} ${workloadMode === option.value ? styles.choiceButtonActive : ''}`}
                    onClick={() => setWorkloadMode(option.value)}
                  >
                    <span className={styles.choiceButtonTitle}>{option.title}</span>
                    <span className={styles.choiceButtonDescription}>{option.description}</span>
                  </button>
                ))}
              </div>
              <div className={styles.formGrid}>
                {workloadMode === 'builtin' ? (
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Built-in profile</span>
                    <select className={styles.select} value={builtinName} onChange={(event) => setBuiltinName(event.target.value)}>
                      {workloads.map((workload) => (
                        <option key={workload.name} value={workload.name}>
                          {formatBuiltinWorkloadName(workload.name)}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : (
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Uploaded trace</span>
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
                  <span className={styles.fieldLabel}>Arrival rate</span>
                  <input className={styles.input} value={lam} onChange={(event) => setLam(event.target.value)} />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>SLO target (ms)</span>
                  <input className={styles.input} value={sloMs} onChange={(event) => setSloMs(event.target.value)} />
                </label>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Replay requests</span>
                  <input className={styles.input} value={nRequests} onChange={(event) => setNRequests(event.target.value)} />
                </label>
              </div>
            </div>

            {jobType !== 'optimize' ? (
              <div className={styles.fieldSection}>
                <div>
                  <span className={styles.sectionKicker}>Fleet target</span>
                  <p className={styles.sectionDescription}>Simulation and what-if runs stay comparable by reusing a saved fleet definition.</p>
                </div>
                <div className={styles.formGrid}>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Saved fleet</span>
                    <select className={styles.select} value={fleetID} onChange={(event) => setFleetID(event.target.value)}>
                      <option value="">Select fleet</option>
                      {fleets.map((fleet) => (
                        <option key={fleet.id} value={fleet.id}>
                          {fleet.name}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>
            ) : null}

            {jobType === 'whatif' ? (
              <div className={styles.fieldSection}>
                <div>
                  <span className={styles.sectionKicker}>Sweep range</span>
                  <p className={styles.sectionDescription}>Provide the arrival-rate checkpoints you want the dashboard to test, separated by commas.</p>
                </div>
                <div className={styles.formGrid}>
                  <label className={`${styles.field} ${styles.fieldWide}`}>
                    <span className={styles.fieldLabel}>Arrival-rate checkpoints</span>
                    <input className={styles.input} value={lamRange} onChange={(event) => setLamRange(event.target.value)} />
                  </label>
                </div>
              </div>
            ) : null}

            {jobType === 'optimize' ? (
              <div className={styles.fieldSection}>
                <div>
                  <span className={styles.sectionKicker}>Search space</span>
                  <p className={styles.sectionDescription}>Define the short and long pool search bounds instead of typing raw solver parameters into one flat form.</p>
                </div>
                <div className={styles.formGrid}>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Short-context boundary</span>
                    <input className={styles.input} value={bShort} onChange={(event) => setBShort(event.target.value)} />
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Short pool GPU</span>
                    <select className={styles.select} value={gpuShort} onChange={(event) => setGpuShort(event.target.value)}>
                      {GPU_OPTIONS.map((gpu) => (
                        <option key={gpu} value={gpu}>
                          {formatGpuLabel(gpu)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Long pool GPU</span>
                    <select className={styles.select} value={gpuLong} onChange={(event) => setGpuLong(event.target.value)}>
                      {GPU_OPTIONS.map((gpu) => (
                        <option key={gpu} value={gpu}>
                          {formatGpuLabel(gpu)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Long max context</span>
                    <input className={styles.input} value={longMaxCtx} onChange={(event) => setLongMaxCtx(event.target.value)} />
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Gamma min</span>
                    <input className={styles.input} value={gammaMin} onChange={(event) => setGammaMin(event.target.value)} />
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Gamma max</span>
                    <input className={styles.input} value={gammaMax} onChange={(event) => setGammaMax(event.target.value)} />
                  </label>
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Gamma step</span>
                    <input className={styles.input} value={gammaStep} onChange={(event) => setGammaStep(event.target.value)} />
                  </label>
                </div>
              </div>
            ) : null}
          </div>
          <div className={styles.buttonRow} style={{ marginTop: '1rem' }}>
            <button type="button" className={styles.primaryButton} onClick={() => void handleSubmit()}>
              Submit Run
            </button>
            <span className={styles.inlineHint}>{describeJobType(jobType)}</span>
          </div>
          {message ? <p className={`${styles.message} ${styles.messageSuccess}`}>{message}</p> : null}
          {error ? <p className={`${styles.message} ${styles.messageError}`}>{error}</p> : null}
        </section>

        <aside className={styles.summaryCard}>
          <div className={styles.summaryHeader}>
            <span className={styles.summaryEyebrow}>Run summary</span>
            <h2 className={styles.summaryTitle}>{formatJobType(jobType)} scenario</h2>
            <p className={styles.summaryText}>{describeJobType(jobType)}</p>
          </div>
          <div className={styles.summaryPillRow}>
            <span className={styles.summaryPill}>{selectedWorkloadLabel}</span>
            {jobType === 'optimize'
              ? <span className={styles.summaryPill}>Fresh search</span>
              : <span className={styles.summaryPill}>{selectedFleet?.name || 'Fleet required'}</span>}
          </div>
          <dl className={styles.summaryList}>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>Arrival rate</dt>
              <dd className={styles.summaryValue}>{lam} rps</dd>
            </div>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>SLO target</dt>
              <dd className={styles.summaryValue}>{sloMs} ms</dd>
            </div>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>Replay size</dt>
              <dd className={styles.summaryValue}>{formatNumber(Number(nRequests) || 0)}</dd>
            </div>
            <div className={styles.summaryItem}>
              <dt className={styles.summaryLabel}>Fleet</dt>
              <dd className={styles.summaryValue}>{jobType === 'optimize' ? 'Optimizer search' : selectedFleet?.name || 'Select a fleet'}</dd>
            </div>
            {jobType === 'optimize' ? (
              <div className={styles.summaryItem}>
                <dt className={styles.summaryLabel}>Pool mix</dt>
                <dd className={styles.summaryValue}>{`${formatGpuLabel(gpuShort)} -> ${formatGpuLabel(gpuLong)}`}</dd>
              </div>
            ) : null}
            {jobType === 'whatif' ? (
              <div className={styles.summaryItem}>
                <dt className={styles.summaryLabel}>Sweep points</dt>
                <dd className={styles.summaryValue}>{formatNumber(lambdaValues.length)}</dd>
              </div>
            ) : null}
            {workloadMode === 'trace' && selectedTrace ? (
              <div className={styles.summaryItem}>
                <dt className={styles.summaryLabel}>Trace size</dt>
                <dd className={styles.summaryValue}>{formatNumber(selectedTrace.n_requests)}</dd>
              </div>
            ) : null}
          </dl>
        </aside>
      </div>

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
          renderExpandedRow={(row) => {
            const workload = extractJobWorkload(row)
            const fleetLabel = resolveFleetLabel(extractJobFleetID(row), fleets)
            return (
              <div className={styles.expandedPanel}>
                <div className={styles.inlineDetailGrid}>
                  <div className={styles.inlineDetailCell}>
                    <span className={styles.inlineDetailLabel}>Scenario</span>
                    <span className={styles.inlineDetailValue}>{formatJobType(row.type)}</span>
                  </div>
                  <div className={styles.inlineDetailCell}>
                    <span className={styles.inlineDetailLabel}>Workload</span>
                    <span className={styles.inlineDetailValue}>{resolveWorkloadLabel(workload, workloads, traces)}</span>
                  </div>
                  <div className={styles.inlineDetailCell}>
                    <span className={styles.inlineDetailLabel}>Fleet</span>
                    <span className={styles.inlineDetailValue}>{fleetLabel}</span>
                  </div>
                  <div className={styles.inlineDetailCell}>
                    <span className={styles.inlineDetailLabel}>Created</span>
                    <span className={styles.inlineDetailValue}>{formatDateTime(row.created_at)}</span>
                  </div>
                </div>
                {renderJobResultRows(row)}
              </div>
            )
          }}
          emptyMessage="No simulator jobs yet."
        />
      </section>
    </FleetSimSurfaceLayout>
  )
}
