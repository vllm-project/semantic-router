import { useEffect, useState } from 'react'
import FleetSimSurfaceLayout from './FleetSimSurfaceLayout'
import styles from './FleetSimPage.module.css'
import { FLEET_SIM_API_PREFIX, listFleets, listJobs, listTraces, listWorkloads, type FleetConfig, type FleetSimJob, type TraceInfo, type BuiltinWorkload } from '../utils/fleetSimApi'
import {
  extractJobWorkload,
  formatDateTime,
  formatMoneyKusd,
  formatNumber,
  JobStatusBadge,
  renderJobResultSummary,
} from './fleetSimPageSupport'

export default function FleetSimOverviewPage() {
  const [workloads, setWorkloads] = useState<BuiltinWorkload[]>([])
  const [fleets, setFleets] = useState<FleetConfig[]>([])
  const [traces, setTraces] = useState<TraceInfo[]>([])
  const [jobs, setJobs] = useState<FleetSimJob[]>([])
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const [workloadsData, fleetsData, tracesData, jobsData] = await Promise.all([
          listWorkloads(),
          listFleets(),
          listTraces(),
          listJobs(),
        ])
        if (cancelled) return
        setWorkloads(workloadsData)
        setFleets(fleetsData)
        setTraces(tracesData)
        setJobs(jobsData)
        setError('')
      } catch (loadError) {
        if (!cancelled) setError(loadError instanceof Error ? loadError.message : 'Failed to load simulator overview')
      }
    }

    void load()
    const intervalID = window.setInterval(() => {
      void load()
    }, 10000)

    return () => {
      cancelled = true
      window.clearInterval(intervalID)
    }
  }, [])

  const finishedJobs = jobs.filter((job) => job.status === 'done')
  const latestJob = [...jobs].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))[0]
  const annualSpend = fleets.reduce((sum, fleet) => sum + fleet.estimated_annual_cost_kusd, 0)

  return (
    <FleetSimSurfaceLayout
      title="Overview"
      description="Keep the simulator sidecar in the same operator loop as the router. Review workload libraries, fleet catalog state, and the latest capacity-planning runs without leaving the dashboard."
      currentPath="/fleet-sim"
      meta={[
        { label: 'Built-in workloads', value: formatNumber(workloads.length) },
        { label: 'Saved fleets', value: formatNumber(fleets.length) },
        { label: 'Tracked traces', value: formatNumber(traces.length) },
      ]}
      panelFooter={
        <div className={styles.buttonRow}>
          <a className={styles.secondaryButton} href={`${FLEET_SIM_API_PREFIX}/docs`} target="_blank" rel="noreferrer">
            API Docs
          </a>
          <a className={styles.ghostButton} href={`${FLEET_SIM_API_PREFIX}/openapi.json`} target="_blank" rel="noreferrer">
            OpenAPI
          </a>
        </div>
      }
    >
      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Sidecar Endpoint</span>
          <strong className={`${styles.statValue} ${styles.mono}`}>/api/fleet-sim</strong>
          <span className={styles.statMeta}>Dashboard backend proxies requests to the simulator container on the shared runtime network.</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Fleet Annual Spend</span>
          <strong className={styles.statValue}>{formatMoneyKusd(annualSpend)}</strong>
          <span className={styles.statMeta}>Combined estimate across every saved fleet profile.</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Run Queue</span>
          <strong className={styles.statValue}>{formatNumber(jobs.length)}</strong>
          <span className={styles.statMeta}>{finishedJobs.length} completed jobs available for comparison.</span>
        </div>
      </div>

      {error ? <p className={`${styles.message} ${styles.messageError}`}>{error}</p> : null}

      <div className={styles.splitGrid}>
        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Latest Run</h2>
              <p className={styles.sectionDescription}>The freshest optimization or simulation result that landed in the shared simulator state.</p>
            </div>
          </div>
          {latestJob ? (
            <>
              <div className={styles.compactListItem}>
                <div className={styles.compactListMeta}>
                  <span className={styles.compactListTitle}>{latestJob.type.toUpperCase()} · {extractJobWorkload(latestJob)?.name || extractJobWorkload(latestJob)?.trace_id || 'mixed'}</span>
                  <span className={styles.compactListText}>Created {formatDateTime(latestJob.created_at)}</span>
                </div>
                <JobStatusBadge status={latestJob.status} />
              </div>
              <div className={styles.resultCard} style={{ marginTop: '0.9rem' }}>
                {renderJobResultSummary(latestJob)}
              </div>
            </>
          ) : (
            <div className={styles.emptyState}>No simulator jobs have been submitted yet.</div>
          )}
        </section>

        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Recent Assets</h2>
              <p className={styles.sectionDescription}>Trace uploads and saved fleets persisted under the shared `vllm-sr` workspace.</p>
            </div>
          </div>
          <ul className={styles.compactList}>
            {fleets.slice(0, 3).map((fleet) => (
              <li key={fleet.id} className={styles.compactListItem}>
                <div className={styles.compactListMeta}>
                  <span className={styles.compactListTitle}>{fleet.name}</span>
                  <span className={styles.compactListText}>{fleet.total_gpus} GPUs · {fleet.router} router</span>
                </div>
                <span className={styles.compactListText}>{formatMoneyKusd(fleet.estimated_annual_cost_kusd)}</span>
              </li>
            ))}
            {traces.slice(0, 3).map((trace) => (
              <li key={trace.id} className={styles.compactListItem}>
                <div className={styles.compactListMeta}>
                  <span className={styles.compactListTitle}>{trace.name}</span>
                  <span className={styles.compactListText}>{trace.format} · {formatNumber(trace.n_requests)} requests</span>
                </div>
                <span className={styles.compactListText}>{formatDateTime(trace.upload_time)}</span>
              </li>
            ))}
            {fleets.length === 0 && traces.length === 0 ? (
              <li className={styles.emptyState}>The shared simulator state is empty. Upload traces or save a fleet to begin.</li>
            ) : null}
          </ul>
        </section>
      </div>
    </FleetSimSurfaceLayout>
  )
}
