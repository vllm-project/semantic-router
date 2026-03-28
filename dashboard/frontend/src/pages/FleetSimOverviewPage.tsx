import { useEffect, useState } from 'react'
import FleetSimSurfaceLayout from './FleetSimSurfaceLayout'
import styles from './FleetSimPage.module.css'
import { FLEET_SIM_API_PREFIX, listFleets, listJobs, listTraces, listWorkloads, type FleetConfig, type FleetSimJob, type TraceInfo, type BuiltinWorkload } from '../utils/fleetSimApi'
import {
  extractJobFleetID,
  describeBuiltinWorkload,
  extractJobWorkload,
  formatBuiltinWorkloadName,
  formatDateTime,
  formatJobStatus,
  formatJobType,
  formatMoneyKusd,
  formatNumber,
  formatRouterType,
  formatTraceFormat,
  JobStatusBadge,
  renderJobResultSummary,
} from './fleetSimPageSupport'

type PlanningAssetItem =
  | {
      key: string
      kind: 'workload'
      title: string
      detail: string
      meta: string
    }
  | {
      key: string
      kind: 'fleet'
      title: string
      detail: string
      meta: string
    }
  | {
      key: string
      kind: 'trace'
      title: string
      detail: string
      meta: string
    }

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
  const latestWorkload = latestJob ? extractJobWorkload(latestJob) : null
  const latestTrace = latestWorkload?.type === 'trace'
    ? traces.find((trace) => trace.id === latestWorkload.trace_id)
    : null
  const latestFleet = latestJob
    ? fleets.find((fleet) => fleet.id === extractJobFleetID(latestJob))
    : null
  const planningAssets = workloads.length + traces.length
  const workloadAssets: PlanningAssetItem[] = workloads.slice(0, 2).map((workload) => ({
    key: `workload-${workload.name}`,
    kind: 'workload',
    title: formatBuiltinWorkloadName(workload.name),
    detail: describeBuiltinWorkload(workload.name, workload.description),
    meta: 'Library',
  }))
  const traceAssets: PlanningAssetItem[] = traces.slice(0, 2).map((trace) => ({
    key: `trace-${trace.id}`,
    kind: 'trace',
    title: trace.name,
    detail: `${formatTraceFormat(trace.format)} · ${formatNumber(trace.n_requests)} requests`,
    meta: formatDateTime(trace.upload_time),
  }))
  const fleetAssets: PlanningAssetItem[] = fleets.slice(0, 2).map((fleet) => ({
    key: `fleet-${fleet.id}`,
    kind: 'fleet',
    title: fleet.name,
    detail: `${formatNumber(fleet.total_gpus)} GPUs · ${formatRouterType(fleet.router)}`,
    meta: formatMoneyKusd(fleet.estimated_annual_cost_kusd),
  }))
  const planningAssetItems: PlanningAssetItem[] = []
  const assetColumns = [workloadAssets, traceAssets, fleetAssets]
  for (let index = 0; index < 2 && planningAssetItems.length < 3; index += 1) {
    assetColumns.forEach((assets) => {
      const asset = assets[index]
      if (asset && planningAssetItems.length < 3) {
        planningAssetItems.push(asset)
      }
    })
  }

  return (
    <FleetSimSurfaceLayout
      title="Overview"
      description="Keep workload libraries, reusable fleets, and recent planning outcomes in one place so operators can decide what to size, replay, or compare next."
      currentPath="/fleet-sim"
      meta={[
        { label: 'Workload library', value: formatNumber(workloads.length) },
        { label: 'Saved fleets', value: formatNumber(fleets.length) },
        { label: 'Planning runs', value: formatNumber(jobs.length) },
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
          <span className={styles.statLabel}>Planning library</span>
          <strong className={styles.statValue}>{formatNumber(planningAssets)}</strong>
          <span className={styles.statMeta}>
            {formatNumber(workloads.length)} built-in profiles and {formatNumber(traces.length)} uploaded traces ready for comparison.
          </span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Annual spend in review</span>
          <strong className={styles.statValue}>{formatMoneyKusd(annualSpend)}</strong>
          <span className={styles.statMeta}>Combined estimate across {formatNumber(fleets.length)} saved fleet plans.</span>
        </div>
        <div className={styles.statCard}>
          <span className={styles.statLabel}>Latest activity</span>
          <strong className={styles.statValue}>{latestJob ? formatJobType(latestJob.type) : 'Idle'}</strong>
          <span className={styles.statMeta}>
            {latestJob
              ? `${formatJobStatus(latestJob.status)} · ${formatDateTime(latestJob.created_at)}`
              : 'Create a fleet or upload a trace to start planning.'}
          </span>
        </div>
      </div>

      {error ? <p className={`${styles.message} ${styles.messageError}`}>{error}</p> : null}

      <div className={styles.splitGrid}>
        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Latest Run</h2>
              <p className={styles.sectionDescription}>The most recent planning scenario, ready for a quick read before you drill into full history.</p>
            </div>
          </div>
          {latestJob ? (
            <>
              <div className={styles.compactListItem}>
                <div className={styles.compactListMeta}>
                  <span className={styles.compactListTitle}>
                    {formatJobType(latestJob.type)} · {latestWorkload?.type === 'builtin'
                      ? formatBuiltinWorkloadName(latestWorkload.name || 'library')
                      : latestTrace?.name || 'Uploaded trace'}
                  </span>
                  <span className={styles.compactListText}>
                    Created {formatDateTime(latestJob.created_at)}
                    {latestFleet ? ` · Fleet ${latestFleet.name}` : ''}
                  </span>
                </div>
                <JobStatusBadge status={latestJob.status} />
              </div>
              <div className={styles.resultCard} style={{ marginTop: '0.9rem' }}>
                {renderJobResultSummary(latestJob)}
              </div>
              <p className={styles.inlineHint}>
                {finishedJobs.length} completed runs are available for comparison in the Runs page.
              </p>
            </>
          ) : (
            <div className={styles.emptyState}>No planning runs yet. Start with a saved fleet or a built-in workload.</div>
          )}
        </section>

        <section className={styles.sectionCard}>
          <div className={styles.sectionHeader}>
            <div>
              <h2 className={styles.sectionTitle}>Planning Assets</h2>
              <p className={styles.sectionDescription}>The reusable workload, trace, and fleet inputs most likely to drive the next scenario.</p>
            </div>
          </div>
          <ul className={styles.compactList}>
            {planningAssetItems.map((asset) => (
              <li key={asset.key} className={styles.compactListItem}>
                <div className={styles.compactListMeta}>
                  <span className={styles.compactListTitle}>{asset.title}</span>
                  <span className={styles.compactListText}>{asset.detail}</span>
                </div>
                <span className={styles.compactListText}>{asset.meta}</span>
              </li>
            ))}
            {planningAssetItems.length === 0 ? (
              <li className={styles.emptyState}>Upload a trace or save a fleet to build out the planning library.</li>
            ) : null}
          </ul>
        </section>
      </div>
    </FleetSimSurfaceLayout>
  )
}
