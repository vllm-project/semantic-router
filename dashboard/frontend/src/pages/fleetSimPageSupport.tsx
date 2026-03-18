import styles from './FleetSimPage.module.css'
import type { FleetSimJob, WorkloadRef } from '../utils/fleetSimApi'

export function formatNumber(value: number, maximumFractionDigits = 0): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits }).format(value)
}

export function formatMoneyKusd(value: number): string {
  return `${formatNumber(value, 1)} KUSD`
}

export function formatPercent(value: number): string {
  return `${formatNumber(value * 100, 1)}%`
}

export function formatDateTime(value?: string | null): string {
  if (!value) return 'Pending'
  return new Date(value).toLocaleString()
}

export function formatWorkloadRef(workload?: WorkloadRef | null): string {
  if (!workload) return 'Unknown workload'
  if (workload.type === 'trace') {
    return workload.trace_id ? `Trace · ${workload.trace_id}` : 'Trace'
  }
  return workload.name ? `Built-in · ${workload.name}` : 'Built-in'
}

export function extractJobWorkload(job: FleetSimJob): WorkloadRef | null {
  return (job.request.optimize?.workload ||
    job.request.simulate?.workload ||
    job.request.whatif?.workload ||
    null) as WorkloadRef | null
}

export function extractJobFleetID(job: FleetSimJob): string {
  const candidate =
    (job.request.simulate?.fleet_id as string | undefined) ||
    (job.request.whatif?.fleet_id as string | undefined)
  return candidate || 'Ad hoc'
}

export function jobStatusClassName(status: FleetSimJob['status']): string {
  switch (status) {
    case 'done':
      return `${styles.statusBadge} ${styles.statusDone}`
    case 'failed':
      return `${styles.statusBadge} ${styles.statusFailed}`
    case 'running':
      return `${styles.statusBadge} ${styles.statusRunning}`
    default:
      return `${styles.statusBadge} ${styles.statusPending}`
  }
}

export function JobStatusBadge({ status }: { status: FleetSimJob['status'] }) {
  return <span className={jobStatusClassName(status)}>{status}</span>
}

export function renderJobResultSummary(job: FleetSimJob) {
  if (job.status === 'failed') {
    return <p className={`${styles.message} ${styles.messageError}`}>{job.error || 'Job failed'}</p>
  }

  if (job.result_optimize) {
    const best = job.result_optimize.best
    return (
      <div className={styles.resultGrid}>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Best Gamma</span>
          <span className={styles.resultMetricValue}>{best.gamma.toFixed(2)}</span>
        </div>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Fleet GPUs</span>
          <span className={styles.resultMetricValue}>{formatNumber(best.total_gpus)}</span>
        </div>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Annual Cost</span>
          <span className={styles.resultMetricValue}>{formatMoneyKusd(best.annual_cost_kusd)}</span>
        </div>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Savings</span>
          <span className={styles.resultMetricValue}>{job.result_optimize.savings_pct.toFixed(1)}%</span>
        </div>
      </div>
    )
  }

  if (job.result_simulate) {
    return (
      <div className={styles.resultGrid}>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Fleet P99 TTFT</span>
          <span className={styles.resultMetricValue}>{formatNumber(job.result_simulate.fleet_p99_ttft_ms, 1)} ms</span>
        </div>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>SLO Compliance</span>
          <span className={styles.resultMetricValue}>{formatPercent(job.result_simulate.fleet_slo_compliance)}</span>
        </div>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Fleet GPUs</span>
          <span className={styles.resultMetricValue}>{formatNumber(job.result_simulate.total_gpus)}</span>
        </div>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Annual Cost</span>
          <span className={styles.resultMetricValue}>{formatMoneyKusd(job.result_simulate.annual_cost_kusd)}</span>
        </div>
      </div>
    )
  }

  if (job.result_whatif) {
    return (
      <div className={styles.resultGrid}>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Sweep Points</span>
          <span className={styles.resultMetricValue}>{formatNumber(job.result_whatif.points.length)}</span>
        </div>
        <div className={styles.resultMetric}>
          <span className={styles.resultMetricLabel}>Break Lambda</span>
          <span className={styles.resultMetricValue}>
            {job.result_whatif.slo_break_lam ? formatNumber(job.result_whatif.slo_break_lam, 1) : 'Stable'}
          </span>
        </div>
      </div>
    )
  }

  return <p className={styles.message}>Result pending.</p>
}
