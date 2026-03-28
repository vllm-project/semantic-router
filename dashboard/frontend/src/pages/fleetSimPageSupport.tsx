import styles from './FleetSimPage.module.css'
import type {
  FleetSimJob,
  FleetSimJobType,
  FleetSimTraceFormat,
  WorkloadRef,
} from '../utils/fleetSimApi'

const BUILTIN_WORKLOAD_LABELS: Record<string, string> = {
  azure: 'Azure chat traffic',
  lmsys: 'LMSYS single-turn',
  lmsys_multiturn: 'LMSYS multi-turn',
  agent_heavy: 'Agent-heavy tools',
}

const BUILTIN_WORKLOAD_DESCRIPTIONS: Record<string, string> = {
  azure: 'General chat traffic with a heavy short-context mix and occasional long prompts.',
  lmsys: 'Single-turn assistant prompts with broad prompt and response length variation.',
  lmsys_multiturn: 'Conversation-heavy traffic with follow-up turns and longer retained context.',
  agent_heavy: 'Tool-using agent requests with larger prompts, longer outputs, and bursty demand.',
}

const TRACE_FORMAT_LABELS: Record<FleetSimTraceFormat, string> = {
  semantic_router: 'Router JSONL',
  jsonl: 'JSONL',
  csv: 'CSV',
}

const JOB_TYPE_LABELS: Record<FleetSimJobType, string> = {
  optimize: 'Optimize',
  simulate: 'Simulate',
  whatif: 'What-if',
}

const JOB_STATUS_LABELS: Record<FleetSimJob['status'], string> = {
  pending: 'Queued',
  running: 'Running',
  done: 'Ready',
  failed: 'Failed',
}

const ROUTER_LABELS: Record<string, string> = {
  length: 'Length split',
  compress_route: 'Compression-aware',
  least_loaded: 'Least loaded',
  random: 'Random',
  model: 'Model-aware',
}

const ROUTER_DESCRIPTIONS: Record<string, string> = {
  length: 'Direct shorter prompts to the fast pool and reserve long-context work for the deeper pool.',
  compress_route: 'Bias traffic toward a cheaper short pool while gradually spilling long requests to the premium pool.',
  least_loaded: 'Keep traffic balanced by sending new work to the least busy pool.',
  random: 'Distribute requests evenly when you only want a neutral baseline.',
  model: 'Pin requests to pools based on model-aware routing logic.',
}

function humanizeKey(value: string): string {
  return value
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

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

export function formatBuiltinWorkloadName(name: string): string {
  return BUILTIN_WORKLOAD_LABELS[name] || humanizeKey(name)
}

export function describeBuiltinWorkload(name: string, fallback?: string): string {
  return BUILTIN_WORKLOAD_DESCRIPTIONS[name] || fallback || 'Reusable traffic mix for planning runs.'
}

export function formatTraceFormat(format: FleetSimTraceFormat): string {
  return TRACE_FORMAT_LABELS[format] || format
}

export function formatGpuLabel(gpu: string): string {
  const upper = gpu.toUpperCase()
  if (upper === 'A10G') return upper
  return upper.replace('_', ' ')
}

export function formatRouterType(router: string): string {
  return ROUTER_LABELS[router] || humanizeKey(router)
}

export function describeRouterType(router: string): string {
  return ROUTER_DESCRIPTIONS[router] || 'Route requests across the available GPU pools.'
}

export function formatJobType(type: FleetSimJobType): string {
  return JOB_TYPE_LABELS[type] || humanizeKey(type)
}

export function describeJobType(type: FleetSimJobType): string {
  switch (type) {
    case 'optimize':
      return 'Search the cheapest short/long GPU split that still meets your latency target.'
    case 'simulate':
      return 'Replay one saved fleet against the selected traffic mix and inspect the outcome.'
    case 'whatif':
      return 'Sweep arrival rates to see where a saved fleet stops meeting its goal.'
    default:
      return 'Review one planning scenario at a time.'
  }
}

export function formatWorkloadRef(workload?: WorkloadRef | null): string {
  if (!workload) return 'Unknown workload'
  if (workload.type === 'trace') {
    return workload.trace_id ? `Trace · ${workload.trace_id}` : 'Trace'
  }
  return workload.name ? `Library · ${formatBuiltinWorkloadName(workload.name)}` : 'Library'
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
  return candidate || 'Search mode'
}

export function formatJobStatus(status: FleetSimJob['status']): string {
  return JOB_STATUS_LABELS[status]
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
  return <span className={jobStatusClassName(status)}>{formatJobStatus(status)}</span>
}

export function renderJobResultSummary(job: FleetSimJob) {
  if (job.status === 'failed') {
    return <p className={`${styles.message} ${styles.messageError}`}>{job.error || 'Job failed'}</p>
  }

  if (job.result_optimize) {
    const best = job.result_optimize.best
    const sourceLabel =
      best.source === 'simulated'
        ? 'Simulation-validated recommendation'
        : best.source === 'analytical'
          ? 'Analytical recommendation'
          : humanizeKey(best.source)
    return (
      <div className={styles.resultSummary}>
        <div className={styles.resultSummaryHeader}>
          <div>
            <span className={styles.resultEyebrow}>Optimize result</span>
            <h4 className={styles.resultTitle}>Recommended fleet split</h4>
            <p className={styles.resultDescription}>
              {sourceLabel} for the selected traffic target.
            </p>
          </div>
          <span
            className={`${styles.resultPill} ${best.slo_met ? styles.resultPillSuccess : styles.resultPillWarning}`}
          >
            {best.slo_met ? 'SLO matched' : 'Needs review'}
          </span>
        </div>
        <div className={styles.resultGrid}>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Best gamma</span>
            <span className={styles.resultMetricValue}>{best.gamma.toFixed(2)}</span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Fleet GPUs</span>
            <span className={styles.resultMetricValue}>{formatNumber(best.total_gpus)}</span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Annual cost</span>
            <span className={styles.resultMetricValue}>{formatMoneyKusd(best.annual_cost_kusd)}</span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Savings vs baseline</span>
            <span className={styles.resultMetricValue}>{job.result_optimize.savings_pct.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    )
  }

  if (job.result_simulate) {
    return (
      <div className={styles.resultSummary}>
        <div className={styles.resultSummaryHeader}>
          <div>
            <span className={styles.resultEyebrow}>Simulation result</span>
            <h4 className={styles.resultTitle}>Saved fleet replay</h4>
            <p className={styles.resultDescription}>
              Replay outcome for the selected fleet and traffic input.
            </p>
          </div>
          <span
            className={`${styles.resultPill} ${
              job.result_simulate.fleet_slo_compliance >= 0.95
                ? styles.resultPillSuccess
                : styles.resultPillWarning
            }`}
          >
            {formatPercent(job.result_simulate.fleet_slo_compliance)} hit rate
          </span>
        </div>
        <div className={styles.resultGrid}>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Fleet P99 TTFT</span>
            <span className={styles.resultMetricValue}>
              {formatNumber(job.result_simulate.fleet_p99_ttft_ms, 1)} ms
            </span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Mean utilization</span>
            <span className={styles.resultMetricValue}>
              {formatPercent(job.result_simulate.fleet_mean_utilisation)}
            </span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Fleet GPUs</span>
            <span className={styles.resultMetricValue}>{formatNumber(job.result_simulate.total_gpus)}</span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Annual cost</span>
            <span className={styles.resultMetricValue}>
              {formatMoneyKusd(job.result_simulate.annual_cost_kusd)}
            </span>
          </div>
        </div>
      </div>
    )
  }

  if (job.result_whatif) {
    const maxLam = job.result_whatif.points.reduce(
      (current, point) => Math.max(current, point.lam),
      0
    )
    return (
      <div className={styles.resultSummary}>
        <div className={styles.resultSummaryHeader}>
          <div>
            <span className={styles.resultEyebrow}>What-if result</span>
            <h4 className={styles.resultTitle}>Traffic envelope</h4>
            <p className={styles.resultDescription}>
              Arrival-rate sweep for one saved fleet configuration.
            </p>
          </div>
          <span
            className={`${styles.resultPill} ${
              job.result_whatif.slo_break_lam != null
                ? styles.resultPillWarning
                : styles.resultPillSuccess
            }`}
          >
            {job.result_whatif.slo_break_lam != null ? 'Break point found' : 'Stable across sweep'}
          </span>
        </div>
        <div className={styles.resultGrid}>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Sweep points</span>
            <span className={styles.resultMetricValue}>
              {formatNumber(job.result_whatif.points.length)}
            </span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Break lambda</span>
            <span className={styles.resultMetricValue}>
              {job.result_whatif.slo_break_lam != null
                ? formatNumber(job.result_whatif.slo_break_lam, 1)
                : 'Stable'}
            </span>
          </div>
          <div className={styles.resultMetric}>
            <span className={styles.resultMetricLabel}>Highest tested lambda</span>
            <span className={styles.resultMetricValue}>{formatNumber(maxLam, 1)}</span>
          </div>
        </div>
      </div>
    )
  }

  return <p className={styles.message}>Result pending.</p>
}

export function renderJobResultRows(job: FleetSimJob) {
  if (job.status === 'failed') {
    return <p className={`${styles.message} ${styles.messageError}`}>{job.error || 'Job failed'}</p>
  }

  if (job.result_optimize) {
    const best = job.result_optimize.best
    return (
      <div className={styles.inlineDetails}>
        <div className={styles.inlineDetailRow}>
          <span className={styles.inlineDetailLabel}>Result</span>
          <span className={styles.inlineDetailValue}>Recommended fleet split</span>
        </div>
        <div className={styles.inlineDetailGrid}>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Best gamma</span>
            <span className={styles.inlineDetailValue}>{best.gamma.toFixed(2)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Fleet GPUs</span>
            <span className={styles.inlineDetailValue}>{formatNumber(best.total_gpus)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Annual cost</span>
            <span className={styles.inlineDetailValue}>{formatMoneyKusd(best.annual_cost_kusd)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Savings</span>
            <span className={styles.inlineDetailValue}>{job.result_optimize.savings_pct.toFixed(1)}%</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>SLO</span>
            <span className={styles.inlineDetailValue}>{best.slo_met ? 'Matched' : 'Needs review'}</span>
          </div>
        </div>
      </div>
    )
  }

  if (job.result_simulate) {
    return (
      <div className={styles.inlineDetails}>
        <div className={styles.inlineDetailRow}>
          <span className={styles.inlineDetailLabel}>Result</span>
          <span className={styles.inlineDetailValue}>Saved fleet replay</span>
        </div>
        <div className={styles.inlineDetailGrid}>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>P99 TTFT</span>
            <span className={styles.inlineDetailValue}>{formatNumber(job.result_simulate.fleet_p99_ttft_ms, 1)} ms</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Hit rate</span>
            <span className={styles.inlineDetailValue}>{formatPercent(job.result_simulate.fleet_slo_compliance)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Utilization</span>
            <span className={styles.inlineDetailValue}>{formatPercent(job.result_simulate.fleet_mean_utilisation)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Fleet GPUs</span>
            <span className={styles.inlineDetailValue}>{formatNumber(job.result_simulate.total_gpus)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Annual cost</span>
            <span className={styles.inlineDetailValue}>{formatMoneyKusd(job.result_simulate.annual_cost_kusd)}</span>
          </div>
        </div>
      </div>
    )
  }

  if (job.result_whatif) {
    const maxLam = job.result_whatif.points.reduce((current, point) => Math.max(current, point.lam), 0)
    return (
      <div className={styles.inlineDetails}>
        <div className={styles.inlineDetailRow}>
          <span className={styles.inlineDetailLabel}>Result</span>
          <span className={styles.inlineDetailValue}>Traffic envelope</span>
        </div>
        <div className={styles.inlineDetailGrid}>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Sweep points</span>
            <span className={styles.inlineDetailValue}>{formatNumber(job.result_whatif.points.length)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Break lambda</span>
            <span className={styles.inlineDetailValue}>
              {job.result_whatif.slo_break_lam != null ? formatNumber(job.result_whatif.slo_break_lam, 1) : 'Stable'}
            </span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Highest tested</span>
            <span className={styles.inlineDetailValue}>{formatNumber(maxLam, 1)}</span>
          </div>
          <div className={styles.inlineDetailCell}>
            <span className={styles.inlineDetailLabel}>Sweep status</span>
            <span className={styles.inlineDetailValue}>
              {job.result_whatif.slo_break_lam != null ? 'Break point found' : 'Stable across sweep'}
            </span>
          </div>
        </div>
      </div>
    )
  }

  return <p className={styles.message}>Result pending.</p>
}
