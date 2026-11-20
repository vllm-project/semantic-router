import type { StatusTone } from '../../ui'
import { modelProviderId } from '../../pages/insightsConfigFacts'
import type { ExperimentMember } from './experimentApi'
import { experimentRoleLabels } from './experimentEvidencePresentation'
import { active, hasScoredOutcomes } from './model'
import { targetName } from './targetPresentation'
import type { ExperimentRunContext, Manifest, Run, Target, TargetMetrics } from './types'

export const RUN_STATUSES = [
  'running',
  'queued',
  'completed',
  'failed',
  'interrupted',
  'cancelled',
] as const
export const RUN_MODES = ['live', 'preview', 'replay'] as const
export const RUN_PROFILES = ['smoke', 'quick', 'standard'] as const

export interface RunFilters {
  query: string
  status: string[]
  mode: string[]
  benchmark: string[]
  profile: string[]
  page: number
}

const list = (value: string | null, allowed?: readonly string[]) =>
  (value ?? '')
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item && (!allowed || allowed.includes(item)))

/** Multi-value filters live in the URL as comma lists; `active` is kept for older links. */
export function parseRunFilters(params: URLSearchParams): RunFilters {
  const status = list(params.get('status')).flatMap((item) =>
    item === 'active' ? ['running', 'queued'] : RUN_STATUSES.includes(item as never) ? [item] : [],
  )
  return {
    query: params.get('q') ?? '',
    status: [...new Set(status)],
    mode: list(params.get('mode'), RUN_MODES),
    benchmark: list(params.get('benchmark')),
    profile: list(params.get('profile'), RUN_PROFILES),
    page: Math.max(0, Math.floor(Number(params.get('page')) || 0)),
  }
}

/** URL entries for a filter patch; empty values remove their parameter. */
export function runFilterEntries(patch: Partial<RunFilters>): Array<[string, string | null]> {
  return Object.entries(patch).map(([field, value]) => {
    const key = field === 'query' ? 'q' : field
    if (Array.isArray(value)) return [key, value.length ? value.join(',') : null]
    if (value === undefined || value === '' || value === 0) return [key, null]
    return [key, String(value)]
  })
}

export const runBenchmarks = (run: Run): string[] => run.manifest.dataset?.benchmarks ?? []

export function filterRuns(runs: Run[], filters: RunFilters): Run[] {
  const query = filters.query.trim().toLowerCase()
  return runs.filter((run) => {
    if (filters.status.length && !filters.status.includes(run.status)) return false
    if (filters.mode.length && !filters.mode.includes(run.manifest.mode)) return false
    if (filters.profile.length && !filters.profile.includes(run.manifest.profile)) return false
    if (
      filters.benchmark.length &&
      !runBenchmarks(run).some((benchmark) => filters.benchmark.includes(benchmark))
    )
      return false
    if (!query) return true
    return [
      run.id,
      run.manifest.name,
      run.manifest.profile,
      ...run.manifest.targets.map((target) => `${target.id} ${target.kind} ${target.model}`),
    ]
      .join(' ')
      .toLowerCase()
      .includes(query)
  })
}

export const runProgressShare = (run: Pick<Run, 'progress'>): number =>
  run.progress.total
    ? Math.min(1, (run.progress.completed + run.progress.failed) / run.progress.total)
    : 0

export interface BestScore {
  score: number
  target: string
}

/** Highest macro accuracy among targets whose every planned outcome is scored. */
export function bestScore(
  targets: TargetMetrics[],
  manifest: Pick<Manifest, 'targets' | 'auxiliary_targets'>,
): BestScore | null {
  let best: BestScore | null = null
  for (const target of targets) {
    const value =
      typeof target.macro_accuracy === 'number' ? target.macro_accuracy : target.accuracy
    if (typeof value !== 'number' || !Number.isFinite(value) || !hasScoredOutcomes(target)) continue
    if (!best || value * 100 > best.score)
      best = { score: value * 100, target: targetName(manifest, target.id) }
  }
  return best
}

/** Only live runs measure capability; dry runs and offline estimates have no score. */
export const runBestScore = (run: Run): BestScore | null =>
  run.manifest.mode === 'live' ? bestScore(run.summary?.targets ?? [], run.manifest) : null

/** Recorded spend across targets; unknown when any target has no recorded cost. */
export function runCost(run: Run): number | null {
  const targets = run.summary?.targets ?? []
  if (!targets.length) return null
  let total = 0
  for (const target of targets) {
    const value = target.total_spend_usd ?? target.cost_usd
    if (typeof value !== 'number' || !Number.isFinite(value)) return null
    total += value
  }
  return total
}

export const runModeLabel = (mode: string) =>
  mode === 'preview' ? 'Dry run' : mode === 'replay' ? 'Offline estimate' : 'Live'

export function relativeTime(iso: string, now = Date.now()): string {
  const time = Date.parse(iso)
  if (!Number.isFinite(time)) return '—'
  const seconds = Math.max(0, Math.round((now - time) / 1000))
  if (seconds < 45) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.round(hours / 24)
  if (days < 30) return `${days}d ago`
  return new Date(time).toLocaleDateString()
}

export interface SequencedRun {
  member: ExperimentMember
  run?: Run
  label: string
}

const ROLE_ORDER: Partial<Record<ExperimentRunContext['role'], number>> = {
  baseline: 0,
  initial: 1,
  candidate: 2,
  validation: 3,
}

/** An experiment reads as a sequence: reference, starting recipe, numbered versions, validation, then checks. */
export function sequenceExperiment(members: ExperimentMember[], runs: Run[]): SequencedRun[] {
  const byID = new Map(runs.map((run) => [run.id, run]))
  const created = (member: ExperimentMember) =>
    byID.get(member.run_id)?.created_at ?? member.linked_at
  const sorted = [...members].sort(
    (a, b) =>
      (ROLE_ORDER[a.role] ?? 4) - (ROLE_ORDER[b.role] ?? 4) ||
      created(a).localeCompare(created(b)) ||
      a.run_id.localeCompare(b.run_id),
  )
  let version = 0
  return sorted.map((member) => {
    const label =
      member.role === 'baseline'
        ? 'Reference'
        : member.role === 'candidate'
          ? `Version ${++version}`
          : experimentRoleLabels[member.role]
    return { member, run: byID.get(member.run_id), label }
  })
}

export const isSupportingRole = (role: ExperimentRunContext['role']) =>
  ROLE_ORDER[role] === undefined

export { active }

const statusPresentation: Record<Run['status'], { label: string; tone: StatusTone }> = {
  running: { label: 'Running', tone: 'info' },
  queued: { label: 'Queued', tone: 'neutral' },
  completed: { label: 'Completed', tone: 'success' },
  failed: { label: 'Failed', tone: 'danger' },
  interrupted: { label: 'Interrupted', tone: 'warning' },
  cancelled: { label: 'Cancelled', tone: 'neutral' },
}

export const runStatusLabel = (status: Run['status']) => statusPresentation[status]?.label ?? status
export const runStatusTone = (status: Run['status']): StatusTone =>
  statusPresentation[status]?.tone ?? 'neutral'

/** The organisation a target's model is published under, when it has a logo. */
export const targetProvider = (target: Pick<Target, 'model'>) =>
  modelProviderId(target.model, {}) ?? 'vllm'

export const RUN_SECTIONS = ['results', 'questions', 'calls', 'activity', 'protocol'] as const
export type RunSection = (typeof RUN_SECTIONS)[number]

/** Older links used `evidence` and `recipe`; they now live under Activity and Protocol. */
export function runSection(value: string | null): RunSection {
  if (value === 'evidence') return 'activity'
  if (value === 'recipe') return 'protocol'
  return RUN_SECTIONS.includes(value as RunSection) ? (value as RunSection) : 'results'
}
