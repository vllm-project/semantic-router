import type { EvaluationChangeProfileId, EvaluationRun } from '../../types/evaluationPlane'
import { evaluationResultScopeLabel } from './evaluationPresentation'

type RunOptionIdentity = Pick<
  EvaluationRun,
  'id' | 'name' | 'change_profile' | 'mode' | 'evidence_level' | 'sample_limit'
>

const CHANGE_PROFILE_LABELS: Record<string, string> = {
  schema_adapter: 'API and integration',
  recipe: 'Routing recipe',
  selector: 'Model selection',
  model_pool: 'Model pool',
  runtime_capacity: 'Runtime and capacity',
  agent_multimodal: 'Agents and multimodal',
  online_adaptation: 'Online learning and feedback',
}

type RunCohortIdentity = Pick<EvaluationRun, 'mixture'>
type RunTargetIdentity = Pick<EvaluationRun, 'target_id' | 'mixture'>
type RunWorkloadIdentity = Pick<EvaluationRun, 'sample_limit' | 'concurrency'>

export function changeProfileLabel(profile: EvaluationChangeProfileId): string {
  return CHANGE_PROFILE_LABELS[profile] || 'Evaluation change'
}

/**
 * A run retains an internal target identifier for attestation, but the Compare
 * view should orient people with the evaluated public Mixture instead.
 */
export function runCohortTargetLabel(run: RunCohortIdentity): string {
  return run.mixture?.entrypoint_model || 'Frozen deployment snapshot'
}

export function runEvaluationTargetLabel(run: RunTargetIdentity): string {
  if (run.mixture?.entrypoint_model) return run.mixture.entrypoint_model
  return 'Saved evaluation target'
}

export function runWorkloadLabel(run: RunWorkloadIdentity): string {
  const cases = `${run.sample_limit} ${run.sample_limit === 1 ? 'case' : 'cases'}`
  const concurrency = `${run.concurrency} concurrent ${run.concurrency === 1 ? 'request' : 'requests'}`
  return `${cases} · ${concurrency}`
}

function runNameLabels<T extends Pick<EvaluationRun, 'id' | 'name'>>(runs: readonly T[]) {
  const totals = new Map<string, number>()
  const seen = new Map<string, number>()
  runs.forEach((run) => totals.set(run.name, (totals.get(run.name) || 0) + 1))
  return new Map(
    runs.map((run) => {
      const occurrence = (seen.get(run.name) || 0) + 1
      seen.set(run.name, occurrence)
      return [run.id, totals.get(run.name) === 1 ? run.name : `${run.name} · Option ${occurrence}`]
    }),
  )
}

function distinctRuns<T extends Pick<EvaluationRun, 'id'>>(runs: readonly T[]): T[] {
  return [...new Map(runs.map((run) => [run.id, run])).values()]
}

export function runOptionLabels(runs: readonly RunOptionIdentity[]): Map<string, string> {
  const distinct = distinctRuns(runs)
  const names = runNameLabels(distinct)
  return new Map(
    distinct.map((run) => [
      run.id,
      [
        names.get(run.id),
        changeProfileLabel(run.change_profile),
        run.mode === 'live' ? 'Live' : 'Replay',
        evaluationResultScopeLabel(run.evidence_level),
        `${run.sample_limit} ${run.sample_limit === 1 ? 'case' : 'cases'}`,
      ].join(' · '),
    ]),
  )
}

export function comparisonRunOptionLabels(
  runs: readonly Pick<EvaluationRun, 'id' | 'name'>[],
): Map<string, string> {
  return runNameLabels(distinctRuns(runs))
}
