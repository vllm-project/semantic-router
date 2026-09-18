import type { EvaluationCatalog, EvaluationRun } from '../../types/evaluationPlane'
import {
  decodeEvaluationCapacityLoadProtocol,
  decodeEvaluationCapacitySLO,
  requiresCapacitySLO,
} from '../../utils/evaluationCapacitySLOContract'
import {
  EVALUATION_RUN_LIMITS,
  type EvaluationExactCohort,
  exactCohortMatchesRun,
  reconcileEvaluationScope,
} from './evaluationExperiment'

export interface EvaluationDraft extends EvaluationExactCohort {
  name: string
  description: string
  baselineRunID: string
}

function sameSet<T>(left: T[], right: T[]): boolean {
  const normalizedLeft = new Set(left)
  const normalizedRight = new Set(right)
  return (
    normalizedLeft.size === left.length &&
    normalizedRight.size === right.length &&
    normalizedLeft.size === normalizedRight.size &&
    [...normalizedLeft].every((value) => normalizedRight.has(value))
  )
}

function isBoundedInteger(value: number, minimum: number, maximum: number): boolean {
  return Number.isSafeInteger(value) && value >= minimum && value <= maximum
}

function utf8Length(value: string): number {
  return new TextEncoder().encode(value).length
}

export function baselineCohortIssue(catalog: EvaluationCatalog, run: EvaluationRun): string | null {
  if (run.status !== 'completed') return 'Only completed runs can be used as a baseline.'
  if (!catalog.change_profiles.some((profile) => profile.id === run.change_profile)) {
    return 'Its change type is no longer available.'
  }
  const target = catalog.targets.find((candidate) => candidate.id === run.target_id)
  if (!target) return 'Its evaluation source is no longer available.'
  if (target.healthy === false) return 'Its evaluation source is currently unavailable.'
  if (!target.modes.includes(run.mode))
    return 'Its evaluation source no longer supports this run type.'
  if (!isBoundedInteger(run.sample_limit, 1, EVALUATION_RUN_LIMITS.sampleLimit)) {
    return 'Its sample size is outside the supported range.'
  }
  if (!isBoundedInteger(run.concurrency, 1, EVALUATION_RUN_LIMITS.concurrency)) {
    return 'Its parallel request count is outside the supported range.'
  }
  const capacityRequired = requiresCapacitySLO(run.mode, run.track_ids)
  if (capacityRequired && run.concurrency < 2) {
    return 'Its performance test does not include enough parallel request levels.'
  }
  if (capacityRequired && (!run.capacity_slo || !run.capacity_load_protocol)) {
    return 'Its performance goals or load pattern were not saved with the run.'
  }
  if (!capacityRequired && (run.capacity_slo || run.capacity_load_protocol)) {
    return 'Its saved performance settings do not belong to a live performance test.'
  }
  if (run.capacity_slo) {
    try {
      const capacitySLO = decodeEvaluationCapacitySLO(run.capacity_slo)
      if (capacitySLO.required_concurrency > run.concurrency) {
        return 'Its required parallel load exceeds the run limit.'
      }
    } catch {
      return 'Its saved performance goals are invalid.'
    }
  }
  if (run.capacity_load_protocol) {
    try {
      decodeEvaluationCapacityLoadProtocol(run.capacity_load_protocol, run.concurrency)
    } catch {
      return 'Its saved load pattern is invalid.'
    }
  }
  if (!isBoundedInteger(run.seed, 0, EVALUATION_RUN_LIMITS.seed)) {
    return 'Its repeatability key is outside the supported range.'
  }
  const reconciled = reconcileEvaluationScope(
    catalog,
    run.target_id,
    run.mode,
    run.suite_ids,
    run.track_ids,
  )
  if (
    run.suite_ids.length === 0 ||
    run.track_ids.length === 0 ||
    !sameSet(reconciled.suiteIDs, run.suite_ids) ||
    !sameSet(reconciled.trackIDs, run.track_ids)
  ) {
    return 'Its benchmark selection or evaluation areas are no longer reproducible.'
  }
  return null
}

export function compatibleSuiteEmptyReason(
  catalog: EvaluationCatalog,
  targetID: string,
  mode: EvaluationExactCohort['mode'],
): string {
  const target = catalog.targets.find((candidate) => candidate.id === targetID)
  if (!target) {
    return `Select an available Mixture that supports ${mode}, or choose another run type.`
  }
  if (target.healthy === false) return 'The selected Mixture is unavailable. Choose another one.'
  if (!target.modes.includes(mode)) {
    return `The selected Mixture does not support ${mode} evaluation. Choose another Mixture or run type.`
  }
  return `No benchmark is fully supported by ${target.name} for this ${mode} run.`
}

function validateEvaluationSelection(
  catalog: EvaluationCatalog,
  draft: EvaluationDraft,
): string | null {
  if (!catalog.change_profiles.some((profile) => profile.id === draft.changeProfile)) {
    return 'Select the type of change being evaluated.'
  }
  const target = catalog.targets.find((candidate) => candidate.id === draft.targetID)
  if (!target || target.healthy === false || !target.modes.includes(draft.mode)) {
    return 'Select an available Mixture that supports this run type.'
  }
  if (draft.suiteIDs.length === 0) return 'Select at least one compatible benchmark.'
  if (draft.trackIDs.length === 0)
    return 'Select at least one evaluation area provided by those benchmarks.'
  const reconciled = reconcileEvaluationScope(
    catalog,
    draft.targetID,
    draft.mode,
    draft.suiteIDs,
    draft.trackIDs,
  )
  if (
    !sameSet(reconciled.suiteIDs, draft.suiteIDs) ||
    !sameSet(reconciled.trackIDs, draft.trackIDs)
  ) {
    return 'The selected benchmarks and evaluation areas do not support this Mixture and run type.'
  }
  return null
}

function validateEvaluationCapacity(draft: EvaluationDraft): string | null {
  if (!isBoundedInteger(draft.sampleLimit, 1, EVALUATION_RUN_LIMITS.sampleLimit)) {
    return `Sample limit must be an integer between 1 and ${EVALUATION_RUN_LIMITS.sampleLimit}.`
  }
  if (!isBoundedInteger(draft.concurrency, 1, EVALUATION_RUN_LIMITS.concurrency)) {
    return `Concurrency must be an integer between 1 and ${EVALUATION_RUN_LIMITS.concurrency}.`
  }
  const capacityRequired = requiresCapacitySLO(draft.mode, draft.trackIDs)
  if (capacityRequired && draft.concurrency < 2) {
    return 'Live performance evaluation requires at least two parallel requests.'
  }
  if (capacityRequired && (!draft.capacitySLO || !draft.capacityLoadProtocol)) {
    return 'Define performance goals and a load pattern before creating this live run.'
  }
  if (!capacityRequired && (draft.capacitySLO || draft.capacityLoadProtocol)) {
    return 'Performance settings are available only for live performance evaluation.'
  }
  if (draft.capacitySLO) {
    try {
      const capacitySLO = decodeEvaluationCapacitySLO(draft.capacitySLO)
      if (capacitySLO.required_concurrency > draft.concurrency) {
        return 'Required parallel load cannot exceed the run limit.'
      }
    } catch {
      return 'Review the performance goals and keep every value within the supported range.'
    }
  }
  if (draft.capacityLoadProtocol) {
    try {
      decodeEvaluationCapacityLoadProtocol(draft.capacityLoadProtocol, draft.concurrency)
    } catch {
      return 'Review the load pattern and keep every stage within the supported range.'
    }
  }
  if (!isBoundedInteger(draft.seed, 0, EVALUATION_RUN_LIMITS.seed)) {
    return `Repeatability key must be an integer between 0 and ${EVALUATION_RUN_LIMITS.seed}.`
  }
  return null
}

export function validateEvaluationDraft(
  catalog: EvaluationCatalog,
  runs: EvaluationRun[],
  draft: EvaluationDraft,
): string | null {
  const name = draft.name.trim()
  const description = draft.description.trim()
  if (!name) return 'Experiment name is required.'
  if (utf8Length(name) > EVALUATION_RUN_LIMITS.name) {
    return `Experiment name must be at most ${EVALUATION_RUN_LIMITS.name} bytes.`
  }
  if (utf8Length(description) > EVALUATION_RUN_LIMITS.description) {
    return `Description must be at most ${EVALUATION_RUN_LIMITS.description} bytes.`
  }
  const selectionIssue = validateEvaluationSelection(catalog, draft)
  if (selectionIssue) return selectionIssue
  const capacityIssue = validateEvaluationCapacity(draft)
  if (capacityIssue) return capacityIssue
  if (!draft.baselineRunID) return null

  const baseline = runs.find((run) => run.id === draft.baselineRunID)
  if (!baseline) return 'The selected baseline run is no longer available.'
  const issue = baselineCohortIssue(catalog, baseline)
  if (issue) return `The selected baseline cannot be reproduced. ${issue}`
  return exactCohortMatchesRun(draft, baseline)
    ? null
    : 'The candidate must use the same comparison setup as the selected baseline.'
}
