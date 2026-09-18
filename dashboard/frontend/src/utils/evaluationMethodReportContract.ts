import type {
  EvaluationMethodAnalysisPlan,
  EvaluationMethodCurvePoint,
  EvaluationMethodDescriptor,
  EvaluationMethodReport,
  EvaluationMethodSlice,
} from '../types/evaluationMethodReport'
import {
  EVALUATION_EVIDENCE_LEVEL_SET,
  EVALUATION_TRACK_ID_SET,
  hasOnlyEvaluationFields,
  isEvaluationRecord,
  isFiniteNumber,
  isNonEmptyText,
  isNonNegativeInteger,
} from './evaluationContractValidation'

type EvaluationMethodCurvePointWire = Omit<EvaluationMethodCurvePoint, 'action'> & {
  action: EvaluationMethodSlice
}
type EvaluationMethodReportWire = Omit<EvaluationMethodReport, 'raw_shared_domain_curve'> & {
  raw_shared_domain_curve: EvaluationMethodCurvePointWire[]
}

const METHOD_V2 = 'evaluation-method.v2'
const METHOD_STATUSES = new Set([
  'native-qualified',
  'exploratory-import',
  'data-required',
  'blocked',
])
const METHOD_OWNERS = new Set(['server', 'worker', 'provider', 'benchmark_native'])
const METHOD_PARITIES = new Set(['native', 'source_qualified', 'none'])

function isMethodIdentity(value: unknown): value is string {
  return typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)
}

function isMethodIdentityArray(value: unknown, nonEmpty = false): value is string[] {
  return Array.isArray(value) && (!nonEmpty || value.length > 0) && value.every(isMethodIdentity)
}

function isMethodTrackArray(value: unknown, nonEmpty = false): value is string[] {
  return (
    Array.isArray(value) &&
    (!nonEmpty || value.length > 0) &&
    value.every((track) => typeof track === 'string' && EVALUATION_TRACK_ID_SET.has(track))
  )
}

function isMethodMetricIDArray(value: unknown): value is string[] {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (metricID) =>
        typeof metricID === 'string' && metricID.length > 0 && metricID.trim() === metricID,
    )
  )
}

function isMethodSlice(value: unknown): value is EvaluationMethodSlice {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, ['schema_version', 'id']) &&
    value.schema_version === METHOD_V2 &&
    isMethodIdentity(value.id)
  )
}

function isMethodAnalysisPlan(value: unknown): value is EvaluationMethodAnalysisPlan {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, [
      'schema_version',
      'id',
      'analysis_unit',
      'cluster_unit',
      'slices',
      'curve_domain',
      'missingness',
    ]) &&
    value.schema_version === METHOD_V2 &&
    isMethodIdentity(value.id) &&
    isNonEmptyText(value.analysis_unit) &&
    isNonEmptyText(value.cluster_unit) &&
    Array.isArray(value.slices) &&
    value.slices.length > 0 &&
    value.slices.every(isMethodSlice) &&
    new Set(value.slices.map((slice) => slice.id)).size === value.slices.length &&
    (value.curve_domain === 'shared_budget' || value.curve_domain === 'not_applicable') &&
    value.missingness === 'fail_closed'
  )
}

function sameMethodAnalysisPlan(
  left: EvaluationMethodAnalysisPlan,
  right: EvaluationMethodAnalysisPlan,
): boolean {
  return (
    left.schema_version === right.schema_version &&
    left.id === right.id &&
    left.analysis_unit === right.analysis_unit &&
    left.cluster_unit === right.cluster_unit &&
    left.curve_domain === right.curve_domain &&
    left.missingness === right.missingness &&
    left.slices.length === right.slices.length &&
    left.slices.every((slice, index) => slice.id === right.slices[index]?.id)
  )
}

function isMethodCurvePoint(value: unknown): value is EvaluationMethodCurvePointWire {
  return (
    isEvaluationRecord(value) &&
    hasOnlyEvaluationFields(value, ['action', 'budget', 'mean_score', 'case_count']) &&
    isMethodSlice(value.action) &&
    isNonNegativeInteger(value.budget) &&
    value.budget > 0 &&
    isFiniteNumber(value.mean_score) &&
    value.mean_score >= 0 &&
    value.mean_score <= 1 &&
    isNonNegativeInteger(value.case_count) &&
    value.case_count > 0
  )
}

function hasUniqueMethodIdentities(values: string[]): boolean {
  return new Set(values).size === values.length
}

export function isEvaluationMethodDescriptor(value: unknown): value is EvaluationMethodDescriptor {
  if (!isEvaluationRecord(value)) return false
  const applicableTracks = value.applicable_tracks
  const liveTracks = value.live_tracks
  const producedMetricIDs = value.produced_metric_ids
  const requiredArtifactIDs = value.required_artifact_ids
  if (
    !hasOnlyEvaluationFields(value, [
      'schema_version',
      'id',
      'version',
      'status',
      'execution_owner',
      'input_schema',
      'export_schema',
      'live_input_complete',
      'live_grader',
      'applicable_tracks',
      'live_tracks',
      'produced_metric_ids',
      'evidence_ceiling',
      'native_parity',
      'required_artifact_ids',
      'analysis_plan',
    ]) ||
    value.schema_version !== METHOD_V2 ||
    value.version !== METHOD_V2 ||
    !isMethodIdentity(value.id) ||
    typeof value.status !== 'string' ||
    !METHOD_STATUSES.has(value.status) ||
    typeof value.execution_owner !== 'string' ||
    !METHOD_OWNERS.has(value.execution_owner) ||
    !isMethodIdentity(value.input_schema) ||
    !isMethodIdentity(value.export_schema) ||
    typeof value.live_input_complete !== 'boolean' ||
    typeof value.live_grader !== 'boolean' ||
    !isMethodTrackArray(applicableTracks, true) ||
    !isMethodTrackArray(liveTracks) ||
    !isMethodMetricIDArray(producedMetricIDs) ||
    typeof value.evidence_ceiling !== 'string' ||
    !EVALUATION_EVIDENCE_LEVEL_SET.has(value.evidence_ceiling) ||
    typeof value.native_parity !== 'string' ||
    !METHOD_PARITIES.has(value.native_parity) ||
    !isMethodIdentityArray(requiredArtifactIDs, true) ||
    !isMethodAnalysisPlan(value.analysis_plan)
  ) {
    return false
  }
  const identities = [
    applicableTracks,
    liveTracks,
    producedMetricIDs,
    requiredArtifactIDs,
  ]
  if (
    identities.some((values) => !hasUniqueMethodIdentities(values)) ||
    liveTracks.some((track) => !applicableTracks.includes(track)) ||
    (value.native_parity === 'native' && value.execution_owner !== 'benchmark_native')
  ) {
    return false
  }
  return value.status === 'native-qualified'
    ? value.live_input_complete && value.live_grader && liveTracks.length > 0
    : !value.live_input_complete && !value.live_grader
}

function hasValidMethodReportShape(
  value: Record<string, unknown>,
): value is Record<string, unknown> & EvaluationMethodReportWire {
  return (
    hasOnlyEvaluationFields(value, [
      'method',
      'analysis_plan',
      'action_refs',
      'slice_refs',
      'raw_shared_domain_curve',
      'audc',
      'nauc',
      'peak',
      'qnc',
      'missing_case_action_budget_cells',
    ]) &&
    isEvaluationMethodDescriptor(value.method) &&
    isMethodAnalysisPlan(value.analysis_plan) &&
    Array.isArray(value.action_refs) &&
    value.action_refs.every(isMethodSlice) &&
    Array.isArray(value.slice_refs) &&
    value.slice_refs.every(isMethodSlice) &&
    Array.isArray(value.raw_shared_domain_curve) &&
    value.raw_shared_domain_curve.every(isMethodCurvePoint) &&
    isFiniteNumber(value.audc) &&
    value.audc >= 0 &&
    isFiniteNumber(value.nauc) &&
    value.nauc >= 0 &&
    value.nauc <= 1 &&
    isFiniteNumber(value.peak) &&
    value.peak >= 0 &&
    value.peak <= 1 &&
    isFiniteNumber(value.qnc) &&
    value.qnc >= 0 &&
    value.qnc <= 1 &&
    value.missing_case_action_budget_cells === 0
  )
}

export function isEvaluationMethodReport(value: unknown): value is EvaluationMethodReport {
  if (!isEvaluationRecord(value) || !hasValidMethodReportShape(value)) return false
  const report = value
  if (
    report.action_refs.length === 0 ||
    report.slice_refs.length === 0 ||
    report.raw_shared_domain_curve.length === 0 ||
    !hasUniqueMethodIdentities(report.action_refs.map((action) => action.id)) ||
    !hasUniqueMethodIdentities(report.slice_refs.map((slice) => slice.id)) ||
    !sameMethodAnalysisPlan(report.method.analysis_plan, report.analysis_plan) ||
    !sameMethodAnalysisPlan(report.analysis_plan, {
      ...report.analysis_plan,
      slices: report.slice_refs,
    })
  ) {
    return false
  }
  const declaredActions = new Set(report.action_refs.map((action) => action.id))
  const coordinates = new Set<string>()
  for (const point of report.raw_shared_domain_curve) {
    const coordinate = `${point.action.id}\u0000${point.budget}`
    if (!declaredActions.has(point.action.id) || coordinates.has(coordinate)) return false
    coordinates.add(coordinate)
  }
  return true
}
