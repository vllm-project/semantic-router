import type {
  EvaluationFailureSummary,
  EvaluationFailureSummaryRow,
} from '../types/evaluationReport'
import type { EvaluationTrackId } from '../types/evaluationPlane'
import { EVALUATION_TRACK_IDS } from '../types/evaluationPlane'
import {
  boundedInteger,
  invalid,
  recordWithExactKeys,
} from './evaluationDiagnosticArtifactValidation'

const ARTIFACT_NAME = 'failure-summary.json'
const TRACK_IDS = new Set<string>(EVALUATION_TRACK_IDS)

function decodeRow(value: unknown, index: number): EvaluationFailureSummaryRow {
  const path = `by_track[${index}]`
  const row = recordWithExactKeys(
    value,
    ['track_id', 'succeeded', 'failed', 'unavailable'],
    ARTIFACT_NAME,
    path,
  )
  if (typeof row.track_id !== 'string' || !TRACK_IDS.has(row.track_id)) {
    invalid(ARTIFACT_NAME, `${path}.track_id is not a supported evaluation track`)
  }
  return {
    track_id: row.track_id as EvaluationTrackId,
    succeeded: boundedInteger(row.succeeded, ARTIFACT_NAME, `${path}.succeeded`),
    failed: boundedInteger(row.failed, ARTIFACT_NAME, `${path}.failed`),
    unavailable: boundedInteger(row.unavailable, ARTIFACT_NAME, `${path}.unavailable`),
  }
}

export function decodeEvaluationFailureSummary(value: unknown): EvaluationFailureSummary {
  const root = recordWithExactKeys(
    value,
    ['schema_version', 'total_records', 'failed', 'unavailable', 'by_track'],
    ARTIFACT_NAME,
    'artifact',
  )
  if (root.schema_version !== 'evaluation.v1') {
    invalid(ARTIFACT_NAME, 'schema_version must be evaluation.v1')
  }
  if (!Array.isArray(root.by_track) || root.by_track.length > EVALUATION_TRACK_IDS.length) {
    invalid(ARTIFACT_NAME, 'by_track must be a bounded array')
  }
  const byTrack = root.by_track.map(decodeRow)
  if (new Set(byTrack.map((row) => row.track_id)).size !== byTrack.length) {
    invalid(ARTIFACT_NAME, 'by_track contains duplicate tracks')
  }
  const totalRecords = boundedInteger(root.total_records, ARTIFACT_NAME, 'total_records')
  const failed = boundedInteger(root.failed, ARTIFACT_NAME, 'failed')
  const unavailable = boundedInteger(root.unavailable, ARTIFACT_NAME, 'unavailable')
  const totals = byTrack.reduce(
    (result, row) => ({
      records: result.records + row.succeeded + row.failed + row.unavailable,
      failed: result.failed + row.failed,
      unavailable: result.unavailable + row.unavailable,
    }),
    { records: 0, failed: 0, unavailable: 0 },
  )
  if (
    totals.records !== totalRecords ||
    totals.failed !== failed ||
    totals.unavailable !== unavailable
  ) {
    invalid(ARTIFACT_NAME, 'aggregate counts do not match by_track')
  }
  return {
    schema_version: 'evaluation.v1',
    total_records: totalRecords,
    failed,
    unavailable,
    by_track: byTrack,
  }
}
