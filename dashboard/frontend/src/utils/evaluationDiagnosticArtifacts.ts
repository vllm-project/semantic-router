import type { EvaluationDiagnosticArtifactIssue } from '../types/evaluationReportDiagnostics'
import { InvalidEvaluationDiagnosticArtifactError } from './evaluationDiagnosticArtifactValidation'

export function evaluationDiagnosticArtifactIssue(
  artifactName: string,
  reason: unknown,
): EvaluationDiagnosticArtifactIssue {
  if (reason instanceof InvalidEvaluationDiagnosticArtifactError || reason instanceof SyntaxError) {
    return {
      kind: 'invalid',
      artifactName,
      message: `${artifactName} did not match the required evaluation.v1 diagnostic schema.`,
    }
  }
  return {
    kind: 'unavailable',
    artifactName,
    message: `${artifactName} could not be loaded. ${reason instanceof Error ? reason.message : 'The artifact request failed.'}`,
  }
}
