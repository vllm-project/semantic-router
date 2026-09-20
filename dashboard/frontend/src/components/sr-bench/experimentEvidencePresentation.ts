import type { ExperimentRunContext, Run } from './types'

export const experimentRoleLabels: Record<ExperimentRunContext['role'], string> = {
  baseline: 'Single-model reference',
  initial: 'Starting recipe',
  candidate: 'Recipe version',
  validation: 'Final validation',
  preview: 'Routing preview',
  smoke: 'Pipeline check',
  estimate: 'Offline estimate',
  recovery: 'Recovery attempt',
}

export function experimentQuestionCount(run: Run): number | undefined {
  if (run.manifest.recovery || run.manifest.execution_cells !== undefined) return undefined
  const cases = run.manifest.cases
  const inlineCount = Array.isArray(cases) && cases.length > 0 ? cases.length : undefined
  const dataset = run.manifest.dataset
  const recordedCount = dataset && 'case_count' in dataset ? dataset.case_count : undefined
  const datasetCount =
    typeof recordedCount === 'number' && Number.isSafeInteger(recordedCount) && recordedCount > 0
      ? recordedCount
      : undefined
  if (inlineCount !== undefined && datasetCount !== undefined && inlineCount !== datasetCount)
    return undefined
  return inlineCount ?? datasetCount
}
