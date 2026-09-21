import type { Dataset, DatasetPreparation } from './types'

const benchmarkTitles: Record<string, string> = {
  'arc-agi': 'ARC-AGI',
  'arc-agi-2': 'ARC-AGI 2',
  'gpqa-diamond': 'GPQA Diamond',
  hle: 'Humanity’s Last Exam',
  livecodebench: 'LiveCodeBench',
  'mmlu-pro': 'MMLU-Pro',
  scicode: 'SciCode',
  'simpleqa-verified': 'SimpleQA Verified',
  tau3: 'τ³-bench',
  'terminal-bench': 'Terminal-Bench',
  'terminal-bench-2.1': 'Terminal-Bench 2.1',
}

export const benchmarkTitle = (id: string) => benchmarkTitles[id] ?? id.replace(/[-_]/g, ' ')
export const profileTitle = (id?: string) =>
  ({ smoke: 'Smoke', quick: 'Quick', standard: 'Standard' })[id ?? ''] ?? 'Custom'
export const profileDescription = (id?: string) =>
  ({
    smoke: 'Check the pipeline with a small case set.',
    quick: 'Compare changes in a shorter evaluation loop.',
    standard: 'Evaluate broader capability coverage.',
  })[id ?? ''] ?? 'A prepared case set for your evaluation.'

export const evaluationRoleTitle = (role: 'holdout' | 'retest' | null) =>
  role === 'holdout' ? 'Holdout' : role === 'retest' ? 'Retest' : 'Role unspecified'

export function datasetEvaluationTitle(
  preparation?: DatasetPreparation,
  benchmarks?: string[],
): string | null {
  if (!preparation || !Object.keys(preparation).length) return null
  const families = new Set([...Object.keys(preparation), ...(benchmarks ?? [])])
  const roles = new Set(
    [...families].map((family) =>
      evaluationRoleTitle(preparation[family]?.evaluation_role ?? null),
    ),
  )
  return roles.size > 1 ? 'Mixed evaluation roles' : ([...roles][0] ?? null)
}

export function friendlyDatasetName(dataset: Pick<Dataset, 'name' | 'benchmarks'>): string {
  const raw = dataset.name?.trim()
  if (
    raw &&
    !raw.includes('+') &&
    !/^[a-f0-9]{32,}$/i.test(raw) &&
    !/\/(smoke|quick|standard)$/.test(raw)
  ) {
    return raw
  }
  const benchmarks = dataset.benchmarks ?? []
  if (benchmarks.length === 1) return benchmarkTitle(benchmarks[0])
  if (benchmarks.length > 1) return 'Capability suite'
  return 'Custom dataset'
}
