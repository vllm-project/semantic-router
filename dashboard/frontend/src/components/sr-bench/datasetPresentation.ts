import type { Dataset, DatasetPreparation, DatasetSelection } from './types'

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

/** The catalog's title when it is more than the id itself, else the known display name. */
export const catalogBenchmarkTitle = (id: string, catalog?: { id: string; title?: string }[]) => {
  const title = catalog?.find((benchmark) => benchmark.id === id)?.title
  return title && title !== id ? title : benchmarkTitle(id)
}
/**
 * Questions per target for the selected benchmarks, or null while any of them has no
 * prepared count yet (a benchmark that is prepared at review time reports case_count 0).
 */
export function selectedQuestionCount(
  selection: Pick<DatasetSelection, 'benchmarks'>,
  benchmarks: string[],
): number | null {
  if (!benchmarks.length) return null
  let total = 0
  for (const id of benchmarks) {
    const entry = selection.benchmarks.find((item) => item.id === id)
    if (!entry?.eligible || entry.case_count <= 0) return null
    total += entry.case_count
  }
  return total
}

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
