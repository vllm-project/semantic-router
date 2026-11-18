import type { PreparationBenchmark, PreparationProfile } from './datasetPreparationApi'

export const PREPARATION_PROFILES: readonly PreparationProfile[] = ['smoke', 'quick', 'standard']

/** A longer catalog than this gets a search field above its list. */
export const BENCHMARK_SEARCH_THRESHOLD = 6

/** The benchmarks whose name or id contains the query, ignoring case. */
export function filterPreparationBenchmarks(
  benchmarks: PreparationBenchmark[],
  query: string,
): PreparationBenchmark[] {
  const needle = query.trim().toLowerCase()
  if (!needle) return benchmarks
  return benchmarks.filter((benchmark) =>
    `${benchmark.name} ${benchmark.id}`.toLowerCase().includes(needle),
  )
}

/** The question count of the largest size a benchmark can be prepared at. */
export function largestPreparationSize(benchmark: PreparationBenchmark): number {
  return Math.max(
    0,
    ...PREPARATION_PROFILES.map((profile) => {
      const count = benchmark.profiles[profile]
      return Number.isFinite(count) ? count : 0
    }),
  )
}
