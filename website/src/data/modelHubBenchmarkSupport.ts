export interface ModelHubBenchmarkSelection {
  benchmark: string
  profile: string
  metric: string
}

interface ModelHubEvaluationSelectionSource {
  status: string
  benchmark: string
  benchmark_profile: string
  metrics?: Record<string, number | null>
}

const selectionKey = (
  benchmark: string,
  profile: string,
  metric: string,
): string => `${benchmark}\u0000${profile}\u0000${metric}`

export function modelHubBenchmarkSelectionCounts(
  evaluations: ModelHubEvaluationSelectionSource[],
): Map<string, number> {
  const counts = new Map<string, number>()
  evaluations.forEach((evaluation) => {
    if (evaluation.status !== 'available') return
    Object.entries(evaluation.metrics ?? {}).forEach(([metric, value]) => {
      if (typeof value !== 'number') return
      const key = selectionKey(
        evaluation.benchmark,
        evaluation.benchmark_profile,
        metric,
      )
      counts.set(key, (counts.get(key) ?? 0) + 1)
    })
  })
  return counts
}

export function preferredModelHubBenchmarkSelection(
  counts: Map<string, number>,
  benchmarkID?: string,
  profileID?: string,
): ModelHubBenchmarkSelection | undefined {
  const prefix = [benchmarkID, profileID]
    .filter(value => value !== undefined)
    .join('\u0000')
  const scopedPrefix = prefix ? `${prefix}\u0000` : ''
  const key = Array.from(counts.entries())
    .filter(([candidate]) => candidate.startsWith(scopedPrefix))
    .sort(
      (left, right) => right[1] - left[1] || left[0].localeCompare(right[0]),
    )[0]?.[0]
  if (!key) return undefined
  const [benchmark, profile, metric] = key.split('\u0000')
  return { benchmark, profile, metric }
}

export function availableModelHubBenchmarkProfiles(
  counts: Map<string, number>,
  benchmarkID: string,
): Set<string> {
  const profiles = new Set<string>()
  const prefix = `${benchmarkID}\u0000`
  counts.forEach((_count, key) => {
    if (!key.startsWith(prefix)) return
    profiles.add(key.split('\u0000')[1])
  })
  return profiles
}

export function availableModelHubBenchmarkMetrics(
  counts: Map<string, number>,
  benchmarkID: string,
  profileID: string,
): Set<string> {
  const metrics = new Set<string>()
  const prefix = `${benchmarkID}\u0000${profileID}\u0000`
  counts.forEach((_count, key) => {
    if (!key.startsWith(prefix)) return
    metrics.add(key.split('\u0000')[2])
  })
  return metrics
}

const modelHubChartColorSlots = 3600
const modelHubChartColorProbe = 137

const stableModelHubHash = (value: string): number => {
  let hash = 2166136261
  for (const character of value) {
    hash ^= character.charCodeAt(0)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

export function modelHubChartHues(modelIDs: string[]): Map<string, number> {
  const hues = new Map<string, number>()
  const used = new Set<number>()
  const uniqueIDs = [...new Set(modelIDs)].sort((left, right) =>
    left.localeCompare(right),
  )

  uniqueIDs.forEach((id) => {
    let slot = stableModelHubHash(id) % modelHubChartColorSlots
    while (used.has(slot))
      slot = (slot + modelHubChartColorProbe) % modelHubChartColorSlots
    used.add(slot)
    hues.set(id, slot / 10)
  })
  return hues
}
