import type {
  CatalogIndex,
  CatalogIndexResult,
  CatalogModel,
  CatalogSnapshot,
} from './modelHubCatalogTypes'
import type { ModelHubArenaScope } from './modelHubUrlState'

export interface ModelHubArenaRow {
  model: CatalogModel
  result: CatalogIndexResult
  rank: number | null
  missingBenchmarks: string[]
}

export interface ModelHubArenaData {
  index: CatalogIndex
  ranked: ModelHubArenaRow[]
  awaitingEvidence: ModelHubArenaRow[]
}

const matchesScope = (model: CatalogModel, scope: ModelHubArenaScope): boolean => {
  if (scope === 'open') return model.distribution.type === 'open_weights'
  if (scope === 'virtual') return model.kind === 'virtual'
  return true
}

const availableComponentCount = (result: CatalogIndexResult): number =>
  result.components.filter(component => component.status === 'available').length

const effortPosition = (catalog: CatalogSnapshot, model: CatalogModel, effort: string): number => {
  const family = catalog.reasoning_families.find(candidate => candidate.id === model.reasoning_family)
  const efforts = family?.levels ?? family?.modes ?? ['default']
  const position = efforts.indexOf(effort)
  return position === -1 ? -1 : position
}

function preferredArenaResult(
  catalog: CatalogSnapshot,
  model: CatalogModel,
  results: CatalogIndexResult[],
): CatalogIndexResult | undefined {
  return [...results].sort((left, right) => {
    const leftAvailable = left.status === 'available' && left.score !== null
    const rightAvailable = right.status === 'available' && right.score !== null
    if (leftAvailable !== rightAvailable) return leftAvailable ? -1 : 1
    const effortDifference
      = effortPosition(catalog, model, right.reasoning_effort)
        - effortPosition(catalog, model, left.reasoning_effort)
    if (leftAvailable && rightAvailable) {
      return (
        effortDifference
        || (right.score ?? 0) - (left.score ?? 0)
        || left.reasoning_effort.localeCompare(right.reasoning_effort)
      )
    }
    return (
      right.coverage - left.coverage
      || availableComponentCount(right) - availableComponentCount(left)
      || effortDifference
      || left.reasoning_effort.localeCompare(right.reasoning_effort)
    )
  })[0]
}

export function modelHubArenaData(
  catalog: CatalogSnapshot,
  scope: ModelHubArenaScope,
): ModelHubArenaData | null {
  const indexID = catalog.catalogs?.[0]?.default_intelligence_index
  const index = catalog.indices.find(candidate => candidate.id === indexID)
  if (!index) return null
  const benchmarkNames = new Map(
    catalog.benchmarks.map(benchmark => [benchmark.id, benchmark.display_name]),
  )
  const resultsByModel = new Map<string, CatalogIndexResult[]>()
  catalog.index_results
    .filter(result => result.index === index.id)
    .forEach((result) => {
      resultsByModel.set(result.model, [...(resultsByModel.get(result.model) ?? []), result])
    })
  const rows = catalog.models
    .filter(model => matchesScope(model, scope))
    .flatMap<ModelHubArenaRow>((model) => {
      const result = preferredArenaResult(catalog, model, resultsByModel.get(model.id) ?? [])
      if (!result) return []
      return [{
        model,
        result,
        rank: null,
        missingBenchmarks: result.components
          .filter(component => component.status !== 'available' && component.benchmark)
          .map(component => benchmarkNames.get(component.benchmark ?? '') ?? component.benchmark ?? '')
          .filter(Boolean),
      }]
    })
  const ranked = rows
    .filter(row => row.result.status === 'available' && row.result.score !== null)
    .sort((left, right) =>
      (right.result.score ?? 0) - (left.result.score ?? 0)
      || left.model.display_name.localeCompare(right.model.display_name),
    )
  let previousScore: number | null = null
  let previousRank = 0
  ranked.forEach((row, indexPosition) => {
    const score = row.result.score ?? 0
    if (previousScore === null || score !== previousScore) previousRank = indexPosition + 1
    row.rank = previousRank
    previousScore = score
  })
  const awaitingEvidence = rows
    .filter(row => row.result.status !== 'available')
    .sort((left, right) =>
      right.result.coverage - left.result.coverage
      || left.model.display_name.localeCompare(right.model.display_name),
    )
  return { index, ranked, awaitingEvidence }
}
