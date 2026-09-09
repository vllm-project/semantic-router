import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { BenchmarkExplorer, ModelList, ModelTable } from './ModelHubViews'
import {
  modelHubBenchmarkBarHeight,
  modelHubBenchmarkOverviewSelections,
  modelHubBenchmarkPoints,
  modelHubRows,
  type ModelHubFilters,
} from './modelHubSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog
const filters: ModelHubFilters = {
  query: '',
  kind: 'physical',
  distribution: 'all',
  lifecycle: 'supported',
  publisher: 'all',
  provider: 'all',
  capability: 'all',
  sort: 'name',
}
const rows = modelHubRows(catalog, filters)
const select = (): void => undefined

describe('model hub views', () => {
  it('renders a semantic results table with the model action inside its first cell', () => {
    const markup = renderToStaticMarkup(
      <ModelTable rows={rows.slice(0, 3)} selected={rows[0]} select={select} />,
    )

    expect(markup).toContain('<table')
    expect(markup).toContain('<th scope="col">Model</th>')
    expect(markup).toContain('Swipe horizontally to compare every column')
    expect(markup).toContain('aria-describedby="model-hub-table-scroll-hint"')
    expect(markup).toMatch(/<td><button[^>]+aria-label="Inspect /)
    expect(markup).not.toContain('role="cell"')
  })

  it('renders every model as a compact directory list item', () => {
    const markup = renderToStaticMarkup(
      <ModelList rows={rows.slice(0, 3)} selected={rows[0]} select={select} />,
    )

    expect(markup.match(/role="listitem"/g)).toHaveLength(3)
    expect(markup).toContain('<small>Results</small>')
  })

  it('announces model, evaluation condition, and exact value without an overall rank', () => {
    const markup = renderToStaticMarkup(
      <BenchmarkExplorer catalog={catalog} rows={rows} openModel={select} />,
    )

    expect(markup).toMatch(/aria-label="[^"]+, [^"]+, [^"]+\. Open model details\."/)
    expect(markup).not.toContain('Rank 1')
    expect(markup).toMatch(/aria-label="[^"]+ comparison with all filtered results"/)
    expect(markup).not.toContain('aria-label="Model catalog pagination"')
    expect(markup).toContain('aria-label="Filter benchmarks"')
    expect(markup).toContain('aria-pressed="true"')
    expect(markup).not.toContain('<select')
    expect(markup).not.toContain('Source ↗')
    expect(markup.indexOf('Core <span>6</span>')).toBeGreaterThan(markup.indexOf('All <span>'))
    expect(markup).toMatch(/\d+\.\d%/)
    const modelIDs = new Set(rows.map((row) => row.model.id))
    const expected = modelHubBenchmarkOverviewSelections(catalog).reduce(
      (total, selection) => total + modelHubBenchmarkPoints(catalog, selection, modelIDs).length,
      0,
    )
    expect(markup.match(/role="listitem"/g)).toHaveLength(expected)
  })

  it('labels an always-on reasoning result as evidence rather than a configurable effort', () => {
    const model = catalog.models.find((candidate) => candidate.id === 'ai21/jamba-reasoning-3b')!
    const models = Array.from({ length: 10 }, (_, index) => ({
      ...model,
      id: `${model.id}-fixture-${index}`,
    }))
    const focusedCatalog: BuiltInModelCatalog = {
      ...catalog,
      models,
      evaluations: models.flatMap((fixtureModel, index) =>
        catalog.evaluations
          .filter((evaluation) => evaluation.model === model.id)
          .map((evaluation) => ({
            ...evaluation,
            id: `${evaluation.id}-fixture-${index}`,
            model: fixtureModel.id,
          })),
      ),
    }
    const focusedRows = modelHubRows(focusedCatalog, filters)
    const markup = renderToStaticMarkup(
      <BenchmarkExplorer catalog={focusedCatalog} rows={focusedRows} openModel={select} />,
    )

    expect(markup).toContain('<small>Reasoning enabled</small>')
    expect(markup).toMatch(
      /aria-label="Jamba Reasoning 3B, Reasoning enabled, [^"]+\. Open model details\."/,
    )
    expect(markup).not.toContain('enabled effort')
  })

  it('renders lower-is-better results with a taller bar for the better value', () => {
    const faster = modelHubBenchmarkBarHeight(1, 1, 10, 'lower_is_better')
    const slower = modelHubBenchmarkBarHeight(10, 1, 10, 'lower_is_better')

    expect(faster).toBeGreaterThan(slower)
  })
})
