import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { BenchmarkExplorer, ModelCards, ModelTable } from './ModelHubViews'
import { modelHubRows, type ModelHubFilters } from './modelHubSupport'

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

  it('wraps every model card in a list item', () => {
    const markup = renderToStaticMarkup(
      <ModelCards rows={rows.slice(0, 3)} selected={rows[0]} select={select} />,
    )

    expect(markup.match(/role="listitem"/g)).toHaveLength(3)
    expect(markup).toContain('<small>Benchmarks</small>')
  })

  it('announces benchmark rank, model, evaluation condition, and exact value', () => {
    const markup = renderToStaticMarkup(
      <BenchmarkExplorer catalog={catalog} rows={rows} selected={rows[0]} select={select} />,
    )

    expect(markup).toMatch(/aria-label="Rank 1, [^"]+, [^"]+, [^"]+"/)
  })

  it('labels an always-on reasoning result as evidence rather than a configurable effort', () => {
    const model = catalog.models.find((candidate) => candidate.id === 'ai21/jamba-reasoning-3b')!
    const focusedCatalog: BuiltInModelCatalog = {
      ...catalog,
      models: [model],
      evaluations: catalog.evaluations.filter((evaluation) => evaluation.model === model.id),
    }
    const focusedRows = modelHubRows(focusedCatalog, filters)
    const markup = renderToStaticMarkup(
      <BenchmarkExplorer
        catalog={focusedCatalog}
        rows={focusedRows}
        selected={focusedRows[0]}
        select={select}
      />,
    )

    expect(markup).toContain('<small>Reasoning enabled</small>')
    expect(markup).toMatch(/aria-label="Rank 1, Jamba Reasoning 3B, Reasoning enabled, [^"]+"/)
    expect(markup).not.toContain('enabled effort')
  })
})
