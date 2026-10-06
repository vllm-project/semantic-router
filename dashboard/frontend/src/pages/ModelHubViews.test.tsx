import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { ModelList, ModelTable } from './ModelHubViews'
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

  it('renders every model as a compact directory list item', () => {
    const markup = renderToStaticMarkup(
      <ModelList rows={rows.slice(0, 3)} selected={rows[0]} select={select} />,
    )

    expect(markup.match(/role="listitem"/g)).toHaveLength(3)
    expect(markup).toContain('<small>Results</small>')
  })
})
