import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { HubFilters } from './ModelHubFilters'
import type { ModelHubFilters } from './modelHubSupport'

const filters = (sort: ModelHubFilters['sort']): ModelHubFilters => ({
  query: '',
  kind: 'all',
  distribution: 'all',
  lifecycle: 'supported',
  publisher: 'all',
  provider: 'all',
  capability: 'all',
  sort,
})

describe('model hub filters', () => {
  it('treats a non-default sort as resettable state', () => {
    const sorted = filters('name')
    const markup = renderToStaticMarkup(
      <HubFilters
        filters={sorted}
        creators={[]}
        providers={[]}
        capabilities={[]}
        view="table"
        collapsed={false}
        update={() => undefined}
        setView={() => undefined}
        toggleCollapsed={() => undefined}
        reset={() => undefined}
      />,
    )

    expect(markup).toContain('Reset (1)')
    expect(markup).not.toMatch(/<button[^>]+disabled=""[^>]*>Reset/)
  })

  it('leaves reset disabled for the initial filter and sort state', () => {
    const initial = filters('released')
    const markup = renderToStaticMarkup(
      <HubFilters
        filters={initial}
        creators={[]}
        providers={[]}
        capabilities={[]}
        view="table"
        collapsed={false}
        update={() => undefined}
        setView={() => undefined}
        toggleCollapsed={() => undefined}
        reset={() => undefined}
      />,
    )

    expect(markup).toMatch(/<button[^>]+disabled=""[^>]*>Reset/)
  })

  it('exposes a labelled control that can collapse the filter rail', () => {
    const markup = renderToStaticMarkup(
      <HubFilters
        filters={filters('released')}
        creators={[]}
        providers={[]}
        capabilities={[]}
        view="list"
        collapsed
        update={() => undefined}
        setView={() => undefined}
        toggleCollapsed={() => undefined}
        reset={() => undefined}
      />,
    )

    expect(markup).toContain('aria-label="Show model filters"')
    expect(markup).toContain('aria-expanded="false"')
    expect(markup).toContain('aria-controls="model-hub-filter-controls"')
  })
})
