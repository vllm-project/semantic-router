import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { DataTable } from './DataTable'

describe('DataTable row details', () => {
  it('makes the row the primary detail action without rendering a redundant action column', () => {
    const markup = renderToStaticMarkup(
      <DataTable<{ id: string; name: string }>
        columns={[{ key: 'name', header: 'Name' }]}
        data={[{ id: 'model-1', name: 'Model one' }]}
        keyExtractor={(row) => row.id}
        onView={vi.fn()}
        openOnRowClick
      />,
    )

    expect(markup).toContain('tabindex="0"')
    expect(markup).toContain('aria-label="Open model-1"')
    expect(markup).not.toContain('>Actions<')
    expect(markup).not.toContain('>View<')
  })
})
