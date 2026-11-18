import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { DataTable, type Column } from './data-table'

interface Row {
  id: string
  name: string
}

const rows: Row[] = [
  { id: 'b', name: 'Bea' },
  { id: 'a', name: 'Ada' },
]

describe('DataTable sorting', () => {
  it('renders rows in the caller order when the caller owns the sort', () => {
    const columns: Column<Row>[] = [
      { id: 'name', header: 'Name', sortable: true, cell: (row) => row.name },
      { id: 'id', header: 'ID', cell: (row) => row.id },
    ]
    const markup = renderToStaticMarkup(
      <DataTable
        data={rows}
        columns={columns}
        getRowId={(row) => row.id}
        sort={{ id: 'name', dir: 'asc' }}
        onSortChange={() => undefined}
      />,
    )
    expect(markup).toContain('aria-sort="ascending"')
    expect(markup.indexOf('Bea')).toBeLessThan(markup.indexOf('Ada'))
    expect(markup.match(/<button type="button"/g)).toHaveLength(1)
  })

  it('keeps sortable-only columns static when the table sorts itself', () => {
    const markup = renderToStaticMarkup(
      <DataTable
        data={rows}
        columns={[{ id: 'name', header: 'Name', sortable: true, cell: (row) => row.name }]}
        getRowId={(row) => row.id}
      />,
    )
    expect(markup).not.toContain('<button type="button"')
    expect(markup).not.toContain('aria-sort')
  })
})
