import { describe, expect, it } from 'vitest'

import { getPageWindow, paginateRows, updatePageSelection } from './dataTableSupport'

describe('data table scale support', () => {
  it('paginates a 500-row inventory without rendering the full collection', () => {
    const rows = Array.from({ length: 500 }, (_, index) => `model-${index + 1}`)
    const window = getPageWindow(rows.length, 8, 50)

    expect(window).toEqual({ page: 8, pageSize: 50, totalPages: 10, start: 350, end: 400 })
    expect(paginateRows(rows, window)).toHaveLength(50)
    expect(paginateRows(rows, window)[0]).toBe('model-351')
  })

  it('clamps stale pages after filtering shrinks the inventory', () => {
    expect(getPageWindow(7, 20, 25)).toEqual({
      page: 1,
      pageSize: 25,
      totalPages: 1,
      start: 0,
      end: 7,
    })
  })

  it('selects and clears only the visible page while preserving other pages', () => {
    const selected = new Set(['model-1', 'model-90'])
    const pageKeys = ['model-26', 'model-27']
    const withPage = updatePageSelection(selected, pageKeys, true)

    expect([...withPage].sort()).toEqual(['model-1', 'model-26', 'model-27', 'model-90'])
    expect([...updatePageSelection(withPage, pageKeys, false)].sort()).toEqual(['model-1', 'model-90'])
  })
})
