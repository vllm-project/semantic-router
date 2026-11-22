import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Pager } from './InsightsDisplay'
import {
  fetchReplayWindow,
  pageHolding,
  pageOf,
  readPage,
  readPageSize,
  writePage,
} from './insightsListData'

afterEach(() => vi.unstubAllGlobals())

const ids = (from: number, to: number) => Array.from({ length: to - from }, (_, i) => from + i)

describe('list paging', () => {
  it('reads the page and page size a URL asks for, and drops what it cannot use', () => {
    expect(readPage(new URLSearchParams('page=3'))).toBe(3)
    expect(readPage(new URLSearchParams('page=0'))).toBe(1)
    expect(readPage(new URLSearchParams('page=abc'))).toBe(1)
    expect(readPageSize(new URLSearchParams('size=100'))).toBe(100)
    expect(readPageSize(new URLSearchParams('size=1000'))).toBe(25)
    expect(readPageSize(new URLSearchParams())).toBe(25)
  })

  it('writes a page, and a new page size from the first page', () => {
    const params = new URLSearchParams('tab=runs&page=4&size=50')
    expect(writePage(params, 2).toString()).toBe('tab=runs&page=2&size=50')
    expect(writePage(params, 1).toString()).toBe('tab=runs&size=50')
    expect(writePage(params, 4, 100).toString()).toBe('tab=runs&size=100')
    expect(writePage(params, 4, 25).toString()).toBe('tab=runs')
  })

  it('pages rows in the browser and finds the page that holds a row', () => {
    const rows = ids(0, 60)
    expect(pageOf(rows, 3, 25)).toEqual({ rows: rows.slice(50), page: 3, pages: 3 })
    expect(pageOf(rows, 9, 25).page).toBe(3)
    expect(pageOf([], 2, 25)).toEqual({ rows: [], page: 1, pages: 1 })
    expect(pageHolding(49, 25)).toBe(2)
    expect(pageHolding(50, 25)).toBe(3)
    expect(pageHolding(-1, 25)).toBe(1)
  })

  it("states the rows shown of the API's total and offers page sizes when they matter", () => {
    const html = renderToStaticMarkup(
      <Pager
        page={2}
        pageSize={25}
        total={720_000}
        noun="request"
        onPage={() => undefined}
        onPageSize={() => undefined}
      />,
    )
    expect(html).toContain('26–50 of 720,000 requests')
    expect(html).toContain('Page 2 of 28,800')
    expect(html).toContain('aria-label="Rows per page"')
    const small = renderToStaticMarkup(
      <Pager
        page={1}
        pageSize={25}
        total={12}
        noun="call"
        onPage={() => undefined}
        onPageSize={() => undefined}
      />,
    )
    expect(small).toContain('1–12 of 12 calls')
    expect(small).not.toContain('Rows per page')
    expect(small).not.toContain('Next page')
  })
})

describe('the window of newest rows', () => {
  const page = (rows: number[], total: number, more: boolean) =>
    new Response(
      JSON.stringify({ data: rows.map((id) => ({ id: `r${id}` })), total, has_more: more }),
    )
  const read = (size: number) =>
    fetchReplayWindow(
      (limit, offset) => `/rows?limit=${limit}&offset=${offset}`,
      'rows',
      size,
      (row: { id: string }) => row.id,
      new AbortController().signal,
    )

  it('reads the first page, then the rest at once, keeping each row once', async () => {
    // a row written while paging pushes r99 onto the second page
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(page(ids(0, 100), 250, true))
      .mockResolvedValueOnce(page([99, ...ids(100, 199)], 250, true))
      .mockResolvedValueOnce(page(ids(199, 250), 250, false))
    vi.stubGlobal('fetch', fetch)
    const { rows, total } = await read(300)
    expect(fetch.mock.calls.map(([url]) => url)).toEqual([
      '/rows?limit=100&offset=0',
      '/rows?limit=100&offset=100',
      '/rows?limit=100&offset=200',
    ])
    expect(total).toBe(250)
    expect(rows).toHaveLength(250)
  })

  it('stops at one page that holds every row, and at the size asked for', async () => {
    const fetch = vi.fn().mockResolvedValueOnce(page(ids(0, 40), 40, false))
    vi.stubGlobal('fetch', fetch)
    expect((await read(300)).rows).toHaveLength(40)
    expect(fetch).toHaveBeenCalledTimes(1)

    const capped = vi
      .fn()
      .mockResolvedValueOnce(page(ids(0, 100), 1000, true))
      .mockResolvedValueOnce(page(ids(100, 150), 1000, true))
    vi.stubGlobal('fetch', capped)
    expect((await read(150)).rows).toHaveLength(150)
    expect(capped.mock.calls.map(([url]) => url)).toEqual([
      '/rows?limit=100&offset=0',
      '/rows?limit=50&offset=100',
    ])
  })
})
