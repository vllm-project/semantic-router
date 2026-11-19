import { createElement, type ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '../ui'

import { codexSubagentCall } from './insightsAgentFixtures'
import InsightsRequestsView from './InsightsRequestsView'
import type { RequestsData, useRequestsData } from './insightsRequestsData'

const requests = vi.hoisted(() => ({
  data: null as RequestsData | null,
  options: null as Parameters<typeof useRequestsData>[0] | null,
}))

// Static markup cannot open a popover: render every filter's options in place.
vi.mock('radix-ui', async (importOriginal) => {
  const actual = await importOriginal<typeof import('radix-ui')>()
  return {
    ...actual,
    Popover: {
      ...actual.Popover,
      Root: (props: { children?: ReactNode }) =>
        createElement(actual.Popover.Root, { ...props, open: true }),
      Portal: ({ children }: { children?: ReactNode }) => children,
    },
  }
})

vi.mock('./insightsRequestsData', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./insightsRequestsData')>()),
  useRequestsData: (options: Parameters<typeof useRequestsData>[0]) => {
    requests.options = options
    return requests.data
  },
}))

function requestsData(fields: Partial<RequestsData>): RequestsData {
  return {
    rows: [],
    total: 0,
    aggregate: null,
    harnessCounts: null,
    window: [],
    windowTotal: 0,
    windowLoaded: false,
    windowLoading: false,
    loading: false,
    stale: false,
    aggregateLoading: false,
    error: null,
    listError: null,
    aggregateError: null,
    refresh: () => undefined,
    ...fields,
  }
}

const render = (entry: string) =>
  renderToStaticMarkup(
    <MemoryRouter initialEntries={[entry]}>
      <TooltipProvider>
        <InsightsRequestsView />
      </TooltipProvider>
    </MemoryRouter>,
  )

// An option of a filter, with the count it shows.
const option = (label: string, count: number) =>
  new RegExp(`>${label}</span><span class="font-mono[^"]*">${count}</span>`)

describe('the requests list', () => {
  beforeEach(() => {
    requests.data = null
    requests.options = null
  })

  it("filters by harness on the server, and pages and counts with the server's numbers", () => {
    requests.data = requestsData({
      rows: [codexSubagentCall],
      total: 977,
      harnessCounts: { api_client: 51234, claude_code: 1820, codex: 977 },
    })
    const html = render('/observe/requests?harness=codex')

    expect(requests.options?.filters.harness).toEqual(['codex'])
    expect(html).toContain('1–25 of 977 requests')
    expect(html).toMatch(option('Claude Code', 1820))
    expect(html).toMatch(option('Codex', 977))
    expect(html).toMatch(option('API client', 51234))
    // a harness without requests under the other filters is not offered
    expect(html).not.toContain('>OpenCode</span>')
    expect(html).not.toContain('in your browser')
  })

  it('reads the totals of the Sessions view over the requests that belong to a session', () => {
    requests.data = requestsData({ rows: [codexSubagentCall], total: 1 })
    render('/observe/requests?view=sessions')
    expect(requests.options?.sessionsOnly).toBe(true)
    render('/observe/requests')
    expect(requests.options?.sessionsOnly).toBe(false)
  })

  it('offers every harness before the counts arrive, and names the class filters the browser runs', () => {
    requests.data = requestsData({ rows: [codexSubagentCall], total: 1 })
    const html = render('/observe/requests?harness=api&class=main')

    expect(requests.options?.filters.harness).toEqual(['api_client'])
    for (const label of ['Claude Code', 'Codex', 'OpenCode', 'DeepSeek Harness', 'API client']) {
      expect(html).toContain(`>${label}</span>`)
    }
    expect(html).toContain(
      'Class filters run in your browser over the newest 300 requests in the last 24 hours.',
    )
  })
})
