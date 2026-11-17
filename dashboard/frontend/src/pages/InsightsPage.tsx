import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { ListTree } from 'lucide-react'

import { IconTile, PageContainer, PageHeader, Tabs, TabsList, TabsTrigger } from '../ui'
import { formatInsightsCompactCount } from '../utils/insightsCost'

import InsightsRequestsView from './InsightsRequestsView'
import InsightsReviewsView from './InsightsReviewsView'
import InsightsRunsView from './InsightsRunsView'
import { InsightsDataProviders } from './insightsModelLabelSupport'
import { fetchAbortableInsightsJSON } from './insightsPageRequestSupport'
import type { InsightsListResponse } from './insightsPageTypes'
import { REVIEW_REQUEST_KIND } from './insightsReviewSupport'
import { INSIGHTS_TABS, readInsightsTab, type InsightsTab } from './insightsRequestsData'
import type { RunListResponse } from './insightsRunTypes'

const REPLAYS = '/api/router/api/v1/observability/replays'

// Every tab's total, unfiltered, for the tab labels; a count that cannot load is left off.
function useTabCounts() {
  const [counts, setCounts] = useState<Partial<Record<InsightsTab, number>>>({})
  useEffect(() => {
    const abort = new AbortController()
    const load = (tab: InsightsTab, url: string) =>
      fetchAbortableInsightsJSON<InsightsListResponse | RunListResponse>(url, tab, abort.signal)
        .then((list) =>
          setCounts((current) => ({
            ...current,
            [tab]: typeof list.total === 'number' ? list.total : list.count,
          })),
        )
        .catch(() => undefined)
    void load('requests', `${REPLAYS}?limit=1&offset=0&showDetails=false`)
    void load('runs', `${REPLAYS}/runs?limit=1&offset=0`)
    void load(
      'reviews',
      `${REPLAYS}?${new URLSearchParams({ request_kind: REVIEW_REQUEST_KIND, limit: '1', offset: '0', showDetails: 'false' })}`,
    )
    return () => abort.abort()
  }, [])
  return counts
}

// Requests: every routed request, the agent runs they make up, and the
// approval reviews among them, the tab and its filters kept in the URL.
export default function InsightsPage() {
  return (
    <InsightsDataProviders>
      <InsightsWorkspace />
    </InsightsDataProviders>
  )
}

function InsightsWorkspace() {
  const [params, setParams] = useSearchParams()
  const tab = readInsightsTab(params)
  const counts = useTabCounts()
  const session = params.get('session') ?? ''

  const openTab = (next: string) => {
    const nextParams = new URLSearchParams()
    if (next !== 'requests') nextParams.set('tab', next)
    // a session carries over, so a session's requests open its reviews
    if (session && next !== 'runs') nextParams.set('session', session)
    setParams(nextParams)
  }

  return (
    <PageContainer>
      <PageHeader
        icon={<IconTile icon={ListTree} tone="blue" size="lg" />}
        title="Requests"
        description="Every request the router handled: the client and agent that sent it, the decision it matched, the model that answered and what it cost."
      >
        <Tabs value={tab} onValueChange={openTab} variant="line" className="gap-0">
          <TabsList aria-label="Requests views">
            {INSIGHTS_TABS.map(({ value, label }) => (
              <TabsTrigger key={value} value={value}>
                {label}
                {counts[value] !== undefined ? (
                  <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] text-muted-foreground">
                    {formatInsightsCompactCount(counts[value] ?? 0)}
                  </span>
                ) : null}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </PageHeader>
      {tab === 'runs' ? (
        <InsightsRunsView />
      ) : tab === 'reviews' ? (
        <InsightsReviewsView key={session} initialSession={session} />
      ) : (
        <InsightsRequestsView />
      )}
    </PageContainer>
  )
}
