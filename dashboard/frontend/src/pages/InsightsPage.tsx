import { useSearchParams } from 'react-router-dom'
import { ListTree } from 'lucide-react'

import { IconTile, PageContainer, PageHeader, Tabs, TabsList, TabsTrigger } from '../ui'

import InsightsRequestsView from './InsightsRequestsView'
import InsightsReviewsView from './InsightsReviewsView'
import InsightsRunsView from './InsightsRunsView'
import { InsightsDataProviders } from './insightsModelLabelSupport'
import { INSIGHTS_TABS, readInsightsTab } from './insightsRequestsData'

// Requests: every routed request, the agent runs they make up, and the
// approval reviews among them, the tab and its filters kept in the URL. Each
// tab loads its own data when it opens.
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
  const session = params.get('session') ?? ''
  const range = params.get('range')

  const openTab = (next: string) => {
    const nextParams = new URLSearchParams()
    if (next !== 'requests') nextParams.set('tab', next)
    // a session carries over, so a session's requests open its reviews
    if (session && next !== 'runs') nextParams.set('session', session)
    // the requests and their runs read one time range
    if (range && next !== 'reviews' && tab !== 'reviews') nextParams.set('range', range)
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
