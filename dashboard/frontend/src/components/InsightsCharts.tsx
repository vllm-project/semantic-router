import { Bot, Cpu, Route } from 'lucide-react'

import type { InsightsAggregateResponse } from '../pages/insightsPageTypes'
import { modelProviderId, useInsightsConfig } from '../pages/insightsConfigFacts'
import { DecisionIcon } from '../pages/InsightsDecisionBadge'
import { HarnessLogo } from '../pages/InsightsHarness'
import { API_CLIENT_LABEL, harnessInfo, sortHarnessIds } from '../pages/insightsHarness'
import { modelLabel, useModelLabels } from '../pages/insightsModelLabelSupport'
import { formatInsightsDecisionName } from '../pages/insightsPageSupport'
import { ProviderLogo } from '../ui'
import InsightsRankedBars, { type InsightsRankedEntry } from './InsightsRankedBars'

interface InsightsChartsProps {
  aggregate: InsightsAggregateResponse
  /** Requests by harness under the page's other filters, from the records list. */
  harnessCounts?: Record<string, number> | null
}

const API_CLIENT = 'api_client'

const plural = (count: number, word: string) =>
  `${count.toLocaleString('en-US')} ${word}${count === 1 ? '' : 's'}`

const byValue = (a: InsightsRankedEntry, b: InsightsRankedEntry) =>
  b.value - a.value || a.label.localeCompare(b.label)

/** The harnesses by their requests, the plain API client last. */
function harnessEntries(counts: Record<string, number>): InsightsRankedEntry[] {
  const harnesses = sortHarnessIds(
    Object.keys(counts).filter((id) => id !== API_CLIENT && counts[id] > 0),
  )
    .map((id) => ({
      id,
      label: harnessInfo(id)?.label ?? id,
      value: counts[id],
      mark: <HarnessLogo harness={id} size={16} />,
    }))
    .sort(byValue)
  if (counts[API_CLIENT] > 0) {
    harnesses.push({
      id: API_CLIENT,
      label: API_CLIENT_LABEL,
      value: counts[API_CLIENT],
      mark: <HarnessLogo size={16} />,
    })
  }
  return harnesses
}

// The Breakdown: the requests by agent harness, model and decision, as three
// cards of one family. The summary row above holds the headline figures.
export default function InsightsCharts({ aggregate, harnessCounts }: InsightsChartsProps) {
  const labels = useModelLabels()
  const { modelProviders } = useInsightsConfig()
  if (aggregate.record_count === 0) {
    return null
  }
  const agents = harnessCounts ? harnessEntries(harnessCounts) : null
  const models = aggregate.model_selection
    .map((entry) => {
      const provider = modelProviderId(entry.name, modelProviders)
      const label = entry.name ? modelLabel(labels, entry.name) : 'No model call'
      return {
        id: entry.name || 'none',
        label,
        title: label === entry.name ? entry.name : `${label} · configured as ${entry.name}`,
        value: entry.value,
        mark: provider ? (
          <ProviderLogo provider={provider} size={16} />
        ) : (
          <span className="size-2.5 rounded-[3px] bg-border-strong" />
        ),
      }
    })
    .sort(byValue)
  const decisions = aggregate.decision_distribution
    .map((entry) => ({
      id: entry.name,
      label: formatInsightsDecisionName(entry.name),
      title: entry.name,
      value: entry.value,
      mark: <DecisionIcon decision={entry.name} className="size-3.5 text-chart-3" />,
    }))
    .sort(byValue)
  return (
    <section className="grid grid-cols-1 gap-3 lg:grid-cols-3" aria-label="Breakdown">
      <InsightsRankedBars
        title="Agents"
        icon={<Bot className="text-chart-1" />}
        color="var(--chart-1)"
        label="Requests by agent harness"
        rowKey="harness"
        entries={agents}
        meta={agents ? plural(agents.length, 'agent') : undefined}
      />
      <InsightsRankedBars
        title="Models"
        icon={<Cpu className="text-chart-6" />}
        color="var(--chart-6)"
        label="Requests by model"
        rowKey="model"
        entries={models}
        total={aggregate.record_count}
        meta={plural(models.filter((entry) => entry.id !== 'none').length, 'model')}
      />
      <InsightsRankedBars
        title="Decisions"
        icon={<Route className="text-chart-3" />}
        color="var(--chart-3)"
        label="Requests by decision"
        rowKey="decision"
        entries={decisions}
        total={aggregate.record_count}
        meta={plural(decisions.length, 'decision')}
      />
    </section>
  )
}
