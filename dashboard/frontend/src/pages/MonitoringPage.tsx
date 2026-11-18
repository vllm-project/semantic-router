import { useSearchParams } from 'react-router-dom'
import { ChartLine } from 'lucide-react'

import EmbeddedServicePage from '../components/EmbeddedServicePage'
import type { ServiceConfig } from '../components/ServiceNotConfigured'
import { useTheme } from '../shell/theme'
import { Segmented } from '../ui'
import { DOCS_LINKS } from '../utils/docsLinks'

const GRAFANA_SERVICE: ServiceConfig = {
  name: 'Grafana',
  envVar: 'TARGET_GRAFANA_URL',
  description:
    'Connect Grafana to inspect routing health, latency, throughput, and model activity.',
  docsUrl: DOCS_LINKS.observability,
  exampleValue: 'http://localhost:3000',
}

const DASHBOARD_UID = 'vllm-semantic-router'
const RANGES = ['15m', '1h', '6h', '24h', '7d'] as const
type Range = (typeof RANGES)[number]
const DEFAULT_RANGE: Range = '6h'

function grafanaDashboardUrl(theme: string, range: Range) {
  const query = new URLSearchParams({
    orgId: '1',
    theme,
    refresh: '30s',
    from: `now-${range}`,
    to: 'now',
  })
  return `/embedded/grafana/d/${encodeURIComponent(DASHBOARD_UID)}?${query}`
}

export default function MonitoringPage() {
  const { resolved } = useTheme()
  const [params, setParams] = useSearchParams()
  const requested = params.get('range')
  const range = RANGES.find((value) => value === requested) ?? DEFAULT_RANGE

  return (
    <EmbeddedServicePage
      title="Metrics"
      description="Throughput, latency, errors and cache efficiency from the router's Prometheus metrics, in the provisioned Grafana dashboard."
      icon={ChartLine}
      tone="blue"
      service={GRAFANA_SERVICE}
      availabilityUrl="/embedded/grafana/"
      src={grafanaDashboardUrl(resolved, range)}
      iframeTitle="Grafana monitoring dashboard"
      fullViewLabel="Open in Grafana"
      toolbar={
        <Segmented
          aria-label="Time range"
          size="sm"
          value={range}
          onValueChange={(value) =>
            setParams(
              (current) => {
                const next = new URLSearchParams(current)
                if (value === DEFAULT_RANGE) next.delete('range')
                else next.set('range', value)
                return next
              },
              { replace: true },
            )
          }
          options={RANGES.map((value) => ({ value, label: value }))}
        />
      }
    />
  )
}
