import { useEffect, useState } from 'react'
import { ChartGantt } from 'lucide-react'

import EmbeddedServicePage from '../components/EmbeddedServicePage'
import type { ServiceConfig } from '../components/ServiceNotConfigured'
import { Badge } from '../ui'
import { DOCS_LINKS } from '../utils/docsLinks'
import { fetchDashboardJson } from './dashboardPageRequests'

const JAEGER_SERVICE: ServiceConfig = {
  name: 'Jaeger',
  envVar: 'TARGET_JAEGER_URL',
  description: 'Connect Jaeger to investigate request paths across the router and model backends.',
  docsUrl: DOCS_LINKS.observability,
  exampleValue: 'http://localhost:16686',
}

interface TracingConfig {
  enabled?: boolean
  resource?: { service_name?: string }
}

/** The Router's trace service name and switch, from `global.services.observability.tracing`. */
function useTracingConfig() {
  const [tracing, setTracing] = useState<TracingConfig | null>(null)
  useEffect(() => {
    let active = true
    fetchDashboardJson<{
      global?: { services?: { observability?: { tracing?: TracingConfig } } }
    }>('/api/router/config/all', 'Router config')
      .then((config) => {
        if (active) setTracing(config.global?.services?.observability?.tracing ?? {})
      })
      .catch(() => undefined)
    return () => {
      active = false
    }
  }, [])
  return tracing
}

export default function TracingPage() {
  const tracing = useTracingConfig()
  const service = tracing?.resource?.service_name?.trim()

  return (
    <EmbeddedServicePage
      title="Traces"
      description="Follow each request through Envoy, the Router and the backend models in Jaeger. Choose the service, time range and filters in the search form."
      icon={ChartGantt}
      tone="violet"
      service={JAEGER_SERVICE}
      availabilityUrl="/embedded/jaeger/"
      src="/embedded/jaeger/search"
      iframeTitle="Jaeger distributed tracing"
      fullViewLabel="Open in Jaeger"
      toolbar={
        tracing ? (
          <>
            {service ? (
              <span>
                Router service <code className="font-mono text-foreground/85">{service}</code>
              </span>
            ) : null}
            {tracing.enabled === false ? (
              <Badge tone="warning" size="sm">
                Tracing is off in the router config
              </Badge>
            ) : null}
          </>
        ) : null
      }
    />
  )
}
