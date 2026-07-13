import { useEffect, useMemo, useState } from 'react'

import EmbeddedServicePage from '../components/EmbeddedServicePage'
import type { ServiceConfig } from '../components/ServiceNotConfigured'
import { withAuthQuery } from '../utils/authFetch'

const GRAFANA_SERVICE: ServiceConfig = {
  name: 'Grafana',
  envVar: 'TARGET_GRAFANA_URL',
  description:
    'Connect Grafana to inspect routing health, latency, throughput, and model activity.',
  docsUrl: 'https://vllm-sr.ai/docs/tutorials/observability/dashboard',
  exampleValue: 'http://localhost:3000',
}

export default function MonitoringPage() {
  const [theme, setTheme] = useState(
    () => document.documentElement.getAttribute('data-theme') || 'dark',
  )

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setTheme(document.documentElement.getAttribute('data-theme') || 'dark')
    })
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    })
    return () => observer.disconnect()
  }, [])

  const src = useMemo(
    () =>
      withAuthQuery(`/embedded/grafana/goto/llm-router-metrics?orgId=1&theme=${theme}&refresh=30s`),
    [theme],
  )

  return (
    <EmbeddedServicePage
      eyebrow="Observability"
      title="Monitoring"
      description="Inspect the live metrics that shape routing quality, latency, and model readiness."
      service={GRAFANA_SERVICE}
      availabilityUrl="/embedded/grafana/"
      src={src}
      iframeTitle="Grafana monitoring dashboard"
    />
  )
}
