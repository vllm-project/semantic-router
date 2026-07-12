import { useMemo } from 'react'

import EmbeddedServicePage from '../components/EmbeddedServicePage'
import type { ServiceConfig } from '../components/ServiceNotConfigured'
import { withAuthQuery } from '../utils/authFetch'

const JAEGER_SERVICE: ServiceConfig = {
  name: 'Jaeger',
  envVar: 'TARGET_JAEGER_URL',
  description: 'Connect Jaeger to investigate request paths across the router and model backends.',
  docsUrl: 'https://vllm-semantic-router.com/docs/tutorials/observability/dashboard',
  exampleValue: 'http://localhost:16686',
}

export default function TracingPage() {
  const src = useMemo(
    () => withAuthQuery('/embedded/jaeger/search?lookback=1h&limit=20&service=vllm-sr'),
    [],
  )

  return (
    <EmbeddedServicePage
      eyebrow="Observability"
      title="Tracing"
      description="Follow each request from captured signals through decisions, plugins, and backend inference."
      service={JAEGER_SERVICE}
      availabilityUrl="/embedded/jaeger/"
      src={src}
      iframeTitle="Jaeger distributed tracing"
    />
  )
}
