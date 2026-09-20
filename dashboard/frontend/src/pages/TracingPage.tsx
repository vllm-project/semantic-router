import EmbeddedServicePage from '../components/EmbeddedServicePage'
import type { ServiceConfig } from '../components/ServiceNotConfigured'
import { DOCS_LINKS } from '../utils/docsLinks'

const JAEGER_SERVICE: ServiceConfig = {
  name: 'Jaeger',
  envVar: 'TARGET_JAEGER_URL',
  description: 'Connect Jaeger to investigate request paths across the router and model backends.',
  docsUrl: DOCS_LINKS.observability,
  exampleValue: 'http://localhost:16686',
}

export default function TracingPage() {
  return (
    <EmbeddedServicePage
      eyebrow="Observability"
      title="Tracing"
      description="Choose a service and time range to follow recorded requests through routing and backend inference."
      service={JAEGER_SERVICE}
      availabilityUrl="/embedded/jaeger/"
      src="/embedded/jaeger/search"
      iframeTitle="Jaeger distributed tracing"
    />
  )
}
