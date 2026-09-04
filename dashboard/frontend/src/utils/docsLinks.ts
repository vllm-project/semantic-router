// Single source of truth for external documentation links surfaced in the
// dashboard UI. Centralising these keeps "View Documentation" actions from
// drifting to renamed or deleted docs pages.

const DOCS_BASE_URL = 'https://vllm-sr.ai/docs'

export const DOCS_LINKS = {
  // Metrics (Grafana) and tracing (Jaeger) setup both live in the observability
  // section of the "API and Observability" tutorial.
  observability: `${DOCS_BASE_URL}/tutorials/global/api-and-observability#observability`,
} as const
