import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import EmbeddedServicePage from './EmbeddedServicePage'
import { DOCS_LINKS } from '../utils/docsLinks'

describe('embedded observability shell', () => {
  it('renders a consistent loading, recovery, and full-view contract', () => {
    const markup = renderToStaticMarkup(
      createElement(EmbeddedServicePage, {
        eyebrow: 'Observability',
        title: 'Monitoring',
        description: 'Inspect live routing metrics.',
        service: {
          name: 'Grafana',
          envVar: 'TARGET_GRAFANA_URL',
          description: 'Configure Grafana.',
        },
        availabilityUrl: '/embedded/grafana/',
        src: '/embedded/grafana/dashboard',
        iframeTitle: 'Grafana monitoring dashboard',
      }),
    )

    expect(markup).toContain('Monitoring')
    expect(markup).toContain('Checking connection')
    expect(markup).toContain('Open full view')
    expect(markup).toContain('Secure same-origin proxy')
  })

  it('keeps monitoring and tracing pages declarative and free of iframe theme mutation', () => {
    const sources = ['../pages/MonitoringPage.tsx', '../pages/TracingPage.tsx'].map((path) =>
      readFileSync(new URL(path, import.meta.url), 'utf8'),
    )

    for (const source of sources) {
      expect(source).toContain('<EmbeddedServicePage')
      expect(source).not.toContain('console.log')
      expect(source).not.toContain('contentWindow')
      expect(source).not.toContain("setItem('theme', 'light')")
    }
  })

  it('points the observability empty-state docs links at a live docs page', () => {
    // Regression guard for #3112: the Monitoring and Tracing empty states both
    // linked to https://vllm-sr.ai/docs/tutorials/observability/dashboard, which
    // only exists in archived v0.1/v0.2 docs and now returns 404.
    const DELETED_DOCS_PATH = '/tutorials/observability/dashboard'

    expect(DOCS_LINKS.observability).toMatch(/^https:\/\/vllm-sr\.ai\/docs\//)
    expect(DOCS_LINKS.observability).not.toContain(DELETED_DOCS_PATH)
    expect(DOCS_LINKS.observability).toContain('/tutorials/global/api-and-observability')

    const sources = ['../pages/MonitoringPage.tsx', '../pages/TracingPage.tsx'].map((path) =>
      readFileSync(new URL(path, import.meta.url), 'utf8'),
    )

    for (const source of sources) {
      // Both pages must resolve the link through the shared constant so they
      // cannot drift apart again.
      expect(source).toContain('DOCS_LINKS.observability')
      expect(source).not.toContain(DELETED_DOCS_PATH)
      expect(source).not.toContain('vllm-sr.ai/docs/tutorials/observability')
    }
  })
})
