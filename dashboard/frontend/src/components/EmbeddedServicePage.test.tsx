import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import EmbeddedServicePage from './EmbeddedServicePage'

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
})
