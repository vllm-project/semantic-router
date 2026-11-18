import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import MonitoringPage from './MonitoringPage'

const theme = vi.hoisted(() => ({ resolved: 'dark' }))

vi.mock('../shell/theme', () => ({ useTheme: () => theme }))

vi.mock('../components/EmbeddedServicePage', () => ({
  default: ({
    src,
    iframeTitle,
    fullViewLabel,
  }: {
    src: string
    iframeTitle: string
    fullViewLabel: string
  }) => createElement('iframe', { src, title: iframeTitle, 'data-full-view': fullViewLabel }),
}))

const provisionedDashboard = JSON.parse(
  readFileSync(
    new URL(
      '../../../../src/vllm-sr/cli/templates/llm-router-dashboard.serve.json',
      import.meta.url,
    ),
    'utf8',
  ),
) as { uid: string }

function embeddedUrl(path = '/observe/metrics') {
  const markup = renderToStaticMarkup(
    createElement(MemoryRouter, { initialEntries: [path] }, createElement(MonitoringPage)),
  )
  const source = markup.match(/src="([^"]+)"/)?.[1].replace(/&amp;/g, '&')
  expect(source).toBeDefined()
  return new URL(source!, 'https://dashboard.example.test')
}

describe('Monitoring provisioned dashboard contract', () => {
  it.each(['dark', 'light'])('embeds the actual serve dashboard UID in %s mode', (resolved) => {
    theme.resolved = resolved
    const target = embeddedUrl()

    expect(provisionedDashboard.uid).toBeTruthy()
    expect(target.pathname).toBe(
      `/embedded/grafana/d/${encodeURIComponent(provisionedDashboard.uid)}`,
    )
    expect(target.searchParams.get('theme')).toBe(resolved)
    expect(target.searchParams.get('orgId')).toBe('1')
  })

  it('applies the time range from the URL and falls back to six hours', () => {
    theme.resolved = 'light'
    expect(embeddedUrl('/observe/metrics?range=24h').searchParams.get('from')).toBe('now-24h')
    expect(embeddedUrl('/observe/metrics?range=bogus').searchParams.get('from')).toBe('now-6h')
    expect(embeddedUrl().searchParams.get('to')).toBe('now')
  })
})
