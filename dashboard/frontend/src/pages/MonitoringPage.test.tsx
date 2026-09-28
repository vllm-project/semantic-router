import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'

import MonitoringPage from './MonitoringPage'

vi.mock('../components/EmbeddedServicePage', () => ({
  default: ({ src, iframeTitle }: { src: string; iframeTitle: string }) =>
    createElement('iframe', { src, title: iframeTitle }),
}))

const provisionedDashboard = JSON.parse(readFileSync(
  new URL('../../../../src/vllm-sr/cli/templates/llm-router-dashboard.serve.json', import.meta.url),
  'utf8',
)) as { uid: string }

afterEach(() => vi.unstubAllGlobals())

describe('Monitoring provisioned dashboard contract', () => {
  it.each(['dark', 'light'])('embeds the actual serve dashboard UID in %s mode', (theme) => {
    vi.stubGlobal('document', {
      documentElement: { getAttribute: () => theme },
    })

    const markup = renderToStaticMarkup(createElement(MonitoringPage))
    const source = markup.match(/src="([^"]+)"/)?.[1].replace(/&amp;/g, '&')
    expect(source).toBeDefined()
    const target = new URL(source!, 'https://dashboard.example.test')

    expect(provisionedDashboard.uid).toBeTruthy()
    expect(target.pathname).toBe(`/embedded/grafana/d/${encodeURIComponent(provisionedDashboard.uid)}`)
    expect(target.searchParams.get('theme')).toBe(theme)
    expect(target.searchParams.get('orgId')).toBe('1')
  })
})
