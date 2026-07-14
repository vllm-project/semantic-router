import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import { DashboardMiniFlowDiagram } from './DashboardMiniFlowDiagram'

describe('DashboardMiniFlowDiagram', () => {
  it('uses a browser-compatible SVG marker orientation', () => {
    const markup = renderToStaticMarkup(createElement(DashboardMiniFlowDiagram, {
      signals: { total: 1, byType: { keywords: 1 } },
      decisions: 1,
      models: 1,
      plugins: 0,
    }))

    expect(markup).toContain('orient="auto"')
    expect(markup).not.toContain('auto-start-auto')
  })
})
