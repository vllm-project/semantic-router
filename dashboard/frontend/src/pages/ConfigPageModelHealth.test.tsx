import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { ModelHealth } from '../utils/modelHealthApi'
import { ModelHealthSummary } from './ConfigPageModelHealth'

const summary = (props: Partial<Parameters<typeof ModelHealthSummary>[0]>) =>
  renderToStaticMarkup(
    createElement(ModelHealthSummary, { phase: 'ready', checking: false, ...props }),
  )

const healthy: ModelHealth = {
  model: 'qwen3-8b',
  status: 'healthy',
  checkedAt: new Date().toISOString(),
  recentMs: [],
  endpoints: [],
}

describe('model health summary', () => {
  it('says a passing failure is being retried', () => {
    const markup = summary({ phase: 'retrying' })
    expect(markup).toContain('Health could not be checked. Retrying automatically.')
    expect(markup).toContain('Retrying')
    expect(markup).not.toContain('no health checks')
  })

  it('names who cannot read health and which Dashboard has none', () => {
    expect(summary({ phase: 'unavailable', issue: 'forbidden' })).toContain(
      'Your account cannot read model health.',
    )
    expect(summary({ phase: 'unavailable', issue: 'unsupported' })).toContain(
      'This Dashboard has no health checks.',
    )
  })

  it('marks a kept answer as the previous result while it retries', () => {
    const markup = summary({ health: healthy, issue: 'unreachable' })
    expect(markup).toContain('Healthy')
    expect(markup).toContain('this is the previous result')
    expect(markup).not.toContain('Every backend answered')
    expect(summary({ health: healthy })).toContain('Every backend answered its last health check.')
  })
})
