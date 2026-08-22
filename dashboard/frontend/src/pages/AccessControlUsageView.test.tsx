import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import type { AccessOverview, UsageSummary } from '../utils/inferenceAccessApi'
import AccessControlUsageView from './AccessControlUsageView'

const overview: AccessOverview = {
  users: 7,
  teams: 3,
  activeKeys: 9,
  expiringKeys: 2,
  accessGroups: 4,
  enabledBudgets: 5,
  requestsToday: 147,
  successfulToday: 146,
  tokensToday: 6649,
  p95LatencyMs: 820,
}

const usage: UsageSummary = {
  granularity: 'hour',
  requests: 147,
  successful: 146,
  failed: 1,
  promptTokens: 4000,
  completionTokens: 2649,
  totalTokens: 6649,
  activeKeys: 9,
  averageLatencyMs: 410,
  p95LatencyMs: 820,
  averageTtftMs: 92,
  p95TtftMs: 160,
  series: [],
  byModel: [],
  byUser: [],
  byTeam: [],
  byKey: [],
}

describe('access usage view', () => {
  it('merges access posture into the complete usage experience', () => {
    const markup = renderToStaticMarkup(
      createElement(AccessControlUsageView, {
        overview,
        usage,
        users: [],
        teams: [],
        keys: [],
        groups: [],
        usageScope: {
          type: 'global',
          id: '',
          model: '',
          range: 'today',
          granularity: 'auto',
          customFrom: '',
          customTo: '',
        },
        onUsageScopeChange: vi.fn(),
        loading: false,
      }),
    )

    expect(markup).toContain('Access at a glance')
    expect(markup).toContain('6,649 tokens')
    expect(markup).toContain('2 expiring soon')
    expect(markup).toContain('7 users · 3 teams')
    expect(markup).toContain('model grants')
    expect(markup).toContain('active quota policies')
    expect(markup).toContain('Traffic over time')
    expect(markup).toContain('1 hour per point')
    expect(markup).toContain('Granularity')
    expect(markup).toContain('Usage leaders')
  })
})
