import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import AgentRouterMetadata from './AgentRouterMetadata'

const metadata = {
  modelStepId: '11111111-1111-4111-8111-111111111111',
  requestId: 'request-42',
  selectedRecipe: 'balance',
  selectedDecision: 'complex_workload',
  selectedModel: 'remote/frontier',
  selectedAlgorithm: 'static',
  responsePath: 'upstream' as const,
  latencyMilliseconds: 420,
  ttftMilliseconds: 84,
  usage: {
    inputTokens: 120,
    outputTokens: 48,
    totalTokens: 168,
    inputCacheReadTokens: 30,
    outputReasoningTokens: 12,
  },
}

describe('Agent Router metadata', () => {
  it('renders a restrained collapsed summary and authoritative details', () => {
    const markup = renderToStaticMarkup(
      <AgentRouterMetadata metadata={metadata} canReadRequestLogs={false} />,
    )

    expect(markup).toContain('<details')
    expect(markup).not.toContain('<details open=""')
    expect(markup).toContain('remote/frontier')
    expect(markup).toContain('Complex Workload')
    expect(markup).toContain('420 ms')
    expect(markup).toContain('168 total')
    expect(markup).toContain('Cache read')
    expect(markup).not.toContain('Open request log')
    expect(markup).not.toContain('providerOpaque')
    expect(markup).not.toContain('Cost')
  })

  it('links to request logs only when the current identity can read them', () => {
    const markup = renderToStaticMarkup(
      <AgentRouterMetadata metadata={metadata} canReadRequestLogs />,
    )

    expect(markup).toContain('href="/logs?q=request-42"')
    expect(markup).toContain('Open request log')
  })
})
