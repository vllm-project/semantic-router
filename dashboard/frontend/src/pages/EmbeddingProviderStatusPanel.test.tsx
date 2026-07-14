import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import EmbeddingProviderStatusPanel from './EmbeddingProviderStatusPanel'

describe('EmbeddingProviderStatusPanel', () => {
  it('renders a healthy remote provider without exposing a credential value', () => {
    const markup = renderToStaticMarkup(
      <EmbeddingProviderStatusPanel
        provider={{
          mode: 'remote',
          backend: 'openai_compatible',
          model: 'text-embedding-3-small',
          dimension: 1536,
          api_key_env: 'OPENAI_API_KEY',
          api_key_env_set: true,
          healthy: true,
          last_checked_at: '2026-07-13T12:00:00Z',
        }}
      />,
    )

    expect(markup).toContain('Remote embedding provider')
    expect(markup).toContain('OpenAI compatible')
    expect(markup).toContain('text-embedding-3-small')
    expect(markup).toContain('OPENAI_API_KEY')
    expect(markup).toContain('Available')
    expect(markup).toContain('Healthy')
    expect(markup).not.toContain('test-secret')
  })

  it('renders probe and credential failures as actionable status', () => {
    const markup = renderToStaticMarkup(
      <EmbeddingProviderStatusPanel
        provider={{
          backend: 'openai_compatible',
          api_key_env: 'OPENAI_API_KEY',
          api_key_env_set: false,
          healthy: false,
          last_probe_error: 'embedding API key env is not set',
        }}
      />,
    )

    expect(markup).toContain('Needs attention')
    expect(markup).toContain('Missing')
    expect(markup).toContain('Last probe failed')
    expect(markup).toContain('embedding API key env is not set')
  })
})
