import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ConfigPageModelLiveVerification from './ConfigPageModelLiveVerification'

describe('ConfigPageModelLiveVerification', () => {
  it('shows a non-green disabled pending state while the real query is in flight', () => {
    const markup = renderToStaticMarkup(
      <ConfigPageModelLiveVerification
        model="logical-model"
        hasBackend
        allowed
        state={{ status: 'pending' }}
        onVerify={vi.fn()}
      />,
    )

    expect(markup).toContain('Sending test query')
    expect(markup).toContain('Verifying… logical-model with a real inference query')
    expect(markup).toContain('disabled=""')
    expect(markup).not.toContain('Live verified')
  })

  it('labels runtime inference evidence as live verification, not catalog verification', () => {
    const markup = renderToStaticMarkup(
      <ConfigPageModelLiveVerification
        model="logical-model"
        hasBackend
        allowed
        state={{
          status: 'verified',
          evidence: {
            verified: true,
            model: 'logical-model',
            providerModel: 'provider/real-model',
            backend: 'logical-model/primary',
            provider: 'openai',
            summary: 'OK from provider',
            verifiedAt: '2026-08-15T05:06:07Z',
            latencyMs: 18,
          },
        }}
        onVerify={vi.fn()}
      />,
    )

    expect(markup).toContain('Live verified')
    expect(markup).toContain('OK from provider')
    expect(markup).toContain('openai · 18 ms')
    expect(markup).toContain('Verify again logical-model with a real inference query')
    expect(markup).not.toContain('catalog verified')
  })

  it('shows an actionable retry state without treating failure as connectivity success', () => {
    const markup = renderToStaticMarkup(
      <ConfigPageModelLiveVerification
        model="logical-model"
        hasBackend
        allowed
        state={{ status: 'failed', message: 'Provider inference returned HTTP 401.' }}
        onVerify={vi.fn()}
      />,
    )

    expect(markup).toContain('Verification failed')
    expect(markup).toContain('Provider inference returned HTTP 401.')
    expect(markup).toContain('Retry logical-model with a real inference query')
    expect(markup).not.toContain('Live verified')
  })

  it('disables generation when the model has no backend or the user lacks evaluation.run', () => {
    const noBackend = renderToStaticMarkup(
      <ConfigPageModelLiveVerification
        model="metadata-only"
        hasBackend={false}
        allowed
        state={{ status: 'idle' }}
        onVerify={vi.fn()}
      />,
    )
    const noPermission = renderToStaticMarkup(
      <ConfigPageModelLiveVerification
        model="logical-model"
        hasBackend
        allowed={false}
        state={{ status: 'idle' }}
        onVerify={vi.fn()}
      />,
    )

    expect(noBackend).toContain('No backend')
    expect(noBackend).toContain('disabled=""')
    expect(noPermission).toContain('Run permission required')
    expect(noPermission).toContain('disabled=""')
  })
})
