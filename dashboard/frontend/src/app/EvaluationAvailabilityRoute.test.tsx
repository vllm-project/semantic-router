import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import EvaluationAvailabilityRoute from './EvaluationAvailabilityRoute'
import type { CapabilityState, DashboardCapability } from '../contexts/dashboardCapabilities'

const capability = (
  state: CapabilityState,
  reason = '',
  remediation = '',
): DashboardCapability => ({
  id: 'evaluation',
  enabled: state !== 'disabled',
  configured: state !== 'unconfigured',
  authorized: state !== 'unauthorized',
  reachable: state !== 'unreachable',
  degraded: state === 'degraded',
  state,
  reason,
  remediation,
  checkedAt: null,
})

const renderRoute = (props: {
  capability?: DashboardCapability
  isLoading?: boolean
  error?: string
  featureName?: string
}) =>
  renderToStaticMarkup(
    <MemoryRouter>
      <EvaluationAvailabilityRoute
        capability={props.capability}
        isLoading={props.isLoading ?? false}
        error={props.error ?? null}
        featureName={props.featureName}
        onRefreshAccess={() => undefined}
      >
        <main>Evaluation workspace</main>
      </EvaluationAvailabilityRoute>
    </MemoryRouter>,
  )

describe('EvaluationAvailabilityRoute', () => {
  it('renders the workspace when the capability is available or degraded', () => {
    expect(renderRoute({ capability: capability('available') })).toContain('Evaluation workspace')
    expect(renderRoute({ capability: capability('degraded') })).toContain('Evaluation workspace')
  })

  it('explains an unavailable capability with its remediation', () => {
    const markup = renderRoute({
      capability: capability(
        'unreachable',
        'upstream_unreachable',
        'The sr-bench service did not answer its health check.',
      ),
    })
    expect(markup).toContain('Evaluation is not available')
    expect(markup).toContain('The sr-bench service did not answer its health check.')
    expect(markup).not.toContain('Evaluation workspace')
  })

  it('does not expose the workspace while capabilities load or a probe is pending', () => {
    for (const props of [
      { isLoading: true },
      { capability: capability('unreachable', 'probe_pending', 'The first health check has not completed yet.') },
    ]) {
      const markup = renderRoute(props)
      expect(markup).toContain('Checking Evaluation')
      expect(markup).not.toContain('Evaluation workspace')
    }
  })

  it('names the ML pipeline in its error state', () => {
    const markup = renderRoute({
      error: 'Dashboard feature availability is unavailable. Refresh access to retry.',
      featureName: 'ML pipeline',
    })
    expect(markup).toContain('Unable to check ML pipeline access')
    expect(markup).not.toContain('Unable to check Evaluation access')
    expect(markup).not.toContain('Evaluation workspace')
  })

  it('offers recovery without exposing a workspace whose access is unverified', () => {
    for (const props of [
      { error: 'Dashboard feature availability is unavailable. Refresh access to retry.' },
      {},
    ]) {
      const markup = renderRoute(props)
      expect(markup).toContain('Unable to check Evaluation access')
      expect(markup).toContain('Refresh access')
      expect(markup).not.toContain('Evaluation workspace')
    }
  })
})
