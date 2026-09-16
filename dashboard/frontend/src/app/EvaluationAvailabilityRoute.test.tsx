import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import EvaluationAvailabilityRoute from './EvaluationAvailabilityRoute'

const renderRoute = (props: { available: boolean; isLoading: boolean; reason?: string }) =>
  renderToStaticMarkup(
    <MemoryRouter>
      <EvaluationAvailabilityRoute
        available={props.available}
        isLoading={props.isLoading}
        reason={props.reason ?? ''}
      >
        <main>Evaluation workspace</main>
      </EvaluationAvailabilityRoute>
    </MemoryRouter>,
  )

describe('EvaluationAvailabilityRoute', () => {
  it('renders the workspace only after server initialization succeeds', () => {
    expect(renderRoute({ available: true, isLoading: false })).toContain('Evaluation workspace')
  })

  it('renders a stable disabled explanation instead of a broken workspace', () => {
    const markup = renderRoute({
      available: false,
      isLoading: false,
      reason: 'Evaluation is disabled for this deployment.',
    })
    expect(markup).toContain('Evaluation is not available')
    expect(markup).toContain('Evaluation is disabled for this deployment.')
    expect(markup).not.toContain('Evaluation workspace')
  })

  it('does not expose the workspace while settings are loading', () => {
    const markup = renderRoute({ available: false, isLoading: true })
    expect(markup).toContain('Checking Evaluation')
    expect(markup).not.toContain('Evaluation workspace')
  })
})
