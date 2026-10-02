import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { StatusTab, TeamTab, WorkerTab } from './OpenClawPageTabs'

// With the feature disabled the collection requests fail with the disabled
// error. The empty states must not invite the user to run operations the
// disabled branch never registers; the error notice is the only message.
const disabledError = 'OpenClaw feature disabled'

describe('OpenClaw tabs suppress the empty state while the request failed', () => {
  it('TeamTab hides the create invitation when teams are unavailable', () => {
    const markup = renderToStaticMarkup(
      <TeamTab
        teams={[]}
        teamsLoading={false}
        teamsError={disabledError}
        containers={[]}
        onTeamsUpdated={vi.fn()}
        onRetryTeams={vi.fn()}
        readOnly={false}
      />,
    )
    expect(markup).toContain(disabledError)
    expect(markup).not.toContain('Create one to organize workers')
    expect(markup).not.toContain('No teams yet')
    expect(markup).not.toContain('New Team')
  })

  it('TeamTab shows the empty state when no request failed', () => {
    const markup = renderToStaticMarkup(
      <TeamTab
        teams={[]}
        teamsLoading={false}
        teamsError={null}
        containers={[]}
        onTeamsUpdated={vi.fn()}
        onRetryTeams={vi.fn()}
        readOnly={false}
      />,
    )
    expect(markup).toContain('No teams yet. Create one to organize workers.')
  })

  it('WorkerTab hides the create invitation when workers are unavailable', () => {
    const markup = renderToStaticMarkup(
      <WorkerTab
        containers={[]}
        readOnly={false}
        teams={[]}
        workersError={disabledError}
        onProvisioned={vi.fn()}
        onRetryWorkers={vi.fn()}
        onSwitchToStatus={vi.fn()}
        onSwitchToTeam={vi.fn()}
      />,
    )
    expect(markup).toContain(disabledError)
    expect(markup).not.toContain('No workers created yet')
    expect(markup).not.toContain('New Worker')
  })

  it('StatusTab hides the provision invitation when the status request failed', () => {
    const markup = renderToStaticMarkup(
      <StatusTab
        containers={[]}
        readOnly={false}
        statusError={disabledError}
        statusLoading={false}
        onRefresh={vi.fn()}
      />,
    )
    expect(markup).toContain(disabledError)
    expect(markup).not.toContain('Use Claw Worker to create one')
  })
})
