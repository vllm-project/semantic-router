import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { AgentProfile } from '../generated/managementApiContract'
import { AgentResourceRow } from './AgentManagementResourceViews'
import { resourcesForAgentTab } from './agentManagementResourceProjection'

const profile = {
  id: 'profile-1',
  name: 'Builder',
  description: 'Router-managed default.',
  status: 'active',
  supportedModes: ['builder'],
  skills: [],
} as unknown as AgentProfile

describe('Agent Management resource transitions', () => {
  it('never renders resources through a different tab shape', () => {
    expect(resourcesForAgentTab('skills', 'profiles', [profile])).toEqual([])
    expect(resourcesForAgentTab('profiles', 'profiles', [profile])).toEqual([profile])
  })

  it('keeps a malformed transition row from crashing the page', () => {
    expect(() =>
      renderToStaticMarkup(
        <table>
          <tbody>
            <AgentResourceRow
              tab="skills"
              resource={profile}
              disabled={false}
              onOpen={() => undefined}
            />
          </tbody>
        </table>,
      ),
    ).not.toThrow()
  })
})
