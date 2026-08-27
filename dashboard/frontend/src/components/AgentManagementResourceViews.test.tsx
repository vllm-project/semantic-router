import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { AgentSkill } from '../generated/managementApiContract'
import { AgentResourceRow } from './AgentManagementResourceViews'
import { resourcesForAgentTab } from './agentManagementResourceProjection'

const skill = {
  id: 'skill-1',
  name: 'Recipe designer',
  description: 'Builds and validates Recipes.',
  status: 'active',
  builtin: true,
  requiredTools: [],
  minimumCapabilities: [],
} as unknown as AgentSkill

describe('Agent Management resource transitions', () => {
  it('never renders resources through a different tab shape', () => {
    expect(resourcesForAgentTab('skills', 'tools', [skill])).toEqual([])
    expect(resourcesForAgentTab('skills', 'skills', [skill])).toEqual([skill])
  })

  it('keeps a malformed transition row from crashing the page', () => {
    expect(() =>
      renderToStaticMarkup(
        <table>
          <tbody>
            <AgentResourceRow
              tab="skills"
              resource={skill}
              disabled={false}
              onOpen={() => undefined}
            />
          </tbody>
        </table>,
      ),
    ).not.toThrow()
  })
})
