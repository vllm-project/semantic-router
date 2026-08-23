import { describe, expect, expectTypeOf, it } from 'vitest'

import {
  assertManagementApiAgentSchema,
  createManagementApiAgentClient,
  type AgentProfile,
  type AgentProfilePage,
  type ManagementApiAgentOperationId,
  type ManagementApiAgentResponse,
  type ManagementApiAgentTransport,
} from './managementApiContract'

const profile: AgentProfile = {
  id: '10000000-0000-4000-8000-000000000001',
  namespaceId: '20000000-0000-4000-8000-000000000001',
  name: 'Builder',
  status: 'active',
  revision: 1,
  createdAt: '2026-08-24T00:00:00Z',
  updatedAt: '2026-08-24T00:00:00Z',
  contentRevision: 1,
  minimumTargetCapabilities: [],
  supportedModes: ['builder'],
  defaultForModes: ['builder'],
  skills: [],
  toolPolicy: { allow: [] },
  approvalPolicy: 'required',
  maximumTurnSeconds: 60,
  maximumToolSteps: 8,
  contextTokenBudget: 4096,
}

describe('generated Management Agent client', () => {
  it('exposes operation-specific response types', () => {
    expectTypeOf<ManagementApiAgentResponse<'getAgentProfiles'>>().toEqualTypeOf<AgentProfilePage>()
  })

  it('forwards typed operation options and validates the response schema', async () => {
    const calls: Array<{ operationId: ManagementApiAgentOperationId; options: unknown }> = []
    const transport: ManagementApiAgentTransport = {
      async request(operationId, options) {
        calls.push({ operationId, options })
        return {
          data: { data: [profile], page: { hasMore: false, pageSize: 25 } },
        }
      },
    }
    const client = createManagementApiAgentClient(transport)

    const response = await client.getAgentProfiles({
      query: { search: 'build', pageSize: 25 },
    })

    expect(response.data.data).toEqual([profile])
    expect(calls).toEqual([
      {
        operationId: 'getAgentProfiles',
        options: { query: { search: 'build', pageSize: 25 } },
      },
    ])
  })

  it('rejects a response that drifts from the canonical OpenAPI schema', async () => {
    const transport: ManagementApiAgentTransport = {
      async request() {
        return {
          data: {
            data: [{ ...profile, revision: 0 }],
            page: { hasMore: false, pageSize: 25 },
          },
        }
      },
    }
    const client = createManagementApiAgentClient(transport)

    await expect(client.getAgentProfiles()).rejects.toThrow('AgentProfilePage')
  })

  it('validates discriminated durable Agent event payloads', () => {
    const event = assertManagementApiAgentSchema('AgentEvent', {
      sessionId: '30000000-0000-4000-8000-000000000001',
      turnId: '40000000-0000-4000-8000-000000000001',
      sequence: 1,
      type: 'terminal',
      payload: { status: 'completed' },
      createdAt: '2026-08-24T00:00:00Z',
    })
    expect(event.type).toBe('terminal')

    expect(() =>
      assertManagementApiAgentSchema('AgentEvent', {
        ...event,
        payload: { content: [{ type: 'text', text: 'wrong payload for terminal' }] },
      }),
    ).toThrow('AgentEvent')
  })
})
