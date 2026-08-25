import { describe, expect, expectTypeOf, it } from 'vitest'

import {
  assertManagementApiSchema,
  createManagementApiClient,
  type APIKeyPage,
  type AccessStatistics,
  type AgentEventPage,
  type AgentProfile,
  type ManagementApiClientOperationId,
  type ManagementApiResponse,
  type ManagementApiTransport,
  type ProviderCredentialPage,
  type RoutingEntrypointPage,
  type RoutingModelPage,
  type RoutingRecipePage,
  type TeamPage,
  type UsageSummary,
  type UserPage,
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

describe('generated Management client', () => {
  it('exposes operation-specific response types across console domains', () => {
    expectTypeOf<ManagementApiResponse<'getApiKeys'>>().toEqualTypeOf<APIKeyPage>()
    expectTypeOf<ManagementApiResponse<'getUsers'>>().toEqualTypeOf<UserPage>()
    expectTypeOf<ManagementApiResponse<'getTeams'>>().toEqualTypeOf<TeamPage>()
    expectTypeOf<ManagementApiResponse<'getRoutingModels'>>().toEqualTypeOf<RoutingModelPage>()
    expectTypeOf<ManagementApiResponse<'getRoutingRecipes'>>().toEqualTypeOf<RoutingRecipePage>()
    expectTypeOf<ManagementApiResponse<'getRoutingEntrypoints'>>().toEqualTypeOf<RoutingEntrypointPage>()
    expectTypeOf<ManagementApiResponse<'getRoutingExportsCurrent'>>().toEqualTypeOf<string>()
    expectTypeOf<ManagementApiResponse<'getProviderCredentials'>>().toEqualTypeOf<ProviderCredentialPage>()
    expectTypeOf<ManagementApiResponse<'getStatistics'>>().toEqualTypeOf<AccessStatistics>()
    expectTypeOf<ManagementApiResponse<'getUsage'>>().toEqualTypeOf<UsageSummary>()
    expectTypeOf<
      ManagementApiResponse<'getAgentSessionsBySessionEvents'>
    >().toEqualTypeOf<AgentEventPage>()
  })

  it('rejects an event stream on the buffered JSON client', async () => {
    const eventStream: ManagementApiTransport = {
      async request() {
        return { data: 'event: progress\n\n', status: 200, mediaType: 'text/event-stream' }
      },
    }
    await expect(
      createManagementApiClient(eventStream).getAgentSessionsBySessionEvents({
        pathParameters: { session: '10000000-0000-4000-8000-000000000001' },
      }),
    ).rejects.toThrow('expected application/vnd.vllm-semantic-router.management.v1+json')
  })

  it('validates text response media types from the generated OpenAPI contract', async () => {
    const validText: ManagementApiTransport = {
      async request() {
        return { data: 'version: v0.3\n', status: 200, mediaType: 'application/yaml' }
      },
    }
    await expect(createManagementApiClient(validText).getRoutingExportsCurrent()).resolves.toEqual({
      data: 'version: v0.3\n',
      status: 200,
      mediaType: 'application/yaml',
    })

    const wrongMediaType: ManagementApiTransport = {
      async request() {
        return { data: 'version: v0.3\n', status: 200, mediaType: 'application/json' }
      },
    }
    await expect(
      createManagementApiClient(wrongMediaType).getRoutingExportsCurrent(),
    ).rejects.toThrow('expected application/yaml')

    const missingMediaType: ManagementApiTransport = {
      async request() {
        return { data: 'version: v0.3\n', status: 200 }
      },
    }
    await expect(
      createManagementApiClient(missingMediaType).getRoutingExportsCurrent(),
    ).rejects.toThrow('omitted the response media type')
  })

  it('forwards typed operation options and validates the response schema', async () => {
    const calls: Array<{ operationId: ManagementApiClientOperationId; options: unknown }> = []
    const transport: ManagementApiTransport = {
      async request(operationId, options) {
        calls.push({ operationId, options })
        return {
          data: { data: [profile], page: { hasMore: false, pageSize: 25 } },
          status: 200,
          mediaType: 'application/vnd.vllm-semantic-router.management.v1+json',
        }
      },
    }
    const client = createManagementApiClient(transport)

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

  it('rejects JSON response drift and a body on a 204 operation', async () => {
    const invalidJSON: ManagementApiTransport = {
      async request() {
        return {
          data: {
            data: [{ ...profile, revision: 0 }],
            page: { hasMore: false, pageSize: 25 },
          },
          status: 200,
          mediaType: 'application/vnd.vllm-semantic-router.management.v1+json',
        }
      },
    }
    await expect(createManagementApiClient(invalidJSON).getAgentProfiles()).rejects.toThrow(
      'AgentProfilePage',
    )

    const invalidEmpty: ManagementApiTransport = {
      async request() {
        return { data: { unexpected: true }, status: 204 }
      },
    }
    await expect(
      createManagementApiClient(invalidEmpty).deleteAccessPoliciesByPolicyId({
        pathParameters: { policyId: '10000000-0000-4000-8000-000000000001' },
        headers: { 'If-Match': '"access-policy:1"' },
      }),
    ).rejects.toThrow('body for empty Management operation')
  })

  it('validates discriminated durable Agent event payloads through the generic schema registry', () => {
    const event = assertManagementApiSchema('AgentEvent', {
      sessionId: '30000000-0000-4000-8000-000000000001',
      turnId: '40000000-0000-4000-8000-000000000001',
      sequence: 1,
      type: 'terminal',
      payload: { status: 'completed' },
      createdAt: '2026-08-24T00:00:00Z',
    })
    expect(event.type).toBe('terminal')

    expect(() =>
      assertManagementApiSchema('AgentEvent', {
        ...event,
        payload: { content: [{ type: 'text', text: 'wrong payload for terminal' }] },
      }),
    ).toThrow('AgentEvent')
  })
})
