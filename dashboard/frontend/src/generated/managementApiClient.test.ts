import { describe, expect, expectTypeOf, it } from 'vitest'

import {
  assertManagementApiSchema,
  assertManagementApiResponseSchema,
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
    expectTypeOf<
      ManagementApiResponse<'getRoutingEntrypoints'>
    >().toEqualTypeOf<RoutingEntrypointPage>()
    expectTypeOf<ManagementApiResponse<'getRoutingExportsCurrent'>>().toEqualTypeOf<string>()
    expectTypeOf<
      ManagementApiResponse<'getProviderCredentials'>
    >().toEqualTypeOf<ProviderCredentialPage>()
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

  it('accepts additive response fields while preserving required-field and type checks', async () => {
    const additiveResponse = {
      data: [
        {
          ...profile,
          futureProfileCapability: { enabled: true },
        },
      ],
      page: {
        hasMore: false,
        pageSize: 25,
        futureCursorHint: 'next-generation',
      },
      futurePageMetadata: { source: 'router' },
    }

    expect(assertManagementApiResponseSchema('AgentProfilePage', additiveResponse)).toEqual(
      additiveResponse,
    )

    const forwardCompatibleTransport: ManagementApiTransport = {
      async request() {
        return {
          data: additiveResponse,
          status: 200,
          mediaType: 'application/vnd.vllm-semantic-router.management.v1+json',
        }
      },
    }
    await expect(
      createManagementApiClient(forwardCompatibleTransport).getAgentProfiles(),
    ).resolves.toMatchObject({ data: additiveResponse })

    const profileWithoutRequiredName: Record<string, unknown> = { ...profile }
    delete profileWithoutRequiredName.name
    expect(() =>
      assertManagementApiResponseSchema('AgentProfilePage', {
        data: [profileWithoutRequiredName],
        page: { hasMore: false, pageSize: 25 },
      }),
    ).toThrow('AgentProfilePage')

    expect(() =>
      assertManagementApiResponseSchema('AgentProfilePage', {
        data: [{ ...profile, revision: '1' }],
        page: { hasMore: false, pageSize: 25 },
      }),
    ).toThrow('AgentProfilePage')

    const serviceBootstrap = {
      principalId: '10000000-0000-4000-8000-000000000001',
      roleBindingId: '20000000-0000-4000-8000-000000000001',
      serviceAccountId: '30000000-0000-4000-8000-000000000001',
      serviceCredential: {
        resourceId: '40000000-0000-4000-8000-000000000001',
        kind: 'service_credential' as const,
        secret: 'one-time-secret',
        expiresAt: '2026-08-26T01:00:00Z',
      },
      finalizationRequired: true,
      futureBootstrapField: 'accepted',
    }
    expect(assertManagementApiResponseSchema('BootstrapResponse', serviceBootstrap)).toEqual(
      serviceBootstrap,
    )
    expect(() =>
      assertManagementApiResponseSchema('BootstrapResponse', {
        ...serviceBootstrap,
        serviceAccountId: 3,
      }),
    ).toThrow('BootstrapResponse')

    expect(() =>
      assertManagementApiResponseSchema('MutationReceipt', {
        resource: { kind: 'api_key', id: 'key-1', revision: 1 },
        operation: 42,
      }),
    ).toThrow('MutationReceipt')
  })

  it('keeps the generic schema assertion strict for request documents', () => {
    expect(() =>
      assertManagementApiSchema('APIKeyCreateRequest', {
        name: 'Customer key',
        owner: { type: 'user', id: 'user-1' },
        futureRequestField: true,
      }),
    ).toThrow('APIKeyCreateRequest')
  })

  it('validates discriminated durable Agent event payloads through the generic schema registry', () => {
    const event = assertManagementApiResponseSchema('AgentEvent', {
      sessionId: '30000000-0000-4000-8000-000000000001',
      turnId: '40000000-0000-4000-8000-000000000001',
      sequence: 1,
      type: 'terminal',
      payload: { status: 'completed' },
      createdAt: '2026-08-24T00:00:00Z',
    })
    expect(event.type).toBe('terminal')

    expect(() =>
      assertManagementApiResponseSchema('AgentEvent', {
        ...event,
        payload: { content: [{ type: 'text', text: 'wrong payload for terminal' }] },
      }),
    ).toThrow('AgentEvent')
  })

  it('accepts canonical empty usage responses and rejects nullable freshness fields', () => {
    const timing = {
      sampleCount: '0',
      totalMilliseconds: '0',
      averageMilliseconds: 0,
      p50Milliseconds: 0,
      p95Milliseconds: 0,
      p99Milliseconds: 0,
      percentilesAreEstimated: true,
    }
    const totals = {
      requests: '0',
      successfulRequests: '0',
      inputTokens: '0',
      outputTokens: '0',
      totalTokens: '0',
      incompleteDispatches: '0',
      completeness: 'complete' as const,
      costs: [],
      latency: timing,
      ttft: timing,
    }
    const responses = [
      ['UsageSummary', { totals, grain: 'hour', final: false }],
      ['UsageSeries', { points: [], grain: 'hour', final: false }],
      ['UsageBreakdown', { dimension: 'api_key', rows: [], grain: 'hour', final: false }],
    ] as const

    for (const [schemaName, payload] of responses) {
      expect(assertManagementApiResponseSchema(schemaName, payload)).toEqual(payload)
      for (const freshnessField of ['asOf', 'ledgerWatermark', 'ingestionLag'] as const) {
        expect(() =>
          assertManagementApiResponseSchema(schemaName, { ...payload, [freshnessField]: null }),
        ).toThrow(schemaName)
      }
    }
  })

  it('accepts explicit unknown request evidence and rejects nullable log arrays', () => {
    const request = {
      admissionId: 'request-with-historical-evidence',
      eventId: '10000000-0000-4000-8000-000000000001',
      occurredAt: '2026-08-24T00:00:00Z',
      completedAt: '2026-08-24T00:00:01Z',
      protocol: 'openai.chat',
      path: '/v1/chat/completions',
      statusCode: 200,
      usageState: 'known_actual',
      inputTokens: '10',
      outputTokens: '2',
      latencyMilliseconds: 1000,
      stream: true,
      toolCall: false,
      models: [],
      costs: [],
    }
    const page = { data: [request], page: { hasMore: false, pageSize: 50 } }
    const detail = {
      data: { request, routing: {}, quotaReceipts: [], dispatches: [] },
    }

    expect(assertManagementApiResponseSchema('RequestLogPage', page)).toEqual(page)
    expect(assertManagementApiResponseSchema('RequestLogDetail', detail)).toEqual(detail)
    expect(() =>
      assertManagementApiResponseSchema('RequestLogPage', {
        ...page,
        data: [{ ...request, models: null }],
      }),
    ).toThrow('RequestLogPage')
    expect(() =>
      assertManagementApiResponseSchema('RequestLogDetail', {
        data: { ...detail.data, quotaReceipts: null },
      }),
    ).toThrow('RequestLogDetail')
  })

  it('accepts the canonical effective-policy wire shape and rejects nullable optional meters', () => {
    const policy = {
      subject: { type: 'api_key' as const, id: '10000000-0000-4000-8000-000000000001' },
      revision: 12,
      appliedRevision: 12,
      access: { grants: [] },
      quota: {
        meters: [
          {
            policyId: '20000000-0000-4000-8000-000000000001',
            ruleId: '30000000-0000-4000-8000-000000000001',
            bindingId: '40000000-0000-4000-8000-000000000001',
            source: {
              subjectType: 'api_key',
              subjectId: '10000000-0000-4000-8000-000000000001',
              bindingId: '40000000-0000-4000-8000-000000000001',
            },
            counterOwner: '10000000-0000-4000-8000-000000000001',
            metric: 'total_tokens',
            algorithm: 'sliding_log',
            accounting: 'response_actual',
            enforcement: 'enforce',
            limit: '30000',
            used: '0',
            remaining: null,
            completeness: 'unknown' as const,
            knownDispatches: '0',
            incompleteDispatches: '1',
            capacityState: 'fenced' as const,
            activeFenceIds: ['50000000-0000-4000-8000-000000000001'],
            freshness: { source: 'valkey', asOf: '2026-08-26T00:00:00Z' },
          },
        ],
        unknownUsageFences: ['50000000-0000-4000-8000-000000000001'],
        asOf: '2026-08-26T00:00:00Z',
      },
    }

    expect(assertManagementApiResponseSchema('EffectivePolicy', policy)).toEqual(policy)
    for (const optional of ['currency', 'overage', 'resetAt'] as const) {
      const meter = { ...policy.quota.meters[0], [optional]: null }
      expect(() =>
        assertManagementApiResponseSchema('EffectivePolicy', {
          ...policy,
          quota: { ...policy.quota, meters: [meter] },
        }),
      ).toThrow('EffectivePolicy')
    }
  })
})
