import {
  MANAGEMENT_API_HEADERS,
  assertManagementApiAgentSchema,
  createManagementApiAgentClient,
  type AgentEvent,
  type AgentLiveModelStepEvent,
  type AgentProfileInput,
  type AgentSessionInput,
  type AgentSkillInput,
  type AgentToolCredentialInput,
  type AgentToolSourceInput,
  type AgentToolSourcePatchInput,
  type AgentTurnCreateRequest,
  type ManagementApiClientResponse,
  type MutationReceipt,
  type OperationReference,
  type ResourceReference,
} from '../generated/managementApiContract'
import { managementApiAgentTransport, managementOperationStream } from './managementApiContract'

const client = createManagementApiAgentClient(managementApiAgentTransport)

type AgentDetail<Resource> = ManagementApiClientResponse<Resource> & { etag: string }

function commandHeaders(): Record<string, string> {
  if (typeof crypto === 'undefined' || typeof crypto.randomUUID !== 'function') {
    throw new Error('This browser cannot create a secure Management request.')
  }
  return { [MANAGEMENT_API_HEADERS.idempotencyKey]: crypto.randomUUID() }
}

function revisionHeaders(etag: string): Record<string, string> {
  if (!etag.trim()) throw new Error('A current resource ETag is required.')
  return { [MANAGEMENT_API_HEADERS.ifMatch]: etag }
}

function listQuery(search?: string, cursor?: string, pageSize = 50) {
  return {
    pageSize: Math.min(Math.max(pageSize, 1), 100),
    ...(search?.trim() ? { search: search.trim() } : {}),
    ...(cursor ? { cursor } : {}),
  }
}

function requireResource(receipt: MutationReceipt): ResourceReference {
  if ('resource' in receipt) return receipt.resource
  throw new Error('Router returned an operation receipt for a resource mutation.')
}

function requireOperation(receipt: MutationReceipt): OperationReference {
  if ('operation' in receipt) return receipt.operation
  throw new Error('Router returned a resource receipt for an asynchronous mutation.')
}

async function requireDetail<Resource>(
  responsePromise: Promise<ManagementApiClientResponse<{ data: Resource }>>,
): Promise<AgentDetail<Resource>> {
  const response = await responsePromise
  if (!response.etag) throw new Error('Router did not return the resource revision token.')
  return { data: response.data.data, etag: response.etag }
}

function abortableDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }
    const onAbort = () => {
      window.clearTimeout(timer)
      reject(new DOMException('Aborted', 'AbortError'))
    }
    const timer = window.setTimeout(() => {
      signal?.removeEventListener('abort', onAbort)
      resolve()
    }, milliseconds)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

const profilePage = async (search?: string, cursor?: string, pageSize = 50, signal?: AbortSignal) =>
  (await client.getAgentProfiles({ query: listQuery(search, cursor, pageSize), signal })).data

const profileDetail = (id: string, signal?: AbortSignal) =>
  requireDetail(client.getAgentProfilesByProfile({ pathParameters: { profile: id }, signal }))

const skillPage = async (search?: string, cursor?: string, pageSize = 50, signal?: AbortSignal) =>
  (await client.getAgentSkills({ query: listQuery(search, cursor, pageSize), signal })).data

const skillDetail = (id: string, signal?: AbortSignal) =>
  requireDetail(client.getAgentSkillsBySkill({ pathParameters: { skill: id }, signal }))

const toolPage = async (search?: string, cursor?: string, pageSize = 50, signal?: AbortSignal) =>
  (await client.getAgentTools({ query: listQuery(search, cursor, pageSize), signal })).data

const credentialPage = async (
  search?: string,
  cursor?: string,
  pageSize = 50,
  signal?: AbortSignal,
) =>
  (
    await client.getAgentToolCredentials({
      query: listQuery(search, cursor, pageSize),
      signal,
    })
  ).data

const credentialDetail = (id: string, signal?: AbortSignal) =>
  requireDetail(
    client.getAgentToolCredentialsByCredential({
      pathParameters: { credential: id },
      signal,
    }),
  )

const toolSourcePage = async (
  search?: string,
  cursor?: string,
  pageSize = 50,
  signal?: AbortSignal,
) => (await client.getAgentToolSources({ query: listQuery(search, cursor, pageSize), signal })).data

const toolSourceDetail = (id: string, signal?: AbortSignal) =>
  requireDetail(client.getAgentToolSourcesBySource({ pathParameters: { source: id }, signal }))

const sessionPage = async (search?: string, cursor?: string, pageSize = 50, signal?: AbortSignal) =>
  (await client.getAgentSessions({ query: listQuery(search, cursor, pageSize), signal })).data

const sessionDetail = (id: string, signal?: AbortSignal) =>
  requireDetail(client.getAgentSessionsBySession({ pathParameters: { session: id }, signal }))

export const assertAgentEvent = (value: unknown): AgentEvent =>
  assertManagementApiAgentSchema('AgentEvent', value)

export const assertAgentLiveModelStepEvent = (value: unknown): AgentLiveModelStepEvent =>
  assertManagementApiAgentSchema('AgentLiveModelStepEvent', value)

export const agentManagementApi = {
  listProfiles: profilePage,
  getProfile: profileDetail,
  createProfile: async (input: AgentProfileInput) => {
    const receipt = requireResource(
      (await client.postAgentProfiles({ body: input, headers: commandHeaders() })).data,
    )
    return profileDetail(receipt.id)
  },
  patchProfile: async (id: string, input: Partial<AgentProfileInput>, etag: string) => {
    await client.patchAgentProfilesByProfile({
      pathParameters: { profile: id },
      body: input,
      headers: revisionHeaders(etag),
    })
    return profileDetail(id)
  },
  deleteProfile: (id: string, etag: string) =>
    client.deleteAgentProfilesByProfile({
      pathParameters: { profile: id },
      headers: revisionHeaders(etag),
    }),

  listSkills: skillPage,
  getSkill: skillDetail,
  createSkill: async (input: AgentSkillInput) => {
    const receipt = requireResource(
      (await client.postAgentSkills({ body: input, headers: commandHeaders() })).data,
    )
    return skillDetail(receipt.id)
  },
  patchSkill: async (id: string, input: Partial<AgentSkillInput>, etag: string) => {
    await client.patchAgentSkillsBySkill({
      pathParameters: { skill: id },
      body: input,
      headers: revisionHeaders(etag),
    })
    return skillDetail(id)
  },
  deleteSkill: (id: string, etag: string) =>
    client.deleteAgentSkillsBySkill({
      pathParameters: { skill: id },
      headers: revisionHeaders(etag),
    }),

  listTools: toolPage,
  findTool: async (name: string, signal?: AbortSignal) => {
    let cursor: string | undefined
    const seenCursors = new Set<string>()
    do {
      const page = await toolPage(name, cursor, 100, signal)
      const exact = page.data.find((tool) => tool.name === name)
      if (exact) return exact
      cursor = page.page.nextCursor
      if (!page.page.hasMore || !cursor || seenCursors.has(cursor)) return null
      seenCursors.add(cursor)
    } while (cursor)
    return null
  },

  listToolCredentials: credentialPage,
  getToolCredential: credentialDetail,
  createToolCredential: async (input: AgentToolCredentialInput) => {
    const receipt = requireResource(
      (await client.postAgentToolCredentials({ body: input, headers: commandHeaders() })).data,
    )
    return credentialDetail(receipt.id)
  },
  patchToolCredential: async (
    id: string,
    input: { name?: string; status?: 'active' | 'disabled' },
    etag: string,
  ) => {
    await client.patchAgentToolCredentialsByCredential({
      pathParameters: { credential: id },
      body: input,
      headers: revisionHeaders(etag),
    })
    return credentialDetail(id)
  },
  rotateToolCredential: async (id: string, secret: string, etag: string) => {
    await client.postAgentToolCredentialsByCredentialRotate({
      pathParameters: { credential: id },
      body: { secret },
      headers: { ...commandHeaders(), ...revisionHeaders(etag) },
    })
    return credentialDetail(id)
  },
  deleteToolCredential: (id: string, etag: string) =>
    client.deleteAgentToolCredentialsByCredential({
      pathParameters: { credential: id },
      headers: revisionHeaders(etag),
    }),

  listToolSources: toolSourcePage,
  getToolSource: toolSourceDetail,
  createToolSource: async (input: AgentToolSourceInput) => {
    const receipt = requireResource(
      (await client.postAgentToolSources({ body: input, headers: commandHeaders() })).data,
    )
    return toolSourceDetail(receipt.id)
  },
  patchToolSource: async (id: string, input: AgentToolSourcePatchInput, etag: string) => {
    await client.patchAgentToolSourcesBySource({
      pathParameters: { source: id },
      body: input,
      headers: revisionHeaders(etag),
    })
    return toolSourceDetail(id)
  },
  deleteToolSource: (id: string, etag: string) =>
    client.deleteAgentToolSourcesBySource({
      pathParameters: { source: id },
      headers: revisionHeaders(etag),
    }),
  testToolSource: async (id: string) =>
    requireOperation(
      (
        await client.postAgentToolSourcesBySourceTest({
          pathParameters: { source: id },
          headers: commandHeaders(),
        })
      ).data,
    ),
  approveToolSource: async (id: string, discoveryDigest: string, etag: string) => {
    requireResource(
      (
        await client.postAgentToolSourcesBySourceApprove({
          pathParameters: { source: id },
          body: { discoveryDigest },
          headers: { ...commandHeaders(), ...revisionHeaders(etag) },
        })
      ).data,
    )
    return toolSourceDetail(id)
  },

  listSessions: sessionPage,
  getSession: sessionDetail,
  createSession: async (input: AgentSessionInput) => {
    const receipt = requireResource(
      (await client.postAgentSessions({ body: input, headers: commandHeaders() })).data,
    )
    return sessionDetail(receipt.id)
  },
  patchSession: async (id: string, input: { title?: string; status?: 'closed' }, etag: string) => {
    await client.patchAgentSessionsBySession({
      pathParameters: { session: id },
      body: input,
      headers: revisionHeaders(etag),
    })
    return sessionDetail(id)
  },
  deleteSession: (id: string, etag: string) =>
    client.deleteAgentSessionsBySession({
      pathParameters: { session: id },
      headers: revisionHeaders(etag),
    }),
  createTurn: async (sessionId: string, input: AgentTurnCreateRequest) =>
    requireResource(
      (
        await client.postAgentSessionsBySessionTurns({
          pathParameters: { session: sessionId },
          body: input,
          headers: commandHeaders(),
        })
      ).data,
    ),
  cancelTurn: (sessionId: string, turnId: string) =>
    client.postAgentSessionsBySessionTurnsByTurnCancel({
      pathParameters: { session: sessionId, turn: turnId },
      headers: commandHeaders(),
    }),
  listLatestEvents: async (sessionId: string, pageSize = 100, signal?: AbortSignal) =>
    (
      await client.getAgentSessionsBySessionEvents({
        pathParameters: { session: sessionId },
        query: { pageSize: Math.min(Math.max(pageSize, 1), 100) },
        signal,
      })
    ).data,
  listEarlierEvents: async (
    sessionId: string,
    before: string,
    pageSize = 100,
    signal?: AbortSignal,
  ) =>
    (
      await client.getAgentSessionsBySessionEvents({
        pathParameters: { session: sessionId },
        query: { cursor: before, pageSize: Math.min(Math.max(pageSize, 1), 100) },
        signal,
      })
    ).data,
  openEventStream: (sessionId: string, since: number, signal: AbortSignal) =>
    managementOperationStream('getAgentSessionsBySessionEvents', {
      pathParameters: { session: sessionId },
      // Zero is an explicit durable cursor. Omitting it could skip a Turn
      // committed between the history request and SSE subscription.
      headers: { 'Last-Event-ID': String(since) },
      signal,
    }),
  getArtifact: async (id: string, signal?: AbortSignal) =>
    (
      await client.getAgentArtifactsByArtifact({
        pathParameters: { artifact: id },
        signal,
      })
    ).data.data,
  getArtifactContent: async (id: string, signal?: AbortSignal) =>
    (
      await client.getAgentArtifactsByArtifactContent({
        pathParameters: { artifact: id },
        signal,
      })
    ).data.data,
  commitPublication: async (planId: string, planDigest: string, planEtag: string) =>
    requireOperation(
      (
        await client.postPublicationPlansByPlanCommit({
          pathParameters: { plan: planId },
          body: { planDigest },
          headers: { ...commandHeaders(), ...revisionHeaders(planEtag) },
        })
      ).data,
    ),
  waitForOperation: async (
    operationId: string,
    options: { signal?: AbortSignal; timeoutMilliseconds?: number } = {},
  ): Promise<void> => {
    const deadline = Date.now() + (options.timeoutMilliseconds ?? 60_000)
    let delay = 250
    while (Date.now() < deadline) {
      const status = (
        await client.getOperationsByOperationId({
          pathParameters: { operationId },
          signal: options.signal,
        })
      ).data
      if (status.state === 'succeeded') return
      if (status.state === 'failed' || status.state === 'cancelled') {
        const detail = status.itemErrors?.[0]?.reason
        throw new Error(detail || `Operation ${status.state}.`)
      }
      if (status.state === 'partially_succeeded') {
        throw new Error('The operation completed only partially.')
      }
      await abortableDelay(delay, options.signal)
      delay = Math.min(Math.round(delay * 1.5), 2_000)
    }
    throw new Error('The operation is still running. Try again shortly.')
  },
}

export type AgentManagementApi = typeof agentManagementApi
