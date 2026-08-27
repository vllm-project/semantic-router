import { createServer, type Server } from 'node:http'
import { expect, test, type Page, type Route } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'
import { openComposerAddMenu } from './support/playground'

const mediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const sessionId = '20000000-0000-4000-8000-000000000001'
const profileId = '20000000-0000-4000-8000-000000000002'
const turnId = '20000000-0000-4000-8000-000000000003'
const planId = '20000000-0000-4000-8000-000000000004'
const entrypointId = 'blend-v2'
const artifactId = '20000000-0000-4000-8000-000000000006'
const inferenceKeyId = '10000000-0000-4000-8000-000000000003'
const digest = `sha256:${'a'.repeat(64)}`
const onePixelGif = Buffer.from('R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==', 'base64')

interface ChatRequestObservation {
  authorization: string
  body: Record<string, unknown>
  sessionId: string
}

const chatRequests: ChatRequestObservation[] = []
const deferredChatFinishes = new Map<string, () => void>()
let inferenceServer: Server
let inferenceOrigin = ''

function finalUserText(body: Record<string, unknown>): string {
  const messages = Array.isArray(body.messages) ? body.messages : []
  const message = messages.at(-1)
  if (!message || typeof message !== 'object' || Array.isArray(message)) return ''
  const content = (message as { content?: unknown }).content
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  return content
    .map((part) =>
      part && typeof part === 'object' && !Array.isArray(part) && 'text' in part
        ? String((part as { text?: unknown }).text ?? '')
        : '',
    )
    .join('')
}

test.beforeAll(async () => {
  inferenceServer = createServer((request, response) => {
    const origin = request.headers.origin ?? '*'
    response.setHeader('Access-Control-Allow-Origin', origin)
    response.setHeader(
      'Access-Control-Allow-Headers',
      'authorization, content-type, x-session-id, x-vsr-debug',
    )
    response.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.setHeader(
      'Access-Control-Expose-Headers',
      'x-request-id, x-vsr-selected-decision, x-vsr-selected-model',
    )
    response.setHeader('Vary', 'Origin')
    if (request.method === 'OPTIONS') {
      response.writeHead(204)
      response.end()
      return
    }
    if (request.method !== 'POST' || request.url !== '/v1/chat/completions') {
      response.writeHead(404)
      response.end()
      return
    }

    const chunks: Buffer[] = []
    request.on('data', (chunk: Buffer) => chunks.push(chunk))
    request.on('end', () => {
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as Record<string, unknown>
      chatRequests.push({
        authorization: request.headers.authorization ?? '',
        body,
        sessionId: String(request.headers['x-session-id'] ?? ''),
      })
      const prompt = finalUserText(body)
      const responseId = `chatcmpl-${chatRequests.length}`
      response.writeHead(200, {
        'Cache-Control': 'no-cache, no-transform',
        'Content-Type': 'text/event-stream; charset=utf-8',
        'x-request-id': `request-${chatRequests.length}`,
        'x-vsr-selected-decision': 'Simple',
        'x-vsr-selected-model': 'local/qwen',
      })
      response.write(
        `data: ${JSON.stringify({ id: responseId, model: 'local/qwen', choices: [{ index: 0, delta: { content: 'The model ' } }] })}\n\n`,
      )
      let finished = false
      const finish = () => {
        if (finished) return
        finished = true
        deferredChatFinishes.delete(prompt)
        response.write(
          `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'path is ready.' }, finish_reason: 'stop' }] })}\n\n`,
        )
        response.write(
          `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: Math.max(1, prompt.length), completion_tokens: 4, total_tokens: Math.max(1, prompt.length) + 4 } })}\n\n`,
        )
        response.end('data: [DONE]\n\n')
      }
      if (prompt.includes('Use my team key')) {
        deferredChatFinishes.set(prompt, finish)
      } else {
        setTimeout(finish, 25)
      }
    })
  })
  await new Promise<void>((resolve, reject) => {
    inferenceServer.once('error', reject)
    inferenceServer.listen(0, '127.0.0.1', resolve)
  })
  const address = inferenceServer.address()
  if (!address || typeof address === 'string')
    throw new Error('Inference test server did not bind.')
  inferenceOrigin = `http://127.0.0.1:${address.port}`
})

test.afterAll(async () => {
  await new Promise<void>((resolve, reject) => {
    inferenceServer.close((error) => (error ? reject(error) : resolve()))
  })
})

test.afterEach(() => {
  for (const finish of deferredChatFinishes.values()) finish()
  deferredChatFinishes.clear()
})

type AgentEvent = {
  sessionId: string
  turnId?: string
  sequence: number
  type: string
  createdAt: string
  payload: Record<string, unknown>
}

const now = '2026-08-23T00:00:00Z'

function session(mode: 'chat' | 'builder' = 'chat') {
  return {
    id: sessionId,
    namespaceId: '20000000-0000-4000-8000-000000000010',
    ownerPrincipalId: '20000000-0000-4000-8000-000000000011',
    keyId: inferenceKeyId,
    profileId,
    profileRevision: 1,
    target: { kind: 'entrypoint', id: 'blend' },
    mode,
    title: mode === 'builder' ? 'Build a support router' : 'Existing chat',
    status: 'active',
    revision: 1,
    createdAt: now,
    updatedAt: now,
  }
}

function userEvent(sequence: number, text: string, eventTurnId = turnId): AgentEvent {
  return {
    sessionId,
    turnId: eventTurnId,
    sequence,
    type: 'user_input',
    createdAt: now,
    payload: { content: [{ type: 'text', text }] },
  }
}

function assistantEvent(sequence: number, text: string, eventTurnId = turnId): AgentEvent {
  return {
    sessionId,
    turnId: eventTurnId,
    sequence,
    type: 'assistant_delta',
    createdAt: now,
    payload: {
      modelStepId: '83edbd6c-f8a3-4cab-935a-5ab28b52bd9f',
      chunkIndex: sequence - 1,
      delta: { kind: 'text', text },
    },
  }
}

function terminalEvent(sequence: number, eventTurnId = turnId): AgentEvent {
  return {
    sessionId,
    turnId: eventTurnId,
    sequence,
    type: 'terminal',
    createdAt: now,
    payload: { status: 'completed' },
  }
}

function sse(events: AgentEvent[]): string {
  if (!events.length) return ': ready\n\n'
  return events
    .map(
      (event) => `id: ${event.sequence}\nevent: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`,
    )
    .join('')
}

async function fulfillJSON(
  route: Route,
  body: unknown,
  headers: Record<string, string> = {},
  status = 200,
) {
  await route.fulfill({
    status,
    headers: { 'Content-Type': mediaType, ...headers },
    body: JSON.stringify(body),
  })
}

interface AgentMockOptions {
  initialEvents?: AgentEvent[]
  existingMode?: 'chat' | 'builder'
  deferFirstTurn?: boolean
  expireFirstResume?: boolean
  publishedModel?: boolean
  connectionApproval?: boolean
  connectionManagement?: boolean
  builtinSkill?: boolean
  singleModelsOnly?: boolean
}

async function mockAgentRuntime(page: Page, options: AgentMockOptions = {}) {
  let activeSession = options.initialEvents ? session(options.existingMode) : null
  let events = options.initialEvents ?? []
  let expired = false
  let published = options.publishedModel ?? false
  let modelReads = 0
  let publicationCommits = 0
  let artifactContentReads = 0
  let connectionApprovals = 0
  let connectionStatus: 'active' | 'disabled' = 'active'
  let deferredTurnText: string | null = null
  let deferredTurnId: string | null = null
  let connectionCredentialId: string | undefined = options.connectionManagement
    ? '50000000-0000-4000-8000-000000000001'
    : undefined
  const resumeHeaders: string[] = []
  const turnBodies: unknown[] = []
  const sessionBodies: unknown[] = []
  const connectionPatches: Array<Record<string, unknown>> = []

  await page.route('**/v1/models*', async (route) => {
    modelReads += 1
    const virtualModels = options.singleModelsOnly
      ? []
      : [
          {
            id: 'blend',
            description: 'Balanced model path',
            routing: {
              resolution: 'virtual',
              selectable: true,
              default_route: false,
              recipe: 'blend',
            },
          },
          ...(published
            ? [
                {
                  id: 'blend-v2',
                  description: 'Published model path',
                  routing: {
                    resolution: 'virtual',
                    selectable: true,
                    default_route: false,
                    recipe: 'blend-v2',
                  },
                },
              ]
            : []),
        ]
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        data: [
          ...virtualModels,
          {
            id: 'local/qwen',
            description: 'Connected model',
            routing: { resolution: 'passthrough', selectable: false, default_route: false },
          },
        ],
      }),
    })
  })

  await page.route('**/api/router/management/v1/routing/model-cards*', async (route) => {
    await fulfillJSON(route, {
      data: [
        {
          id: 'model_qwen',
          name: 'Qwen',
          card: {
            aliases: [],
            capabilities: ['text', 'tools', 'images'],
            loras: [],
            tags: [],
          },
        },
      ],
      page: { hasMore: false, pageSize: 100 },
    })
  })

  await page.route('**/api/router/management/v1/api-keys/*/routing-catalog', async (route) => {
    const keyId = new URL(route.request().url()).pathname.split('/').at(-2) ?? inferenceKeyId
    await fulfillJSON(route, {
      keyId,
      policyRevision: 1,
      policyDigest: 'a'.repeat(64),
      routingRevision: 1,
      routingDigest: 'b'.repeat(64),
      models: [
        {
          id: 'model_blend',
          revision: 1,
          name: 'local/qwen',
          aliases: [],
          capabilities: ['text'],
          loras: [],
          tags: [],
          pricing: {
            inputCostPerMillionTokens: null,
            outputCostPerMillionTokens: null,
            cacheReadCostPerMillionTokens: null,
            cacheWriteCostPerMillionTokens: null,
          },
        },
      ],
      recipes: [
        {
          id: 'recipe_blend',
          revision: 1,
          name: 'Blend',
          decisions: [{ id: 'decision_blend', name: 'Blend', dispatchCardinality: 'single' }],
        },
      ],
      entrypoints: [
        {
          id: 'entrypoint_blend',
          revision: 1,
          name: 'blend',
          aliases: ['blend'],
          rules: [
            {
              id: 'rule_blend',
              name: 'Default',
              recipeId: 'recipe_blend',
              recipeRevision: 1,
              assignments: {
                decision_blend: {
                  models: [
                    {
                      modelId: 'model_blend',
                      modelRevision: 1,
                      priority: 0,
                      weight: '1',
                    },
                  ],
                },
              },
            },
          ],
        },
      ],
    })
  })

  await page.route('**/api/router/management/v1/agent-sessions**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()
    const accept = request.headers().accept ?? ''

    if (path.endsWith('/events') && accept.includes('text/event-stream')) {
      const resume =
        request.headers()['last-event-id'] ?? url.searchParams.get('afterSequence') ?? ''
      resumeHeaders.push(resume)
      if (options.expireFirstResume && !expired) {
        expired = true
        events = [...events, assistantEvent(events.length + 1, 'Recovered')]
        await route.fulfill({
          status: 410,
          headers: { 'Content-Type': mediaType },
          body: JSON.stringify({
            error: { code: 'event_history_expired', message: 'History archived' },
            recovery: {
              checkpointId: 'checkpoint-1',
              throughSequence: events.at(-1)?.sequence ?? 0,
            },
          }),
        })
        return
      }
      const after = Number(resume || 0)
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
        body: sse(events.filter((event) => event.sequence > after)),
      })
      return
    }

    if (path.endsWith('/events') && method === 'GET') {
      await fulfillJSON(route, {
        data: events,
        page: { hasMore: false, pageSize: 100 },
      })
      return
    }

    if (path.endsWith('/turns') && method === 'POST') {
      const input = request.postDataJSON() as { input?: { content?: Array<{ text?: string }> } }
      turnBodies.push(input)
      const text = input.input?.content?.find((item) => item.text)?.text ?? 'Hello'
      const baseSequence = events.at(-1)?.sequence ?? 0
      const currentTurnId =
        turnBodies.length === 1
          ? turnId
          : `20000000-0000-4000-8000-${String(turnBodies.length + 11).padStart(12, '0')}`
      if (options.deferFirstTurn && turnBodies.length === 1) {
        deferredTurnText = text
        deferredTurnId = currentTurnId
        events = [...events, userEvent(baseSequence + 1, text, currentTurnId)]
        await fulfillJSON(
          route,
          { resource: { kind: 'agent-turn', id: currentTurnId, revision: 1 } },
          {},
          201,
        )
        return
      }
      const agentChat = activeSession?.mode === 'chat'
      const toolName = agentChat ? 'search_web' : 'routing.validate_recipe'
      events = [
        ...events,
        userEvent(baseSequence + 1, text, currentTurnId),
        {
          sessionId,
          turnId: currentTurnId,
          sequence: baseSequence + 2,
          type: 'tool_request',
          createdAt: now,
          payload: {
            invocationId: 'tool-1',
            toolName,
            arguments: agentChat ? { query: text } : { recipeId: 'blend' },
            class: 'read',
          },
        },
        {
          sessionId,
          turnId: currentTurnId,
          sequence: baseSequence + 3,
          type: 'tool_result',
          createdAt: now,
          payload: {
            invocationId: 'tool-1',
            toolName,
            status: 'completed',
            result: agentChat ? { results: [{ title: 'Current source' }] } : { recipeId: 'blend' },
            artifactId,
          },
        },
        assistantEvent(
          baseSequence + 4,
          agentChat ? 'I found a current source.' : 'The model path is ready.',
          currentTurnId,
        ),
        terminalEvent(baseSequence + 5, currentTurnId),
      ]
      await fulfillJSON(
        route,
        { resource: { kind: 'agent-turn', id: currentTurnId, revision: 1 } },
        {},
        201,
      )
      return
    }

    if (path.endsWith(':cancel') && method === 'POST') {
      await fulfillJSON(route, { resource: { kind: 'agent-turn', id: turnId, revision: 2 } })
      return
    }

    if (path.endsWith(`/${sessionId}`) && method === 'GET') {
      await fulfillJSON(route, { data: activeSession ?? session() }, { ETag: '"session-1"' })
      return
    }

    if (path.endsWith(`/${sessionId}`) && method === 'DELETE') {
      activeSession = null
      await route.fulfill({ status: 204 })
      return
    }

    if (path.endsWith('/agent-sessions') && method === 'POST') {
      const input = request.postDataJSON() as {
        keyId: string
        mode: 'chat' | 'builder'
        target: { kind: string; id: string }
        title?: string
      }
      sessionBodies.push(input)
      activeSession = {
        ...session(input.mode),
        keyId: input.keyId,
        target: input.target,
        title: input.title ?? 'New conversation',
      }
      await fulfillJSON(
        route,
        { resource: { kind: 'agent-session', id: sessionId, revision: 1 } },
        {},
        201,
      )
      return
    }

    if (path.endsWith('/agent-sessions') && method === 'GET') {
      const query = url.searchParams.get('search')?.toLowerCase() ?? ''
      const data =
        activeSession && (!query || activeSession.title.toLowerCase().includes(query))
          ? [activeSession]
          : []
      await fulfillJSON(route, { data, page: { hasMore: false, pageSize: 50 } })
      return
    }

    await route.fulfill({ status: 404 })
  })

  await page.route('**/api/router/management/v1/agent-skills**', async (route) => {
    const skill = {
      id: '30000000-0000-4000-8000-000000000010',
      name: 'Recipe designer',
      description: 'Build and validate model paths',
      status: 'active',
      revision: 1,
      contentRevision: 1,
      builtin: true,
      instructions: 'Design a recipe.',
      requiredTools: [],
      minimumCapabilities: [],
      contentDigest: digest,
      createdAt: now,
      updatedAt: now,
    }
    if (options.builtinSkill && new URL(route.request().url()).pathname.endsWith(`/${skill.id}`)) {
      await fulfillJSON(route, { data: skill }, { ETag: '"skill-1"' })
      return
    }
    await fulfillJSON(route, {
      data: options.builtinSkill ? [skill] : [],
      page: { hasMore: false, pageSize: 50 },
    })
  })
  await page.route('**/api/router/management/v1/agent-tools**', async (route) => {
    await fulfillJSON(route, {
      data: [],
      page: { hasMore: false, pageSize: 50 },
      registryRevision: digest,
    })
  })
  await page.route('**/api/router/management/v1/agent-tool-sources**', async (route) => {
    const request = route.request()
    const path = new URL(request.url()).pathname
    const source = {
      id: '40000000-0000-4000-8000-000000000001',
      namespaceId: '20000000-0000-4000-8000-000000000010',
      name: 'Knowledge tools',
      description: 'Search internal references',
      status: connectionStatus,
      revision: connectionApprovals + connectionPatches.length + 1,
      contentRevision: connectionApprovals + connectionPatches.length + 1,
      createdAt: now,
      updatedAt: now,
      kind: 'remote',
      transport: 'streamable_http',
      endpoint: 'https://tools.example.com/connect',
      ...(connectionCredentialId ? { credentialId: connectionCredentialId } : {}),
      egressPolicy: { allowedHosts: ['tools.example.com'], allowedPorts: [443] },
      discoveredTools: [
        {
          name: 'knowledge.search',
          description: 'Search references',
          inputSchema: { type: 'object' },
          outputSchema: { type: 'object' },
          requiredPermissions: ['tool.invoke'],
          class: 'read',
          idempotency: 'invocation',
          timeoutMilliseconds: 10_000,
        },
      ],
      discoveryDigest: digest,
      availability:
        connectionStatus === 'disabled'
          ? 'disabled'
          : connectionApprovals
            ? 'ready'
            : 'pending_approval',
      ...(connectionApprovals ? { approvedDiscoveryDigest: digest } : {}),
    }
    if (options.connectionApproval && path.endsWith(':approve')) {
      expect(request.headers()['if-match']).toBe('"source-1"')
      expect(request.headers()['idempotency-key']).toBeTruthy()
      expect(request.postDataJSON()).toEqual({ discoveryDigest: digest })
      connectionApprovals += 1
      await fulfillJSON(route, {
        resource: { kind: 'agent-tool-source', id: source.id, revision: 2 },
      })
      return
    }
    if (
      (options.connectionApproval || options.connectionManagement) &&
      path.endsWith(`/${source.id}`) &&
      request.method() === 'PATCH'
    ) {
      const patch = request.postDataJSON() as Record<string, unknown>
      connectionPatches.push(patch)
      if (Object.prototype.hasOwnProperty.call(patch, 'credentialId')) {
        connectionCredentialId =
          typeof patch.credentialId === 'string' ? patch.credentialId : undefined
      }
      if (patch.status === 'active' || patch.status === 'disabled') connectionStatus = patch.status
      await fulfillJSON(route, {
        resource: {
          kind: 'agent-tool-source',
          id: source.id,
          revision: connectionPatches.length + 1,
        },
      })
      return
    }
    if (
      (options.connectionApproval || options.connectionManagement) &&
      path.endsWith(`/${source.id}`)
    ) {
      await fulfillJSON(route, { data: source }, { ETag: '"source-1"' })
      return
    }
    await fulfillJSON(route, {
      data: options.connectionApproval || options.connectionManagement ? [source] : [],
      page: { hasMore: false, pageSize: 50 },
    })
  })
  await page.route('**/api/router/management/v1/agent-tool-credentials**', async (route) => {
    const credential = {
      id: '50000000-0000-4000-8000-000000000001',
      namespaceId: '20000000-0000-4000-8000-000000000010',
      name: 'Knowledge token',
      status: 'active',
      revision: 1,
      createdAt: now,
      updatedAt: now,
    }
    const path = new URL(route.request().url()).pathname
    if (path.endsWith(`/${credential.id}`)) {
      await fulfillJSON(route, { data: credential }, { ETag: '"credential-1"' })
      return
    }
    await fulfillJSON(route, {
      data: options.connectionManagement ? [credential] : [],
      page: { hasMore: false, pageSize: 50 },
    })
  })
  await page.route('**/api/router/management/v1/agent-artifacts/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.endsWith('/content')) {
      artifactContentReads += 1
      await fulfillJSON(route, {
        data: {
          id: artifactId,
          mediaType: 'application/json',
          encoding: 'base64',
          content: Buffer.from(JSON.stringify({ score: 0.97 })).toString('base64'),
          digest,
        },
      })
      return
    }
    await fulfillJSON(route, {
      data: {
        id: artifactId,
        sessionId,
        turnId,
        kind: 'probe_result',
        mediaType: 'application/json',
        digest,
        safePreview: { status: 'Passed', score: 0.97 },
        expiresAt: '2099-08-23T00:00:00Z',
        createdAt: now,
      },
    })
  })

  await page.route(
    `**/api/router/management/v1/publication-plans/${planId}:commit`,
    async (route) => {
      publicationCommits += 1
      expect(route.request().headers()['if-match']).toBe('"opaque-plan-7"')
      expect(route.request().headers()['idempotency-key']).toBeTruthy()
      published = true
      await fulfillJSON(route, { operation: { operationId: 'publish-operation-1' } }, {}, 202)
    },
  )
  await page.route('**/api/router/management/v1/operations/publish-operation-1', async (route) => {
    await fulfillJSON(route, {
      operationId: 'publish-operation-1',
      kind: 'routing.publish',
      state: 'succeeded',
      progress: { total: '1', completed: '1', failed: '0' },
      revisions: { desiredRevision: 2, appliedRevision: 2 },
      createdAt: now,
      updatedAt: now,
    })
  })
  await page.route(
    `**/api/router/management/v1/routing/entrypoints/${entrypointId}`,
    async (route) => {
      await fulfillJSON(route, {
        data: {
          id: entrypointId,
          name: 'blend-v2',
          status: 'active',
          revision: 1,
          entrypointRevision: 1,
          aliases: [],
          recipeIds: ['recipe-agent'],
          ruleCount: 1,
          assignedModelCount: 2,
          createdAt: now,
          updatedAt: now,
        },
      })
    },
  )

  return {
    modelReads: () => modelReads,
    publicationCommits: () => publicationCommits,
    artifactContentReads: () => artifactContentReads,
    connectionApprovals: () => connectionApprovals,
    connectionPatches,
    resumeHeaders,
    turnBodies,
    sessionBodies,
    finishDeferredTurn: () => {
      if (deferredTurnText === null || deferredTurnId === null) return
      const baseSequence = events.at(-1)?.sequence ?? 0
      events = [
        ...events,
        assistantEvent(baseSequence + 1, `Finished ${deferredTurnText}.`, deferredTurnId),
        terminalEvent(baseSequence + 2, deferredTurnId),
      ]
      deferredTurnText = null
      deferredTurnId = null
    },
  }
}

async function bootstrap(page: Page, options: AgentMockOptions = {}) {
  await mockAuthenticatedAppShell(page, { settings: { routerPublicUrl: inferenceOrigin } })
  return mockAgentRuntime(page, options)
}

async function enableBuilderMode(page: Page) {
  const menu = await openComposerAddMenu(page)
  await menu.getByRole('menuitemcheckbox', { name: /Builder/ }).click()
}

async function openConversationSidebar(page: Page) {
  const sidebar = page.getByTestId('agent-conversation-sidebar')
  await sidebar.getByRole('button', { name: 'Open conversations' }).click()
}

test.describe('Router Agent Playground', () => {
  test('groups authorized Mixture-of-Models and Single Models for an operator', async ({
    page,
  }) => {
    await bootstrap(page)
    await page.goto('/playground')

    await page.getByTestId('playground-composer-model-select').click()
    await expect(page.getByText('Mixture-of-Models', { exact: true })).toBeVisible()
    await expect(page.getByText('Single Model', { exact: true })).toBeVisible()
    await expect(page.getByRole('option', { name: /blend/ })).toBeVisible()
    await expect(page.getByRole('option', { name: /local\/qwen/ })).toBeVisible()
  })

  test('keeps an authorized Single Model usable before the first mixture is published', async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page)
    await mockAgentRuntime(page, { singleModelsOnly: true })
    await page.goto('/playground')

    await expect(page.getByText('Models are unavailable.', { exact: true })).toHaveCount(0)
    await page.getByTestId('playground-composer-model-select').click()
    await expect(page.getByText('Mixture-of-Models', { exact: true })).toHaveCount(0)
    await expect(page.getByText('Single Model', { exact: true })).toBeVisible()
    await expect(page.getByRole('option', { name: /local\/qwen/ })).toBeVisible()
  })

  test('uses one selected API key for a durable Chat turn', async ({ page }) => {
    const requestsBefore = chatRequests.length
    const secondKeyId = '10000000-0000-4000-8000-000000000009'
    await mockAuthenticatedAppShell(page, {
      settings: { routerPublicUrl: inferenceOrigin },
      user: {
        id: 'consumer-1',
        email: 'consumer@example.com',
        name: 'Consumer',
        role: 'read',
        permissions: [],
      },
      managementPermissions: [
        'access_policy.read',
        'agent.read',
        'agent.use',
        'delegation.use',
        'key.read',
        'routing_context.read',
        'tool.invoke',
        'tool.read',
      ],
    })
    const mock = await mockAgentRuntime(page)
    const issuedFor: string[] = []
    const revokedSessions: string[] = []
    const catalogReads: string[] = []
    const modelAuthorizations: string[] = []

    await page.route('**/api/router/management/v1/self/inference-keys*', async (route) => {
      await fulfillJSON(route, {
        data: [
          {
            keyId: inferenceKeyId,
            name: 'Personal',
            owner: { type: 'user', id: '10000000-0000-4000-8000-000000000011' },
          },
          {
            keyId: secondKeyId,
            name: 'Team',
            owner: { type: 'team', id: '10000000-0000-4000-8000-000000000012' },
            contextTeamId: '10000000-0000-4000-8000-000000000012',
          },
        ],
        page: { hasMore: false, pageSize: 100 },
      })
    })
    await page.route('**/api/router/management/v1/self/inference-sessions', async (route) => {
      const { keyId } = route.request().postDataJSON() as { keyId: string }
      issuedFor.push(keyId)
      await fulfillJSON(
        route,
        {
          resourceId: `session-${keyId}`,
          kind: 'delegated_inference_credential',
          secret: keyId === secondKeyId ? 'vsd_key-b' : 'vsd_key-a',
          expiresAt: '2099-08-23T00:00:00Z',
        },
        {},
        201,
      )
    })
    await page.route('**/api/router/management/v1/self/inference-sessions/*', async (route) => {
      revokedSessions.push(new URL(route.request().url()).pathname.split('/').at(-1) ?? '')
      await route.fulfill({ status: 204 })
    })
    await page.route('**/api/router/management/v1/api-keys/*/routing-catalog', async (route) => {
      const keyId = new URL(route.request().url()).pathname.split('/').at(-2) ?? ''
      catalogReads.push(keyId)
      const suffix = keyId === secondKeyId ? 'b' : 'a'
      await fulfillJSON(route, {
        keyId,
        policyRevision: 1,
        policyDigest: 'a'.repeat(64),
        routingRevision: 1,
        routingDigest: 'b'.repeat(64),
        models: [
          {
            id: `model_${suffix}`,
            revision: 1,
            name: `local/${suffix}`,
            aliases: [],
            capabilities: ['text'],
            loras: [],
            tags: [],
            pricing: {
              inputCostPerMillionTokens: null,
              outputCostPerMillionTokens: null,
              cacheReadCostPerMillionTokens: null,
              cacheWriteCostPerMillionTokens: null,
            },
          },
        ],
        recipes: [
          {
            id: `recipe_${suffix}`,
            revision: 1,
            name: `Recipe ${suffix.toUpperCase()}`,
            decisions: [{ id: `decision_${suffix}`, name: 'Route', dispatchCardinality: 'single' }],
          },
        ],
        entrypoints: [
          {
            id: `entrypoint_${suffix}`,
            revision: 1,
            name: `blend-${suffix}`,
            aliases: [`blend-${suffix}`],
            rules: [
              {
                id: `rule_${suffix}`,
                name: 'Default',
                recipeId: `recipe_${suffix}`,
                recipeRevision: 1,
                assignments: {
                  [`decision_${suffix}`]: {
                    models: [
                      {
                        modelId: `model_${suffix}`,
                        modelRevision: 1,
                        priority: 0,
                        weight: '1',
                      },
                    ],
                  },
                },
              },
            ],
          },
        ],
      })
    })
    await page.route('**/v1/models*', async (route) => {
      const authorization = route.request().headers().authorization ?? ''
      modelAuthorizations.push(authorization)
      const suffix = authorization.includes('vsd_key-b') ? 'b' : 'a'
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: [
            {
              id: `blend-${suffix}`,
              description: `Key ${suffix.toUpperCase()} model`,
              routing: {
                resolution: 'virtual',
                selectable: true,
                default_route: false,
                recipe: `recipe_${suffix}`,
              },
            },
          ],
        }),
      })
    })

    await page.goto('/playground')
    const keySelect = page.getByRole('combobox', { name: 'Use' })
    await expect(keySelect).toHaveValue('Personal')
    await expect.poll(() => catalogReads).toContain(inferenceKeyId)
    await expect.poll(() => issuedFor).toContain(inferenceKeyId)

    await keySelect.click()
    await page.getByRole('option', { name: /Team/ }).click()
    await expect(keySelect).toHaveValue('Team')
    await expect.poll(() => catalogReads).toContain(secondKeyId)
    await expect.poll(() => issuedFor).toContain(secondKeyId)
    await expect.poll(() => revokedSessions).toContain(`session-${inferenceKeyId}`)
    await expect.poll(() => modelAuthorizations).toContain('Bearer vsd_key-b')
    await page.getByTestId('playground-composer-model-select').click()
    await expect(page.getByRole('option', { name: /blend-b/ })).toBeVisible()
    await page.keyboard.press('Escape')

    await page.getByRole('textbox', { name: 'Message' }).fill('Use my team key')
    await page.getByRole('button', { name: 'Send message' }).click()
    const assistant = page.getByTestId('agent-message-assistant')
    await expect(assistant).toContainText('I found a current source.')
    await expect.poll(() => mock.sessionBodies.length).toBe(1)
    expect(mock.sessionBodies[0]).toMatchObject({ keyId: secondKeyId, mode: 'chat' })
    await expect.poll(() => mock.turnBodies.length).toBe(1)
    expect(mock.turnBodies[0]).toMatchObject({
      input: { content: [{ type: 'text', text: 'Use my team key' }] },
    })
    expect(chatRequests.slice(requestsBefore)).toHaveLength(0)

    await openConversationSidebar(page)
    const localConversation = page
      .getByTestId('agent-conversation-sidebar')
      .getByRole('button', { name: /^Use my team key\b/ })
    await expect(localConversation).toBeVisible()
    await page
      .getByTestId('agent-conversation-sidebar')
      .getByRole('button', { name: 'New chat' })
      .click()
    await expect(page.getByTestId('agent-message-assistant')).toHaveCount(0)
    await localConversation.click()
    await expect(page.getByTestId('agent-message-assistant')).toContainText(
      'I found a current source.',
    )
    expect(mock.sessionBodies).toHaveLength(1)
    expect(mock.turnBodies).toHaveLength(1)
  })

  test('queues a compact follow-up and sends it after the active stream completes', async ({
    page,
  }) => {
    const activePrompt = 'Use my team key while I prepare a follow-up'
    const queuedPrompt = 'Then compare the selected route with the fastest option'
    const mock = await bootstrap(page, { deferFirstTurn: true })
    await page.goto('/playground')

    const composer = page.getByRole('textbox', { name: 'Message' })
    await composer.fill(activePrompt)
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect.poll(() => mock.turnBodies.length).toBe(1)

    await expect(composer).toBeEnabled()
    await composer.fill(queuedPrompt)
    await page.getByRole('button', { name: 'Queue' }).click()
    const queue = page.getByRole('region', { name: 'Queued messages' })
    await expect(queue).toContainText(queuedPrompt)

    mock.finishDeferredTurn()
    await expect.poll(() => mock.turnBodies.length).toBe(2)
    expect(mock.turnBodies[1]).toMatchObject({
      input: { content: [{ type: 'text', text: queuedPrompt }] },
    })
    await expect(queue).toHaveCount(0)
    await expect(page.getByTestId('agent-message-user')).toHaveCount(2)
    await expect(page.getByTestId('agent-message-assistant')).toContainText(
      'I found a current source.',
    )
  })

  test('streams a durable turn and collapses completed tool work', async ({ page }) => {
    const mock = await bootstrap(page)
    await page.goto('/playground')
    await enableBuilderMode(page)

    await page.getByRole('textbox', { name: 'Builder instruction' }).fill('Design a support route')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(page.getByTestId('agent-message-user')).toContainText('Design a support route')
    await expect(page.getByTestId('agent-message-assistant')).toContainText(
      'The model path is ready.',
    )
    const toolRow = page.getByRole('button', { name: /Validate Recipe/ })
    await expect(toolRow).toContainText('Done')
    await toolRow.click()
    await page.getByRole('button', { name: 'View result' }).click()
    await expect(page.getByText('Passed', { exact: false })).toBeVisible()
    expect(mock.artifactContentReads()).toBe(0)
    await page.getByRole('button', { name: 'Load original' }).click()
    await expect(page.getByText('"score":0.97', { exact: false })).toBeVisible()
    expect(mock.artifactContentReads()).toBe(1)
  })

  test('runs an authorized web-search tool from Chat without a separate Agent mode', async ({
    page,
  }) => {
    const mock = await bootstrap(page)
    await page.goto('/playground')

    const menu = await openComposerAddMenu(page)
    await expect(menu.getByRole('menuitemcheckbox', { name: /Agent/ })).toHaveCount(0)
    await page.keyboard.press('Escape')

    await page.getByRole('textbox', { name: 'Message' }).fill('Find the latest release')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect.poll(() => mock.sessionBodies.length).toBe(1)
    expect(mock.sessionBodies[0]).toMatchObject({ mode: 'chat' })
    await expect(page.getByRole('button', { name: /Search Web/ })).toContainText('Done')
    await expect(page.getByTestId('agent-message-assistant')).toContainText(
      'I found a current source.',
    )
  })

  test('sends an image as durable Agent content instead of a browser-owned chat request', async ({
    page,
  }) => {
    const mock = await bootstrap(page)
    await page.goto('/playground')
    await enableBuilderMode(page)

    const menu = await openComposerAddMenu(page)
    const chooserPromise = page.waitForEvent('filechooser')
    await menu.getByRole('menuitem', { name: /Attach files/ }).click()
    const chooser = await chooserPromise
    await chooser.setFiles({ name: 'vision.gif', mimeType: 'image/gif', buffer: onePixelGif })
    await expect(page.getByAltText('Preview of vision.gif')).toBeVisible()
    await page.getByRole('textbox', { name: 'Builder instruction' }).fill('What is visible?')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect.poll(() => mock.turnBodies.length).toBe(1)
    expect(mock.turnBodies[0]).toMatchObject({
      input: {
        content: [
          { type: 'text', text: 'What is visible?' },
          {
            type: 'image_url',
            url: expect.stringMatching(/^data:image\/gif;base64,/),
            detail: 'auto',
          },
        ],
      },
    })
  })

  test('resumes after archived history and keeps search from changing the active session', async ({
    page,
  }) => {
    const mock = await bootstrap(page, {
      initialEvents: [userEvent(1, 'Earlier question'), assistantEvent(2, 'Earlier answer')],
      expireFirstResume: true,
      existingMode: 'builder',
    })
    await page.goto('/playground')
    await openConversationSidebar(page)
    await page.getByRole('button', { name: /^Build a support router\b/ }).click()

    await expect(page.getByRole('alert')).toContainText('Earlier events were archived')
    await expect(page.getByTestId('agent-message-assistant')).toContainText('Recovered')
    await expect.poll(() => mock.resumeHeaders).toContain('3')

    await page.getByRole('searchbox', { name: 'Search conversations' }).fill('no match')
    await expect(page.getByRole('textbox', { name: 'Builder instruction' })).toBeVisible()
    await expect(page.getByTestId('agent-message-assistant')).toContainText('Recovered')
  })

  test('keeps publication separate and proves the model through Entrypoint and model discovery', async ({
    page,
  }) => {
    const approval: AgentEvent = {
      sessionId,
      turnId,
      sequence: 1,
      type: 'approval_request',
      createdAt: now,
      payload: {
        planId,
        planDigest: digest,
        planRevision: 7,
        planEtag: '"opaque-plan-7"',
        expiresAt: '2099-08-23T00:00:00Z',
        summary: {
          recipeName: 'Support Blend',
          entrypointId,
          entrypointName: 'blend-v2',
          changedResources: ['Recipe', 'Mixture-of-Models'],
        },
      },
    }
    const mock = await bootstrap(page, { initialEvents: [approval], existingMode: 'builder' })
    await page.goto('/playground')
    await openConversationSidebar(page)
    await page.getByRole('button', { name: /^Build a support router\b/ }).click()

    await expect(page.getByRole('dialog', { name: 'Publish blend-v2' })).toBeVisible()
    expect(mock.publicationCommits()).toBe(0)
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog', { name: 'Publish blend-v2' })).toBeHidden()
    expect(mock.publicationCommits()).toBe(0)
    await page.getByRole('button', { name: 'Review' }).click()
    const readsBefore = mock.modelReads()
    await page.getByTestId('agent-publish-confirm').click()
    await expect(page.getByRole('dialog', { name: 'Publish blend-v2' })).toBeHidden()
    expect(mock.publicationCommits()).toBe(1)
    expect(mock.modelReads()).toBeGreaterThan(readsBefore)

    await page.getByRole('button', { name: 'New chat' }).click()
    await page.getByTestId('playground-composer-model-select').click()
    await expect(page.getByRole('option', { name: /blend-v2/ })).toBeVisible()
  })

  test('hides Builder but shows Router-authorized Single Models without routing manage permission', async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'consumer-1',
        email: 'consumer@example.com',
        name: 'Consumer',
        role: 'viewer',
        permissions: ['config.read'],
      },
      managementPermissions: [
        'agent.read',
        'agent.use',
        'access_policy.read',
        'delegation.use',
        'key.read',
        'routing_context.read',
        'tool.read',
        'tool.invoke',
      ],
    })
    await mockAgentRuntime(page)
    await page.goto('/playground')

    const addMenu = await openComposerAddMenu(page)
    await expect(addMenu.getByRole('menuitemcheckbox', { name: /Builder/ })).toHaveCount(0)
    await expect(addMenu.getByRole('menuitemcheckbox', { name: /Agent/ })).toHaveCount(0)
    await page.keyboard.press('Escape')
    await page.getByTestId('playground-composer-model-select').click()
    await expect(page.getByText('Single Model', { exact: true })).toBeVisible()
    await expect(page.getByRole('option', { name: /local\/qwen/ })).toBeVisible()

    await page.goto('/config/agent')
    await expect(page).toHaveURL(/\/dashboard$/)
  })

  test('keeps standard Playground available without Agent or Tool read access', async ({
    page,
  }) => {
    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'limited-1',
        email: 'limited@example.com',
        name: 'Limited user',
        role: 'viewer',
        permissions: [],
      },
      managementPermissions: ['delegation.use', 'routing.read'],
    })
    await mockAgentRuntime(page)
    await page.goto('/config/agent')

    await expect(page).toHaveURL(/\/dashboard$/)
    await page.goto('/playground/fullscreen')
    await expect(page).toHaveURL(/\/playground\/fullscreen$/)
    await expect(page.getByTestId('agent-playground')).toBeVisible()
  })

  test('keeps Agent configuration focused on skills, tools, and connections', async ({ page }) => {
    await bootstrap(page)
    await page.goto('/config/agent')

    await expect(page.getByRole('button', { name: 'Skills' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Tools' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Connections' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Profiles' })).toHaveCount(0)
  })

  test('requires explicit approval when a Connection discovers tools', async ({ page }) => {
    const mock = await bootstrap(page, { connectionApproval: true })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'Connections' }).click()
    await page.getByRole('button', { name: 'Open Knowledge tools' }).click()
    const detail = page.getByRole('dialog', { name: 'Knowledge tools' })
    await expect(detail.getByText('Pending approval', { exact: true })).toBeVisible()
    await detail.getByRole('button', { name: 'Approve tools' }).click()
    const confirmation = page.getByRole('alertdialog', { name: 'Approve discovered tools?' })
    await expect(confirmation).toBeVisible()
    await confirmation.getByRole('button', { name: 'Approve tools' }).click()
    await expect(confirmation).toBeHidden()
    await expect(detail.getByText('Ready', { exact: true })).toBeVisible()
    expect(mock.connectionApprovals()).toBe(1)
  })

  test('keeps built-in Skills read-only', async ({ page }) => {
    await bootstrap(page, { builtinSkill: true })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'Skills' }).click()
    await page.getByRole('button', { name: 'Open Recipe designer' }).click()
    const detail = page.getByRole('dialog', { name: 'Recipe designer' })
    await expect(detail.getByText('Built in', { exact: true })).toBeVisible()
    await expect(detail.getByRole('button', { name: 'Edit' })).toHaveCount(0)
    await expect(detail.getByRole('button', { name: /Delete/ })).toHaveCount(0)
  })

  test('clears a Connection credential explicitly and supports disabling it', async ({ page }) => {
    const mock = await bootstrap(page, { connectionManagement: true })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'Connections' }).click()
    await page.getByRole('button', { name: 'Open Knowledge tools' }).click()
    const detail = page.getByRole('dialog', { name: 'Knowledge tools' })
    await detail.getByRole('button', { name: 'Edit' }).click()
    await detail.getByRole('checkbox', { name: /Knowledge token/ }).uncheck()
    await detail.getByRole('button', { name: 'Save connection' }).click()

    await expect.poll(() => mock.connectionPatches.length).toBe(1)
    expect(mock.connectionPatches[0]).toMatchObject({ credentialId: null })
    await detail.getByRole('button', { name: 'Disable' }).click()
    await expect.poll(() => mock.connectionPatches.length).toBe(2)
    expect(mock.connectionPatches[1]).toEqual({ status: 'disabled' })
    await expect(detail.getByRole('button', { name: 'Enable' })).toBeVisible()
  })

  test('rejects unsafe Connection URLs before saving', async ({ page }) => {
    const mock = await bootstrap(page, { connectionManagement: true })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'Connections' }).click()
    await page.getByRole('button', { name: 'Open Knowledge tools' }).click()
    const detail = page.getByRole('dialog', { name: 'Knowledge tools' })
    await detail.getByRole('button', { name: 'Edit' }).click()
    await detail
      .getByRole('textbox', { name: /HTTPS endpoint/ })
      .fill('https://tools.example.com/connect?token=secret')
    await detail.getByRole('button', { name: 'Save connection' }).click()

    await expect(detail.getByRole('alert')).toContainText(
      'Use an HTTPS URL without credentials, query parameters, or fragments.',
    )
    expect(mock.connectionPatches).toHaveLength(0)
  })

  test('keeps Connection mutations hidden for a read-only Tool user', async ({ page }) => {
    await mockAuthenticatedAppShell(page, {
      user: {
        id: 'tool-reader-1',
        email: 'reader@example.com',
        name: 'Tool reader',
        role: 'viewer',
        permissions: [],
      },
      managementPermissions: ['tool.read'],
    })
    await mockAgentRuntime(page, { connectionManagement: true })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'Connections' }).click()
    await page.getByRole('button', { name: 'Open Knowledge tools' }).click()
    const detail = page.getByRole('dialog', { name: 'Knowledge tools' })
    await expect(detail.getByRole('button', { name: 'Edit' })).toHaveCount(0)
    await expect(detail.getByRole('button', { name: 'Disable' })).toHaveCount(0)
    await expect(detail.getByRole('button', { name: 'Test connection' })).toHaveCount(0)
    await expect(detail.getByRole('button', { name: 'Approve tools' })).toHaveCount(0)
    await expect(detail.getByRole('button', { name: /Delete/ })).toHaveCount(0)
  })

  test('supports keyboard menus and removes sidebar controls when collapsed on mobile', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await bootstrap(page)
    await page.goto('/playground')

    await expect(page.getByRole('searchbox', { name: 'Search conversations' })).toHaveCount(0)
    await page.getByRole('button', { name: 'Open conversations' }).click()
    await expect(page.getByRole('searchbox', { name: 'Search conversations' })).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('searchbox', { name: 'Search conversations' })).toHaveCount(0)

    await page.getByTestId('playground-composer-add').focus()
    await page.keyboard.press('Enter')
    await expect(page.getByRole('menuitem', { name: /Attach files/ })).toBeFocused()
    await page.keyboard.press('ArrowDown')
    await expect(page.getByRole('menuitemcheckbox', { name: /Builder/ })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('playground-composer-add')).toBeFocused()
  })
})
