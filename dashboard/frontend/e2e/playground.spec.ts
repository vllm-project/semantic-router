import { expect, test, type Page, type Route } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'
import { openComposerAddMenu } from './support/playground'

const mediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const sessionId = '20000000-0000-4000-8000-000000000001'
const profileId = '20000000-0000-4000-8000-000000000002'
const turnId = '20000000-0000-4000-8000-000000000003'
const planId = '20000000-0000-4000-8000-000000000004'
const entrypointId = '20000000-0000-4000-8000-000000000005'
const artifactId = '20000000-0000-4000-8000-000000000006'
const digest = `sha256:${'a'.repeat(64)}`
const onePixelGif = Buffer.from('R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==', 'base64')

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

function userEvent(sequence: number, text: string): AgentEvent {
  return {
    sessionId,
    turnId,
    sequence,
    type: 'user_input',
    createdAt: now,
    payload: { content: [{ type: 'text', text }] },
  }
}

function assistantEvent(sequence: number, text: string): AgentEvent {
  return {
    sessionId,
    turnId,
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

function terminalEvent(sequence: number): AgentEvent {
  return {
    sessionId,
    turnId,
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

async function fulfillJSON(route: Route, body: unknown, headers: Record<string, string> = {}) {
  await route.fulfill({
    status: 200,
    headers: { 'Content-Type': mediaType, ...headers },
    body: JSON.stringify(body),
  })
}

interface AgentMockOptions {
  initialEvents?: AgentEvent[]
  existingMode?: 'chat' | 'builder'
  expireFirstResume?: boolean
  publishedModel?: boolean
  profilePages?: boolean
  profileSearchRace?: boolean
  connectionApproval?: boolean
  connectionManagement?: boolean
  capabilityLoadFailure?: boolean
  profileCapability?: string
  builtinSkill?: boolean
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
  let connectionCredentialId: string | undefined = options.connectionManagement
    ? '50000000-0000-4000-8000-000000000001'
    : undefined
  const resumeHeaders: string[] = []
  const turnBodies: unknown[] = []
  const profileQueries: string[] = []
  const connectionPatches: Array<Record<string, unknown>> = []

  await page.route('**/v1/models*', async (route) => {
    modelReads += 1
    const virtualModels = [
      {
        id: 'blend',
        description: 'Balanced model path',
        routing: { resolution: 'virtual', selectable: true, default_route: false, recipe: 'blend' },
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
            routing: { resolution: 'passthrough', selectable: true, default_route: false },
          },
        ],
      }),
    })
  })

  await page.route('**/api/router/management/v1/routing/model-cards*', async (route) => {
    if (options.capabilityLoadFailure) {
      await route.fulfill({
        status: 503,
        headers: { 'Content-Type': mediaType },
        body: JSON.stringify({
          error: { code: 'catalog_unavailable', message: 'Model capabilities are unavailable.' },
        }),
      })
      return
    }
    await fulfillJSON(route, {
      data: [
        {
          id: '60000000-0000-4000-8000-000000000001',
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
      events = [
        userEvent(1, text),
        {
          sessionId,
          turnId,
          sequence: 2,
          type: 'tool_request',
          createdAt: now,
          payload: {
            invocationId: 'tool-1',
            toolName: 'routing.validate_recipe',
            arguments: { recipeId: 'blend' },
            class: 'read',
          },
        },
        {
          sessionId,
          turnId,
          sequence: 3,
          type: 'tool_result',
          createdAt: now,
          payload: {
            invocationId: 'tool-1',
            toolName: 'routing.validate_recipe',
            status: 'completed',
            result: { recipeId: 'blend' },
            artifactId,
          },
        },
        assistantEvent(4, 'The model path is ready.'),
        terminalEvent(5),
      ]
      await fulfillJSON(route, { resource: { kind: 'agent-turn', id: turnId, revision: 1 } })
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
        mode: 'chat' | 'builder'
        target: { kind: string; id: string }
        title?: string
      }
      activeSession = {
        ...session(input.mode),
        target: input.target,
        title: input.title ?? 'New conversation',
      }
      await fulfillJSON(route, { resource: { kind: 'agent-session', id: sessionId, revision: 1 } })
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

  await page.route('**/api/router/management/v1/agent-profiles*', async (route) => {
    const url = new URL(route.request().url())
    profileQueries.push(url.search)
    const profiles = ['General', 'Builder'].map((name, index) => ({
      id: `30000000-0000-4000-8000-00000000000${index + 1}`,
      namespaceId: '20000000-0000-4000-8000-000000000010',
      name,
      status: 'active',
      revision: 1,
      contentRevision: 1,
      createdAt: now,
      updatedAt: now,
      minimumTargetCapabilities: options.profileCapability ? [options.profileCapability] : [],
      supportedModes: index === 0 ? ['chat'] : ['builder'],
      defaultForModes: [],
      skills: [],
      toolPolicy: { allow: [] },
      approvalPolicy: 'required',
      maximumTurnSeconds: 900,
      maximumToolSteps: 24,
      contextTokenBudget: 32768,
    }))
    const detail = profiles.find((profile) => url.pathname.endsWith(`/${profile.id}`))
    if (detail) {
      await fulfillJSON(route, { data: detail }, { ETag: '"profile-1"' })
      return
    }
    const search = url.searchParams.get('search')?.toLowerCase()
    if (options.profileSearchRace && search === 'general') {
      await new Promise((resolve) => setTimeout(resolve, 450))
    }
    const cursor = url.searchParams.get('cursor')
    const data = search
      ? profiles.filter((profile) => profile.name.toLowerCase().includes(search))
      : options.profilePages
        ? [profiles[cursor ? 1 : 0]]
        : []
    await fulfillJSON(route, {
      data,
      page: {
        hasMore: Boolean(options.profilePages && !search && !cursor),
        ...(options.profilePages && !search && !cursor ? { nextCursor: 'profiles-page-2' } : {}),
        pageSize: 50,
      },
    })
  })
  await page.route('**/api/router/management/v1/agent-skills*', async (route) => {
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
  await page.route('**/api/router/management/v1/agent-tools*', async (route) => {
    await fulfillJSON(route, {
      data: [],
      page: { hasMore: false, pageSize: 50 },
      registryRevision: digest,
    })
  })
  await page.route('**/api/router/management/v1/agent-tool-sources*', async (route) => {
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
  await page.route('**/api/router/management/v1/agent-tool-credentials*', async (route) => {
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
      await fulfillJSON(route, {
        operation: { operationId: 'publish-operation-1', state: 'pending' },
      })
    },
  )
  await page.route('**/api/router/management/v1/operations/publish-operation-1', async (route) => {
    await fulfillJSON(route, { data: { operationId: 'publish-operation-1', state: 'succeeded' } })
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
    profileQueries,
  }
}

async function bootstrap(page: Page, options: AgentMockOptions = {}) {
  await mockAuthenticatedAppShell(page)
  return mockAgentRuntime(page, options)
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

  test('streams a durable turn and collapses completed tool work', async ({ page }) => {
    const mock = await bootstrap(page)
    await page.goto('/playground')

    await page.getByRole('textbox', { name: 'Message' }).fill('Design a support route')
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

  test('sends an image as durable Agent content instead of a browser-owned chat request', async ({
    page,
  }) => {
    const mock = await bootstrap(page)
    await page.goto('/playground')

    const menu = await openComposerAddMenu(page)
    const chooserPromise = page.waitForEvent('filechooser')
    await menu.getByRole('menuitem', { name: /Attach files/ }).click()
    const chooser = await chooserPromise
    await chooser.setFiles({ name: 'vision.gif', mimeType: 'image/gif', buffer: onePixelGif })
    await expect(page.getByAltText('Preview of vision.gif')).toBeVisible()
    await page.getByRole('textbox', { name: 'Message' }).fill('What is visible?')
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
    })
    await page.goto('/playground')
    await page.getByRole('button', { name: /Existing chat/ }).click()

    await expect(page.getByRole('alert')).toContainText('Earlier events were archived')
    await expect(page.getByTestId('agent-message-assistant')).toContainText('Recovered')
    await expect.poll(() => mock.resumeHeaders).toContain('3')

    await page.getByRole('searchbox', { name: 'Search conversations' }).fill('no match')
    await expect(
      page
        .getByTestId('agent-playground')
        .locator('header')
        .first()
        .getByText('Existing chat', { exact: true }),
    ).toBeVisible()
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
    await page.getByRole('button', { name: /Build a support router/ }).click()

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

  test('hides Builder and Single Models without routing manage permission', async ({ page }) => {
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
        'delegation.use',
        'routing.read',
        'tool.read',
        'tool.invoke',
      ],
    })
    await mockAgentRuntime(page, { profilePages: true })
    await page.goto('/playground')

    const addMenu = await openComposerAddMenu(page)
    await expect(addMenu.getByRole('menuitemcheckbox', { name: /Builder/ })).toHaveCount(0)
    await page.keyboard.press('Escape')
    await page.getByTestId('playground-composer-model-select').click()
    await expect(page.getByText('Single Model', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('option', { name: /local\/qwen/ })).toHaveCount(0)

    await page.goto('/config/agent')
    await expect(page.getByRole('heading', { name: 'Profiles' })).toBeVisible()
    await expect(page.getByRole('button', { name: /New profile/ })).toHaveCount(0)
    await page.getByRole('button', { name: 'Open General' }).click()
    const detail = page.getByRole('dialog', { name: 'General' })
    await expect(detail.getByRole('button', { name: 'Edit' })).toHaveCount(0)
    await expect(detail.getByRole('button', { name: /Delete/ })).toHaveCount(0)
  })

  test('rejects a direct Agent management URL without Agent or Tool read access', async ({
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
    await expect(page).toHaveURL(/\/dashboard$/)
  })

  test('searches and paginates Agent resources on the Router', async ({ page }) => {
    const mock = await bootstrap(page, { profilePages: true })
    await page.goto('/config/agent')

    await expect(page.getByRole('button', { name: 'Open General' })).toBeVisible()
    await page.getByRole('button', { name: 'Load more' }).click()
    await expect(page.getByRole('button', { name: 'Open Builder' })).toBeVisible()
    expect(mock.profileQueries.some((query) => query.includes('cursor=profiles-page-2'))).toBe(true)

    await page.getByRole('searchbox', { name: 'Search Profiles' }).fill('Builder')
    await expect(page.getByRole('button', { name: 'Open General' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Open Builder' })).toBeVisible()
    expect(mock.profileQueries.some((query) => query.includes('search=Builder'))).toBe(true)
  })

  test('ignores an aborted stale resource search', async ({ page }) => {
    await bootstrap(page, { profilePages: true, profileSearchRace: true })
    await page.goto('/config/agent')

    const search = page.getByRole('searchbox', { name: 'Search Profiles' })
    await search.fill('General')
    await page.waitForTimeout(250)
    await search.fill('Builder')

    await expect(page.getByRole('button', { name: 'Open Builder' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Open General' })).toHaveCount(0)
  })

  test('selects target capabilities from authorized Model Cards', async ({ page }) => {
    await bootstrap(page, { profilePages: true, profileCapability: 'legacy-audio' })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'Open General' }).click()
    const dialog = page.getByRole('dialog', { name: 'General' })
    await dialog.getByRole('button', { name: 'Edit' }).click()
    await dialog.getByText('Advanced settings', { exact: true }).click()

    await expect(dialog.getByRole('checkbox', { name: /legacy-audio/ })).toBeChecked()
    await expect(dialog.getByText('Not currently advertised', { exact: true })).toBeVisible()
    await expect(dialog.getByRole('checkbox', { name: 'images' })).toBeVisible()
    await expect(dialog.getByRole('checkbox', { name: 'tools' })).toBeVisible()
    await dialog.getByRole('checkbox', { name: /legacy-audio/ }).uncheck()
    await expect(dialog.getByRole('checkbox', { name: /legacy-audio/ })).toHaveCount(0)
  })

  test('shows catalog failure in the editor and prevents a destructive save', async ({ page }) => {
    await bootstrap(page, { capabilityLoadFailure: true })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'New profile' }).click()
    const dialog = page.getByRole('dialog', { name: 'Create profile' })
    await dialog.getByText('Advanced settings', { exact: true }).click()

    await expect(dialog.getByRole('alert')).toContainText('Model capabilities are unavailable.')
    await expect(dialog.getByRole('button', { name: 'Create profile' })).toBeDisabled()
  })

  test('requires explicit approval when a Connection discovers tools', async ({ page }) => {
    const mock = await bootstrap(page, { connectionApproval: true })
    await page.goto('/config/agent')

    await page.getByRole('button', { name: 'Connections' }).click()
    await page.getByRole('button', { name: 'Open Knowledge tools' }).click()
    const detail = page.getByRole('dialog', { name: 'Knowledge tools' })
    await expect(detail.getByText('Pending approval', { exact: true })).toBeVisible()
    await detail.getByRole('button', { name: 'Approve tools' }).click()
    const confirmation = page.getByRole('dialog', { name: 'Approve discovered tools?' })
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
