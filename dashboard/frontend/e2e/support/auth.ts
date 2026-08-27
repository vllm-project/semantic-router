import type { Page } from '@playwright/test'

type SessionUser = {
  id: string
  email: string
  name: string
  role?: string
  permissions?: string[]
}

type BootstrapOptions = {
  user?: SessionUser
  managementPermissions?: string[]
  settings?: Record<string, unknown>
}

const defaultUser: SessionUser = {
  id: 'user-admin-1',
  email: 'admin@example.com',
  name: 'Admin User',
  role: 'admin',
  permissions: [
    'config.deploy',
    'config.read',
    'config.write',
    'evaluation.read',
    'evaluation.run',
    'evaluation.write',
    'logs.read',
    'mlpipeline.manage',
    'openclaw.manage',
    'openclaw.read',
    'tools.use',
    'topology.read',
    'users.manage',
    'users.view',
  ],
}

const defaultSettings = {
  readonlyMode: false,
  serverReadonly: false,
  platform: '',
  envoyUrl: '',
  routerPublicUrl: '',
  routerEvalEndpoint: '',
}

const statusThrough = '2026-08-23T00:00:00Z'
const statusServices = ['Router', 'Routing access', 'Envoy', 'Dashboard']

function operationalSystemStatus() {
  const through = Date.parse(statusThrough)
  return {
    overall: 'operational',
    services: statusServices.map((name) => ({ name, status: 'operational', healthy: true })),
    history: {
      windowHours: 90,
      through: statusThrough,
      services: statusServices.map((name) => ({
        name,
        hours: Array.from({ length: 90 }, (_, index) => ({
          observedAt: new Date(through - (89 - index) * 3_600_000)
            .toISOString()
            .replace('.000Z', 'Z'),
          status: 'operational',
        })),
      })),
    },
  }
}

const managementMediaType = 'application/vnd.vllm-semantic-router.management.v1+json'
const namespaceId = '10000000-0000-4000-8000-000000000001'
const routerPrincipalId = '10000000-0000-4000-8000-000000000010'
const routerUserId = '10000000-0000-4000-8000-000000000011'
const defaultManagementPermissions = [
  'access_policy.manage',
  'access_policy.read',
  'agent.manage',
  'agent.read',
  'agent.use',
  'audit.read',
  'key.manage',
  'key.read',
  'delegation.use',
  'log.read',
  'log_payload.read',
  'quota.read',
  'rate_policy.manage',
  'rate_policy.read',
  'routing.manage',
  'routing.publish',
  'routing.read',
  'team.manage',
  'team.read',
  'tool.invoke',
  'tool.manage',
  'tool.read',
  'usage.internal_dimensions.read',
  'usage.read',
  'user.manage',
  'user.read',
]

export async function mockAuthenticatedSession(
  page: Page,
  {
    user = defaultUser,
    managementPermissions = defaultManagementPermissions,
  }: BootstrapOptions = {},
): Promise<{ user: SessionUser }> {
  await page.route('**/api/auth/me', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user }),
    })
  })

  await page.route('**/api/router/management/v1/me', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': managementMediaType },
      body: JSON.stringify({
        principal: {
          principalId: routerPrincipalId,
          displayName: user.name,
          kind: 'human',
          status: 'active',
        },
        session: {
          sessionId: '10000000-0000-4000-8000-000000000002',
          authenticatedAt: '2026-08-23T00:00:00Z',
          expiresAt: '2099-08-23T00:00:00Z',
          evidenceKind: 'human',
        },
        clusterPermissions: [],
        namespaces: [
          {
            namespace: {
              namespaceId,
              name: 'Test',
              status: 'active',
              desiredRevision: 1,
              appliedRevision: 1,
            },
            permissions: managementPermissions,
            roleBindings: [],
            user: {
              userId: routerUserId,
              email: user.email,
              displayName: user.name,
              status: 'active',
            },
            teams: [],
            selfServicePolicy: {
              maxKeysPerUser: 1,
              maxDelegatedSessions: 3,
              delegatedSessionTtlSeconds: 900,
              allowTeamKeyDelegation: false,
              automaticFirstKey: true,
              revision: 1,
            },
          },
        ],
      }),
    })
  })

  await page.route('**/api/router/management/v1/self/inference-keys*', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': managementMediaType },
      body: JSON.stringify({
        data: [
          {
            keyId: '10000000-0000-4000-8000-000000000003',
            name: 'Playground',
            owner: { type: 'user', id: routerUserId },
            expiresAt: '2099-08-23T00:00:00Z',
          },
        ],
        page: { hasMore: false, pageSize: 100 },
      }),
    })
  })

  await page.route('**/api/router/management/v1/self/inference-sessions**', async (route) => {
    if (route.request().method() === 'DELETE') {
      await route.fulfill({ status: 204 })
      return
    }
    await route.fulfill({
      status: 201,
      headers: { 'Content-Type': managementMediaType, 'Cache-Control': 'no-store' },
      body: JSON.stringify({
        resourceId: '10000000-0000-4000-8000-000000000004',
        kind: 'delegated_inference_credential',
        secret: 'vsd_test-delegated-credential',
        expiresAt: '2099-08-23T00:00:00Z',
      }),
    })
  })

  return { user }
}

export async function mockAuthenticatedAppShell(
  page: Page,
  options: BootstrapOptions = {},
): Promise<{ user: SessionUser }> {
  const session = await mockAuthenticatedSession(page, options)
  const settings = { ...defaultSettings, ...(options.settings ?? {}) }

  await page.route('**/api/settings', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
  })

  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
      body: JSON.stringify(operationalSystemStatus()),
    })
  })

  return session
}
