import type { Page } from '@playwright/test'

import playwrightConfig from '../../playwright.config'

// Readable by design, unlike vsr_session — the app reads it to populate X-CSRF-Token.
export const TEST_CSRF_TOKEN = 'test-csrf-token'

// addCookies needs an absolute URL and derives domain and path from it. Always the app's
// base URL, never page.url(): a page that has navigated deeper would scope the cookie to
// that subpath, and about:blank is rejected outright.
const BASE_URL = playwrightConfig.use?.baseURL ?? 'http://localhost:3001'

type SessionUser = {
  id: string
  email: string
  name: string
  role?: string
  permissions?: string[]
}

type BootstrapOptions = {
  token?: string
  user?: SessionUser
  setupState?: Record<string, unknown>
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
    'feedback.submit',
    'logs.read',
    'mcp.manage',
    'mcp.read',
    'mlpipeline.manage',
    'openclaw.manage',
    'openclaw.read',
    'replay.read',
    'tools.use',
    'topology.read',
    'users.manage',
    'users.view',
  ],
}

const defaultSetupState = {
  setupMode: false,
  listenerPort: 8700,
  models: 1,
  decisions: 1,
  hasModels: true,
  hasDecisions: true,
  canActivate: true,
}

const defaultSettings = {
  readonlyMode: false,
  serverReadonly: false,
  runtimeConfigWritable: true,
  recipeStoreWritable: true,
  setupMode: false,
  platform: '',
  envoyUrl: '',
  routerEvalEndpoint: '',
  evaluationAvailable: true,
  evaluationUnavailableReason: '',
}

export const dashboardSettingsResponse = (
  overrides: Record<string, unknown> = {},
): Record<string, unknown> => ({ ...defaultSettings, ...overrides })

export async function mockAuthenticatedSession(
  page: Page,
  { token = 'test-auth-token', user = defaultUser }: BootstrapOptions = {},
): Promise<{ token: string; user: SessionUser }> {
  // The app authenticates with the HttpOnly vsr_session cookie the server sets at login,
  // so tests seed a cookie rather than localStorage. httpOnly is what makes the suite
  // exercise the real constraint: a cookie page script could read is not the shipped
  // behaviour. Cookies go on the context so they apply before the first navigation. #2465
  await page.context().addCookies([
    { name: 'vsr_session', value: token, url: BASE_URL, httpOnly: true, sameSite: 'Lax' },
    { name: 'vsr_csrf', value: TEST_CSRF_TOKEN, url: BASE_URL, httpOnly: false, sameSite: 'Lax' },
  ])

  // Keep browser tests hermetic. A missing mock should surface as a local 404,
  // never fall through to a developer's running Dashboard and invalidate the
  // synthetic session with an unrelated 401. More specific routes registered
  // below (or by the calling spec) take precedence in Playwright.
  await page.route('**/api/**', async (route) => {
    await route.fulfill({
      status: 404,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: { message: 'Unmocked browser-test API request' } }),
    })
  })

  await page.route('**/api/auth/me', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user }),
    })
  })

  return { token, user }
}

export async function mockAuthenticatedAppShell(
  page: Page,
  options: BootstrapOptions = {},
): Promise<{ token: string; user: SessionUser }> {
  const session = await mockAuthenticatedSession(page, options)
  const setupState = { ...defaultSetupState, ...(options.setupState ?? {}) }
  const settings = { ...defaultSettings, ...(options.settings ?? {}) }

  await page.route('**/api/setup/state', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(setupState),
    })
  })

  await page.route('**/api/settings', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
    })
  })

  await page.route('**/api/mcp/servers', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify([]),
    })
  })

  await page.route('**/api/mcp/tools', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tools: [] }),
    })
  })

  await page.route('**/api/auth/bootstrap/can-register', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ canRegister: false }),
    })
  })

  await page.route('**/api/status', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        overall: 'healthy',
        deployment_type: 'local',
        services: [],
      }),
    })
  })

  await page.route('**/api/router/v1/models*', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        object: 'list',
        data: [
          {
            id: 'vllm-sr/default',
            object: 'model',
            owned_by: 'vllm-sr',
            metadata: { type: 'mixture-of-models', recipe: 'default' },
          },
        ],
      }),
    })
  })

  await page.route('**/api/admin/permissions', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rolePermissions: {}, allPermissions: [] }),
    })
  })

  return session
}
