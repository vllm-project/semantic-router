import { lstatSync } from 'node:fs'
import { isAbsolute } from 'node:path'
import { expect, type Page } from '@playwright/test'

export interface AcceptanceConnection {
  base_url: string
  public_https_origin?: string
  auth_state_path?: string
  expected_deployment_sha?: string
}

export function acceptanceOrigin(
  plan: AcceptanceConnection,
  publicOptIn = process.env.SR_BENCH_PUBLIC_ORIGIN,
): string {
  const url = new URL(plan.base_url)
  if (url.username || url.password || url.pathname !== '/' || url.search || url.hash)
    throw new Error('Acceptance requires a credential-free origin without a path or query.')
  const loopback =
    url.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)
  if (loopback && !plan.public_https_origin && !plan.auth_state_path) return url.origin
  if (
    url.protocol !== 'https:' ||
    plan.public_https_origin !== url.origin ||
    publicOptIn !== url.origin ||
    !/^[a-f0-9]{40}$/.test(plan.expected_deployment_sha ?? '') ||
    !plan.auth_state_path ||
    !isAbsolute(plan.auth_state_path)
  )
    throw new Error(
      'Public acceptance requires an explicit HTTPS opt-in, deployment SHA and private auth state.',
    )
  const state = lstatSync(plan.auth_state_path)
  if (
    !state.isFile() ||
    state.isSymbolicLink() ||
    (state.mode & 0o777) !== 0o600 ||
    (process.getuid && state.uid !== process.getuid()) ||
    state.size > 1024 * 1024
  )
    throw new Error('Auth state must be an owner-only 0600 regular file, at most 1 MiB.')
  return url.origin
}

export async function authenticateAcceptance(page: Page, plan: AcceptanceConnection) {
  const origin = acceptanceOrigin(plan)
  // Continue through the caller's mutation guard; never allow an external redirect
  // or resource request to receive an authenticated browser request.
  await page.route('**/*', (route) =>
    new URL(route.request().url()).origin === origin
      ? route.fallback()
      : route.abort('blockedbyclient'),
  )
  if (!plan.auth_state_path) {
    await page.goto(`${origin}/__acceptance/login`)
    await page.getByRole('button', { name: 'Start acceptance session', exact: true }).click()
    await page.waitForURL((url) => url.origin === origin && url.pathname !== '/__acceptance/login')
  }
  // Playwright loads the private storageState directly into its disposable context.
  // Do not inspect, serialize, attach or log its cookie values.
  const session = await page.request.get(`${origin}/api/auth/me`, { maxRedirects: 0 })
  expect(session.status(), 'The existing authenticated session must remain valid.').toBe(200)
  const account = await session.json()
  expect(account.user?.permissions?.includes('evaluation.read')).toBe(true)
}
