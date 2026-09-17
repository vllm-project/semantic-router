import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedSession } from './support/auth'

type Reply = 'healthy' | 'network' | 'invalid-json' | number

async function sessionResponses(page: Page, initial: Reply = 'healthy') {
  const session = await mockAuthenticatedSession(page)
  let reply = initial
  let logoutRequests = 0
  await page.route('**/api/auth/logout', async (route) => {
    logoutRequests += 1
    await route.fulfill({ json: { ok: true } })
  })
  await page.route('**/api/auth/me', async (route) => {
    expect(route.request().headers().cookie).toContain('vsr_session=' + session.token)
    if (reply === 'network') await route.abort('failed')
    else if (reply === 'invalid-json') await route.fulfill({ status: 200, body: 'invalid json' })
    else if (reply === 'healthy') await route.fulfill({ json: { user: session.user } })
    else await route.fulfill({ status: reply, body: 'Unavailable' })
  })
  return {
    session,
    set: (next: Reply) => {
      reply = next
    },
    logouts: () => logoutRequests,
  }
}

for (const failure of ['network', 503, 403, 'invalid-json'] as const) {
  test(`keeps the authenticated workspace and draft through ${failure}, then retries`, async ({
    page,
  }) => {
    const replies = await sessionResponses(page)
    await page.goto('/e2e/support/auth-session.html')
    await expect(page.getByRole('heading', { name: 'Workspace for Admin User' })).toBeVisible()
    await page.getByLabel('Draft').fill('Preserve this unfinished message')
    replies.set(failure)
    await page.getByRole('button', { name: 'Check session', exact: true }).click()
    await expect(page.getByRole('alert')).toBeVisible()
    await expect(page.getByTestId('session-state')).toHaveText('Authenticated')
    await expect(page.getByRole('heading', { name: 'Sign in' })).toHaveCount(0)
    await expect(page.getByLabel('Draft')).toHaveValue('Preserve this unfinished message')
    expect(replies.logouts()).toBe(0)
    expect(
      (await page.context().cookies()).find((cookie) => cookie.name === 'vsr_session')?.value,
    ).toBe(replies.session.token)
    replies.set('healthy')
    await page.getByRole('button', { name: 'Retry', exact: true }).click()
    await expect(page.getByRole('alert')).toHaveCount(0)
    await expect(page.getByLabel('Draft')).toHaveValue('Preserve this unfinished message')
  })
}

for (const failure of ['network', 503] as const) {
  test(`does not invent a session on initial ${failure} and recovers without login`, async ({
    page,
  }) => {
    const replies = await sessionResponses(page, failure)
    await page.goto('/e2e/support/auth-session.html')
    await expect(page.getByRole('heading', { name: 'Unable to verify your session' })).toBeVisible()
    await expect(page.getByTestId('session-state')).toHaveText('Unverified')
    await expect(page.getByLabel('Draft')).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Sign in' })).toHaveCount(0)
    replies.set('healthy')
    await page.getByRole('button', { name: 'Retry', exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Workspace for Admin User' })).toBeVisible()
    expect(replies.logouts()).toBe(0)
  })
}

test('a definitive 401 during retry clears session state and routes to login', async ({ page }) => {
  const replies = await sessionResponses(page)
  await page.goto('/e2e/support/auth-session.html')
  await expect(page.getByLabel('Draft')).toBeVisible()
  replies.set(503)
  await page.getByRole('button', { name: 'Check session', exact: true }).click()
  await expect(page.getByRole('alert')).toBeVisible()
  replies.set(401)
  await page.getByRole('button', { name: 'Retry', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Sign in' })).toBeVisible()
  await expect(page.getByTestId('session-state')).toHaveText('Unverified')
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByLabel('Draft')).toHaveCount(0)
})
