import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedSession } from './support/auth'

const settings = {
  readonlyMode: true,
  serverReadonly: true,
  platform: '',
  envoyUrl: '',
  routerPublicUrl: '',
}

async function mockReadonlyOpenClaw(page: Page) {
  await mockAuthenticatedSession(page)
  await page.route('**/api/settings', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(settings),
    })
  })
  await page.route('**/api/openclaw/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          running: true,
          containerName: 'worker-a',
          gatewayUrl: 'http://127.0.0.1:18788',
          port: 18788,
          healthy: true,
          error: '',
          teamId: 'team-alpha',
          teamName: 'Team Alpha',
          agentName: 'Worker A',
          agentEmoji: '',
          agentRole: 'Operator',
          agentVibe: 'Precise',
          agentPrinciples: 'Do the work',
          createdAt: '2026-03-09T00:00:00Z',
        },
      ]),
    })
  })
  await page.route('**/api/openclaw/teams', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          id: 'team-alpha',
          name: 'Team Alpha',
          vibe: 'Calm',
          role: 'Operations',
          principal: 'Safety first',
          leaderId: 'leader-1',
        },
      ]),
    })
  })
  await page.route('**/api/openclaw/workers', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          name: 'worker-a',
          teamId: 'team-alpha',
          agentName: 'Worker A',
          agentEmoji: '',
          agentRole: 'Operator',
          agentVibe: 'Precise',
          agentPrinciples: 'Do the work',
          roleKind: 'worker',
        },
      ]),
    })
  })
}

test('OpenClaw stays browsable in readonly mode without exposing mutations', async ({ page }) => {
  await mockReadonlyOpenClaw(page)
  await page.goto('/openclaw')

  await page.getByRole('tab', { name: /Claw Team/ }).click()
  await expect(page.getByRole('button', { name: 'New Team' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Edit' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Delete' })).toHaveCount(0)

  await page.getByRole('tab', { name: /Claw Worker/ }).click()
  await expect(page.getByRole('button', { name: 'New Worker' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Edit' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Delete' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Status' }).first()).toBeEnabled()

  await page.getByRole('tab', { name: /Claw Dashboard/ }).click()
  await expect(page.getByRole('button', { name: 'Dashboard', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Stop' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Remove' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Refresh Status' })).toBeEnabled()
  await expect(page.locator('iframe[title*="OpenClaw Control UI"]')).toHaveCount(0)
})
