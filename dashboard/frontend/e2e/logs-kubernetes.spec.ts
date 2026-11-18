import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const podLogs = (component: string) => ({
  deployment_type: 'kubernetes',
  service: component,
  supported: true,
  components: ['router', 'dashboard'],
  namespace: 'vllm-sr',
  pods: ['router-7d9f8c6b5-x2x4z', 'router-7d9f8c6b5-k8m2p'],
  logs: [
    {
      service: 'router',
      pod: 'router-7d9f8c6b5-x2x4z',
      timestamp: '2026-11-18T02:40:00.000000001Z',
      line: '{"level":"info","msg":"replica one ready"}',
    },
    {
      service: 'router',
      pod: 'router-7d9f8c6b5-k8m2p',
      timestamp: '2026-11-18T02:40:01Z',
      line: '{"level":"warn","msg":"replica two slow"}',
    },
  ],
  count: 2,
  message: '1 dashboard pod has not started yet.',
})

test('reads pod logs from Kubernetes and labels each line with its pod', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  const components: string[] = []
  await page.route('**/api/logs**', async (route) => {
    const component = new URL(route.request().url()).searchParams.get('component') ?? 'router'
    components.push(component)
    await route.fulfill({ json: podLogs(component) })
  })

  await page.goto('/observe/logs')

  await expect(page.getByText('Kubernetes · vllm-sr')).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Router', exact: true })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Dashboard', exact: true })).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Envoy', exact: true })).toHaveCount(0)
  await expect(page.getByText('router-x2x4z', { exact: true })).toBeVisible()
  await expect(page.getByText('router-k8m2p', { exact: true })).toBeVisible()
  await expect(page.getByText('{"level":"warn","msg":"replica two slow"}')).toBeVisible()
  await expect(page.getByRole('status').filter({ hasText: 'has not started yet' })).toBeVisible()

  await page.getByRole('searchbox', { name: 'Search logs' }).fill('k8m2p')
  await expect(page.getByText('router-x2x4z', { exact: true })).toHaveCount(0)
  await expect(page.getByText('router-k8m2p', { exact: true })).toBeVisible()

  await page.getByRole('radio', { name: 'Router', exact: true }).click()
  await expect.poll(() => components[components.length - 1]).toBe('router')
})

test('names the permission the Dashboard service account lacks', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/logs**', async (route) => {
    await route.fulfill({
      json: {
        deployment_type: 'kubernetes',
        service: 'all',
        supported: false,
        components: ['router', 'dashboard'],
        namespace: 'vllm-sr',
        logs: [],
        count: 0,
        message:
          'The Dashboard\'s service account cannot list pods in namespace "vllm-sr". Grant it the list verb on pods in a Role bound to that service account.',
      },
    })
  })

  await page.goto('/observe/logs')

  await expect(page.getByText('The Dashboard cannot read pod logs')).toBeVisible()
  await expect(page.getByText(/Grant it the list verb on pods/)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Check again' })).toBeVisible()
  await expect(page.getByRole('switch', { name: 'Follow' })).toBeDisabled()
})
