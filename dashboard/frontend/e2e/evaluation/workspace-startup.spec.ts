import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

test('evaluation opens without downloading the configuration compiler', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  const compilerRequests: string[] = []
  const inventoryRequests = new Set<string>()
  page.on('request', (request) => {
    const path = new URL(request.url()).pathname
    if (path === '/signal-compiler.wasm' || path === '/wasm_exec.js') {
      compilerRequests.push(path)
    }
  })
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const resource = new URL(route.request().url()).pathname.split('/').at(-1)!
    inventoryRequests.add(resource)
    const body = resource === 'catalog'
      ? { version: 'sr-bench-1.0', benchmarks: [], profiles: [] }
      : { [resource]: [] }
    await route.fulfill({ status: 200, json: body })
  })

  await page.goto('/evaluation')
  await expect(page.getByRole('navigation', { name: 'Global navigation' })).toBeVisible()
  await expect.poll(() => [...inventoryRequests].sort()).toEqual([
    'catalog', 'datasets', 'runs', 'targets',
  ])
  expect(compilerRequests).toEqual([])
})
