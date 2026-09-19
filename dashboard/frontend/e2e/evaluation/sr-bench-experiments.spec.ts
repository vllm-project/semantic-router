import { expect, test, type Page, type Route } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
const id = 'exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
const experiment = {
  id,
  name: 'Balance recipe study',
  created_at: '2026-09-18T00:00:00Z',
  updated_at: '2026-09-18T00:00:00Z',
  run_count: 21,
}
const run = (index: number, roles: string[] = ['candidate']) => ({
  id: `run-${index}`,
  status: 'completed',
  created_at: '2026-09-18T00:00:00Z',
  updated_at: '2026-09-18T00:00:00Z',
  experiment_roles: roles,
  progress: { total: 2, completed: 2, failed: 0 },
  manifest: {
    name: `Recipe ${index}`,
    profile: 'quick',
    mode: 'live',
    targets: [{ id: 'model', kind: index === 0 ? 'single' : 'mom', model: 'model' }],
  },
})
async function setup(page: Page, handler: (route: Route) => Promise<void>) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/sr-bench/v1/**', (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.includes('/experiments')) return handler(route)
    if (path.endsWith('/catalog')) return route.fulfill({ json: { profiles: [], benchmarks: [] } })
    if (path.endsWith('/datasets')) return route.fulfill({ json: { datasets: [] } })
    if (path.endsWith('/targets')) return route.fulfill({ json: { targets: [] } })
    if (path.endsWith('/runs'))
      return route.fulfill({
        json: {
          runs: [
            run(0, ['baseline']),
            ...Array.from({ length: 22 }, (_, i) =>
              run(i + 1, i === 19 ? ['recovery'] : ['candidate']),
            ),
            run(99, []),
          ],
        },
      })
    return route.fulfill({ status: 404, json: { error: 'Missing fixture' } })
  })
}
async function choose(page: Page, label: string, value: string) {
  await page.getByRole('combobox', { name: label, exact: true }).click()
  await page.getByRole('listbox').locator(`[data-value="${value}"]`).click()
}
test('creates only metadata and reconciles a lost response using the same durable identity after reload', async ({
  page,
}) => {
  const bodies: unknown[] = []
  await setup(page, async (route) => {
    const path = new URL(route.request().url()).pathname
    if (route.request().method() === 'POST') {
      expect(path).toBe('/api/sr-bench/v1/experiments')
      bodies.push(route.request().postDataJSON())
      if (bodies.length === 1) return route.abort('failed')
      return route.fulfill({ json: experiment })
    }
    return route.fulfill({
      json: path.endsWith('/runs')
        ? { experiment, members: [], next_cursor: null, has_more: false }
        : { experiments: [], next_cursor: null, has_more: false },
    })
  })
  await page.goto('/evaluation?view=experiments')
  await page.getByLabel('Experiment name', { exact: true }).fill(experiment.name)
  await page.getByRole('button', { name: 'Create experiment', exact: true }).click()
  await expect(
    page.getByRole('button', { name: 'Check or create same experiment', exact: true }),
  ).toBeEnabled()
  await page.reload()
  await expect(
    page.getByText('Experiment submission needs reconciliation', { exact: true }),
  ).toBeVisible()
  await expect(page.getByLabel('Experiment name', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Check or create same experiment', exact: true }).click()
  await expect(page.getByRole('heading', { name: experiment.name, exact: true })).toBeVisible()
  expect(bodies).toHaveLength(2)
  expect(bodies[1]).toEqual(bodies[0])
  await page.getByRole('button', { name: 'Create baseline', exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`view=new&experiment=${id}&role=baseline`))
})
test('paginates experiments and members and uses only authoritative run roles', async ({
  page,
}, testInfo) => {
  const attached: unknown[] = []
  await setup(page, async (route) => {
    const url = new URL(route.request().url())
    if (route.request().method() === 'POST') {
      attached.push(route.request().postDataJSON())
      return route.fulfill({ json: experiment })
    }
    expect(url.searchParams.get('limit')).toBe('20')
    if (!url.pathname.endsWith('/runs'))
      return route.fulfill({
        json: {
          experiments:
            url.searchParams.get('after') === '0'
              ? [experiment]
              : [
                  {
                    ...experiment,
                    id: 'exp-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
                    name: 'Second page',
                  },
                ],
          next_cursor: url.searchParams.get('after') === '0' ? 20 : null,
          has_more: url.searchParams.get('after') === '0',
        },
      })
    const first = url.searchParams.get('after') === '0'
    return route.fulfill({
      json: {
        experiment,
        members: Array.from({ length: first ? 20 : 1 }, (_, i) => ({
          run_id: `run-${first ? i : 20}`,
          role: !first ? 'recovery' : i === 0 ? 'baseline' : 'candidate',
          hypothesis: i === 1 ? 'Prefer the less costly model for this scope.' : '',
          linked_at: '2026-09-18T00:00:00Z',
        })),
        next_cursor: first ? 20 : null,
        has_more: first,
      },
    })
  })
  await page.goto('/evaluation?view=experiments')
  await page
    .getByRole('navigation', { name: 'Experiment pages' })
    .getByRole('button', { name: 'Next', exact: true })
    .click()
  await expect(page.getByRole('button', { name: /Second page/ })).toBeVisible()
  await page
    .getByRole('navigation', { name: 'Experiment pages' })
    .getByRole('button', { name: 'Previous', exact: true })
    .click()
  await page.getByRole('button', { name: /Balance recipe study/ }).click()
  await expect(page.getByRole('article')).toHaveCount(20)
  await page.getByRole('button', { name: 'Compare iterations', exact: true }).last().click()
  await expect(page).toHaveURL(/view=compare&baseline=run-0/)
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await page
    .getByRole('navigation', { name: 'Experiment pages' })
    .getByRole('button', { name: 'Next', exact: true })
    .click()
  await expect(page.getByRole('article')).toHaveCount(1)
  await expect(page.getByRole('article')).toContainText('Recipe 20')
  await expect(page.getByRole('article')).toContainText('Recovery attempt')
  await page.getByRole('combobox', { name: 'Saved run', exact: true }).click()
  await expect(page.getByRole('listbox').locator('[data-value="run-99"]')).toHaveCount(0)
  await page.getByRole('listbox').locator('[data-value="run-21"]').click()
  await choose(page, 'Role in this experiment', 'candidate')
  await page.getByLabel('Hypothesis or note', { exact: true }).fill('Test a lower-cost route.')
  await page.getByRole('button', { name: 'Link run', exact: true }).click()
  expect(attached).toEqual([
    { run_id: 'run-21', role: 'candidate', hypothesis: 'Test a lower-cost route.' },
  ])
  await page
    .getByRole('heading', { name: experiment.name, exact: true })
    .evaluate((element) => element.scrollIntoView({ block: 'start' }))
  for (const button of [
    page.getByRole('button', { name: 'Create baseline', exact: true }),
    page
      .getByRole('navigation', { name: 'Experiment pages' })
      .getByRole('button', { name: 'Previous', exact: true }),
    page.getByRole('button', { name: 'Link run', exact: true }),
  ]) {
    expect((await button.locator('svg').boundingBox())!.width).toBeLessThanOrEqual(20)
  }
  await page.screenshot({ path: testInfo.outputPath('experiment-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('experiment-mobile.png') })
})

test('permits experiment metadata writes without generation permission and hides them from readers', async ({
  page,
}) => {
  const bodies: unknown[] = []
  await setup(page, async (route) => {
    if (route.request().method() === 'POST') {
      bodies.push(route.request().postDataJSON())
      return route.fulfill({ json: experiment })
    }
    return route.fulfill({
      json: new URL(route.request().url()).pathname.endsWith('/runs')
        ? {
            experiment,
            members:
              bodies.length < 2
                ? []
                : [
                    {
                      run_id: 'run-0',
                      role: 'baseline',
                      hypothesis: '',
                      linked_at: '2026-09-18T00:00:00Z',
                    },
                  ],
            next_cursor: null,
            has_more: false,
          }
        : { experiments: [], next_cursor: null, has_more: false },
    })
  })
  let permissions = ['evaluation.read', 'evaluation.write']
  await page.route('**/api/auth/me', (route) =>
    route.fulfill({
      json: {
        user: {
          id: 'metadata-writer',
          email: 'writer@example.com',
          name: 'Metadata writer',
          role: 'write',
          permissions,
        },
      },
    }),
  )
  await page.goto('/evaluation?view=experiments')
  await page.getByLabel('Experiment name', { exact: true }).fill('Metadata only')
  await page.getByRole('button', { name: 'Create experiment', exact: true }).click()
  await expect(page.getByRole('heading', { name: experiment.name, exact: true })).toBeVisible()
  await choose(page, 'Saved run', 'run-0')
  await page.getByRole('button', { name: 'Link run', exact: true }).click()
  expect(bodies).toHaveLength(2)
  expect(bodies[0]).toMatchObject({ name: 'Metadata only', idempotency_key: expect.any(String) })
  expect(bodies[1]).toEqual({ run_id: 'run-0', role: 'baseline', hypothesis: '' })
  await expect(
    page
      .getByRole('region', { name: 'Evaluation experiments', exact: true })
      .getByRole('button', { name: 'Compare iterations', exact: true }),
  ).toBeVisible()
  for (const action of ['Create baseline', 'Preview candidate', 'Evaluate candidate'])
    await expect(page.getByRole('button', { name: action, exact: true })).toHaveCount(0)
  permissions = ['evaluation.read']
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await expect(page.getByRole('heading', { name: experiment.name, exact: true })).toBeVisible()
  await expect(
    page
      .getByRole('region', { name: 'Evaluation experiments', exact: true })
      .getByRole('button', { name: 'Compare iterations', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Link run', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Create baseline', exact: true })).toHaveCount(0)
  await page.goto('/evaluation?view=experiments')
  await expect(page.getByRole('heading', { name: 'Experiments', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Create experiment', exact: true })).toHaveCount(0)
  expect(bodies).toHaveLength(2)
})
