import { expect, test, type Page, type Route } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
const id = 'exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
const experiment = {
  id,
  name: 'Balance recipe study',
  created_at: '2026-09-18T00:00:00Z',
  updated_at: '2026-09-18T00:00:00Z',
  run_count: 21,
  active_run_count: 0,
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

test('makes an empty experiment actionable without exposing an empty linking form', async ({
  page,
}, testInfo) => {
  const mutations: string[] = []
  await setup(page, async (route) => {
    if (route.request().method() !== 'GET') mutations.push(route.request().method())
    return route.fulfill({
      json: {
        experiment: { ...experiment, run_count: 0 },
        members: [],
        next_cursor: null,
        has_more: false,
      },
    })
  })
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await expect(
    page.getByRole('heading', { name: 'Start with your single models', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Run single models', exact: true })).toBeEnabled()
  await expect(page.getByRole('button', { name: 'Use saved results', exact: true })).toBeEnabled()
  await expect(page.getByRole('combobox', { name: 'Saved run', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Compare results', exact: true })).toHaveCount(0)
  await page.setViewportSize({ width: 1600, height: 1000 })
  await page.screenshot({ path: testInfo.outputPath('experiment-empty-desktop.png') })
  await page.getByRole('button', { name: 'Use saved results', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Saved run', exact: true })).toBeVisible()
  await expect(
    page.getByText('Reuse a run you already have. Adding it here does not rerun the evaluation.'),
  ).toBeVisible()
  await page.getByRole('button', { name: 'Close', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Saved run', exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Run single models', exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`view=new&experiment=${id}&role=baseline`))
  expect(mutations).toEqual([])
})

test('groups recipe versions and keeps supporting checks and saved-run tools collapsed', async ({
  page,
}, testInfo) => {
  const roles = ['baseline', 'initial', 'candidate', 'candidate', 'preview']
  const names = [
    'Single-model reference run',
    'Starting Balance',
    'Cost-priority recipe',
    'Factual-guard recipe',
    'Routing check',
  ]
  await setup(page, async (route) =>
    route.fulfill({
      json: {
        experiment: { ...experiment, run_count: roles.length },
        members: roles.map((role, index) => ({
          run_id: `run-${index}`,
          role,
          hypothesis:
            index === 2 ? 'Try a lower-cost route without changing the question set.' : '',
          linked_at: experiment.created_at,
        })),
        next_cursor: null,
        has_more: false,
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({
      json: {
        runs: roles.map((role, index) => ({
          ...run(index, [role]),
          progress: {
            total: index === 0 ? 42 : 14,
            completed: index === 0 ? 41 : 14,
            failed: index === 0 ? 1 : 0,
          },
          status: index === 0 ? 'failed' : 'completed',
          manifest: {
            ...run(index).manifest,
            name: names[index],
            mode: role === 'preview' ? 'preview' : 'live',
            dataset: { case_count: 14 },
          },
        })),
      },
    }),
  )
  await page.setViewportSize({ width: 1600, height: 1000 })
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await expect(
    page.getByRole('heading', { name: 'Review your recipe versions', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Test another recipe', exact: true })).toBeEnabled()
  await expect(
    page
      .getByRole('button', { name: 'Compare results', exact: true })
      .locator('..')
      .getByRole('button')
      .first(),
  ).toHaveAccessibleName('Compare results')
  await expect(
    page.getByRole('button', { name: 'Evaluate starting recipe', exact: true }),
  ).toHaveCount(0)
  await expect(
    page.getByRole('region', { name: 'Single-model reference', exact: true }),
  ).toContainText('41 / 42 attempts completed')
  await expect(page.getByRole('region', { name: 'Starting recipe', exact: true })).toContainText(
    'Starting Balance',
  )
  const versions = page.getByRole('region', { name: 'Recipe versions', exact: true })
  await expect(versions.locator('li')).toHaveCount(2)
  await expect(versions.getByText('Recipe version', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Saved results', exact: true })).toHaveCount(1)
  await expect(page.getByRole('button', { name: 'Routing check', exact: true })).toBeHidden()
  await expect(page.getByRole('combobox', { name: 'Saved run', exact: true })).toHaveCount(0)
  await page.screenshot({ path: testInfo.outputPath('experiment-grouped-desktop.png') })
  await page
    .getByRole('heading', { name: 'Saved results', exact: true })
    .evaluate((element) => element.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: testInfo.outputPath('experiment-results-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.evaluate(() => scrollTo(0, 0))
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('experiment-grouped-mobile.png') })
  await page
    .getByRole('heading', { name: 'Recipe versions', exact: true })
    .evaluate((element) => element.scrollIntoView({ block: 'start' }))
  await page.screenshot({ path: testInfo.outputPath('experiment-results-mobile.png') })
  await page.getByText('Supporting checks', { exact: true }).click()
  await expect(page.getByRole('button', { name: 'Routing check', exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Test another recipe', exact: true }).click()
  await expect(page).toHaveURL(
    new RegExp(`view=new&baseline=run-0&experiment=${id}&role=candidate`),
  )
})

test('waits for complete membership before offering the next recipe, including a second-page reference', async ({
  page,
}) => {
  let failNextPage = true
  await setup(page, async (route) => {
    const later = new URL(route.request().url()).searchParams.get('after') === '20'
    if (later && failNextPage)
      return route.fulfill({ status: 503, json: { error: 'Membership page unavailable.' } })
    return route.fulfill({
      json: {
        experiment: { ...experiment, run_count: 3 },
        members: (later
          ? [
              { run_id: 'run-0', role: 'baseline' },
              { run_id: 'run-1', role: 'initial' },
            ]
          : [{ run_id: 'run-2', role: 'candidate' }]
        ).map((member) => ({ ...member, hypothesis: '', linked_at: experiment.created_at })),
        next_cursor: later ? null : 20,
        has_more: !later,
      },
    })
  })
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await expect(page.getByRole('alert')).toContainText('Could not load all runs in this experiment.')
  for (const action of [
    'Evaluate starting recipe',
    'Test another recipe',
    'Run single models',
    'Add saved results',
  ])
    await expect(page.getByRole('button', { name: action, exact: true })).toHaveCount(0)
  failNextPage = false
  await page.getByRole('button', { name: 'Reload experiment runs', exact: true }).click()
  await expect(page.getByRole('button', { name: 'Test another recipe', exact: true })).toBeEnabled()
  await expect(
    page.getByRole('button', { name: 'Evaluate starting recipe', exact: true }),
  ).toHaveCount(0)
  await expect(
    page.getByRole('region', { name: 'Single-model reference', exact: true }),
  ).toHaveCount(0)
  await page.getByRole('button', { name: 'Add saved results', exact: true }).click()
  await page.getByRole('combobox', { name: 'Saved run', exact: true }).click()
  await expect(page.getByRole('listbox').locator('[data-value="run-0"]')).toHaveCount(0)
  await expect(page.getByRole('listbox').locator('[data-value="run-1"]')).toHaveCount(0)
})

test('offers candidate protocol reuse for a failed baseline while preserving its failed status', async ({
  page,
}) => {
  let active = false
  await setup(page, async (route) =>
    route.fulfill({
      json: {
        experiment: { ...experiment, run_count: 1, active_run_count: active ? 1 : 0 },
        members: [
          { run_id: 'run-0', role: 'baseline', hypothesis: '', linked_at: experiment.created_at },
        ],
        next_cursor: null,
        has_more: false,
      },
    }),
  )
  await page.route('**/api/sr-bench/v1/runs', (route) =>
    route.fulfill({
      json: {
        runs: [
          {
            ...run(0, ['baseline']),
            status: active ? 'running' : 'failed',
            progress: { total: 42, completed: 41, failed: 1 },
          },
        ],
      },
    }),
  )
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  const reference = page.getByRole('region', { name: 'Single-model reference', exact: true })
  await expect(reference.getByText('failed', { exact: true })).toBeVisible()
  await expect(reference).toContainText('41 / 42 attempts completed')
  await expect(reference).toContainText('1 failed')
  await expect(page.getByRole('button', { name: 'Check routing first', exact: true })).toBeEnabled()
  await expect(
    page.getByRole('button', { name: 'Evaluate starting recipe', exact: true }),
  ).toBeEnabled()
  await page.getByRole('button', { name: 'Evaluate starting recipe', exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`view=new&baseline=run-0&experiment=${id}&role=initial`))
  active = true
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await expect(reference.getByText('running', { exact: true })).toBeVisible()
  await expect(reference).toContainText('41 / 42 attempts completed')
  await expect(page.getByRole('button', { name: 'Check routing first', exact: true })).toHaveCount(
    0,
  )
  await expect(
    page.getByRole('button', { name: 'Evaluate starting recipe', exact: true }),
  ).toHaveCount(0)
})
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
    page.getByRole('button', { name: 'Check creation request', exact: true }),
  ).toBeEnabled()
  await page.reload()
  await expect(
    page.getByText('Check your previous creation request', { exact: true }),
  ).toBeVisible()
  await expect(page.getByLabel('Experiment name', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Check creation request', exact: true }).click()
  await expect(page.getByRole('heading', { name: experiment.name, exact: true })).toBeVisible()
  expect(bodies).toHaveLength(2)
  expect(bodies[1]).toEqual(bodies[0])
  await page.getByRole('button', { name: 'Run single models', exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`view=new&experiment=${id}&role=baseline`))
})

test('clears only a confirmed deleted create submission and waits for an explicit new creation', async ({
  page,
}) => {
  const bodies: Array<{ name: string; idempotency_key: string }> = []
  await setup(page, async (route) => {
    if (route.request().method() === 'POST') {
      bodies.push(route.request().postDataJSON())
      if (bodies.length === 1) return route.abort('failed')
      if (bodies.length === 2)
        return route.fulfill({
          status: 409,
          json: { error: 'Experiment was deleted.', code: 'experiment_deleted' },
        })
      return route.fulfill({ json: { ...experiment, name: bodies[2].name } })
    }
    return route.fulfill({
      json: new URL(route.request().url()).pathname.endsWith('/runs')
        ? {
            experiment: { ...experiment, name: bodies[2]?.name ?? experiment.name },
            members: [],
            next_cursor: null,
            has_more: false,
          }
        : { experiments: [], next_cursor: null, has_more: false },
    })
  })
  await page.goto('/evaluation?view=experiments')
  await page.getByLabel('Experiment name', { exact: true }).fill('Deleted study')
  await page.getByRole('button', { name: 'Create experiment', exact: true }).click()
  await expect(
    page.getByRole('button', { name: 'Check creation request', exact: true }),
  ).toBeEnabled()
  await page.reload()
  await page.getByRole('button', { name: 'Check creation request', exact: true }).click()
  await expect(
    page.getByRole('status').filter({ hasText: 'That experiment was deleted.' }),
  ).toBeVisible()
  await expect(page.getByLabel('Experiment name', { exact: true })).toHaveValue('')
  expect(bodies).toHaveLength(2)
  expect(bodies[1]).toEqual(bodies[0])
  expect(
    await page.evaluate(() => sessionStorage.getItem('sr-bench-experiment:user-admin-1')),
  ).toBeNull()
  await page.getByLabel('Experiment name', { exact: true }).fill('New study')
  await page.getByRole('button', { name: 'Create experiment', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'New study', exact: true })).toBeVisible()
  expect(bodies).toHaveLength(3)
  expect(bodies[2].idempotency_key).not.toBe(bodies[0].idempotency_key)
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
  await expect(
    page.getByRole('region', { name: 'Saved runs', exact: true }).locator('ul > li'),
  ).toHaveCount(20)
  await page.getByRole('button', { name: 'Compare results', exact: true }).last().click()
  await expect(page).toHaveURL(/view=compare&baseline=run-0/)
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await page
    .getByRole('navigation', { name: 'Experiment pages' })
    .getByRole('button', { name: 'Next', exact: true })
    .click()
  await expect(
    page.getByRole('region', { name: 'Saved runs', exact: true }).locator('ul > li'),
  ).toHaveCount(1)
  await page.getByText('Supporting checks', { exact: true }).click()
  await expect(
    page.getByRole('region', { name: 'Saved runs', exact: true }).locator('ul > li'),
  ).toContainText('Recipe 20')
  await expect(
    page.getByRole('region', { name: 'Saved runs', exact: true }).locator('ul > li'),
  ).toContainText('Recovery attempt')
  await page.getByRole('button', { name: 'Add saved results', exact: true }).click()
  await page.getByRole('combobox', { name: 'Saved run', exact: true }).click()
  await expect(page.getByRole('listbox').locator('[data-value="run-99"]')).toHaveCount(0)
  await page.getByRole('listbox').locator('[data-value="run-21"]').click()
  await expect(page.getByRole('combobox', { name: 'Use this result as', exact: true })).toHaveCount(
    0,
  )
  await page
    .getByLabel('What changed? (optional)', { exact: true })
    .fill('Test a lower-cost route.')
  await page.getByRole('button', { name: 'Add run', exact: true }).click()
  expect(attached).toEqual([
    { run_id: 'run-21', role: 'candidate', hypothesis: 'Test a lower-cost route.' },
  ])
  await page
    .getByRole('heading', { name: experiment.name, exact: true })
    .evaluate((element) => element.scrollIntoView({ block: 'start' }))
  for (const button of [
    page.getByRole('button', { name: 'Evaluate starting recipe', exact: true }),
    page
      .getByRole('navigation', { name: 'Experiment pages' })
      .getByRole('button', { name: 'Previous', exact: true }),
    page.getByRole('button', { name: 'Add saved results', exact: true }),
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
  await page.getByRole('button', { name: 'Use saved results', exact: true }).click()
  await choose(page, 'Saved run', 'run-0')
  await page.getByRole('button', { name: 'Add run', exact: true }).click()
  expect(bodies).toHaveLength(2)
  expect(bodies[0]).toMatchObject({ name: 'Metadata only', idempotency_key: expect.any(String) })
  expect(bodies[1]).toEqual({ run_id: 'run-0', role: 'baseline', hypothesis: '' })
  await expect(
    page
      .getByRole('region', { name: 'Evaluation experiments', exact: true })
      .getByRole('button', { name: 'Compare results', exact: true }),
  ).toBeVisible()
  for (const action of [
    'Run single models',
    'Check routing first',
    'Evaluate starting recipe',
    'Test another recipe',
  ])
    await expect(page.getByRole('button', { name: action, exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Delete experiment', exact: true })).toBeEnabled()
  permissions = ['evaluation.read']
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await expect(page.getByRole('heading', { name: experiment.name, exact: true })).toBeVisible()
  await expect(
    page
      .getByRole('region', { name: 'Evaluation experiments', exact: true })
      .getByRole('button', { name: 'Compare results', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'Add run', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Delete experiment', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Run single models', exact: true })).toHaveCount(0)
  await page.goto('/evaluation?view=experiments')
  await expect(page.getByRole('heading', { name: 'Experiments', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Create experiment', exact: true })).toHaveCount(0)
  expect(bodies).toHaveLength(2)
})

test('confirms group-only deletion, blocks duplicate clicks and preserves saved runs after reload', async ({
  page,
}, testInfo) => {
  let deleted = false
  let release: (() => void) | undefined
  const mutations: string[] = []
  await setup(page, async (route) => {
    const path = new URL(route.request().url()).pathname
    if (route.request().method() !== 'GET') {
      mutations.push(`${route.request().method()} ${path}`)
      expect(route.request().postData()).toBeNull()
      await new Promise<void>((resolve) => {
        release = resolve
      })
      deleted = true
      return route.fulfill({
        json: { id, deleted: true, unlinked_runs: 1, runs_deleted: 0, model_requests: 0 },
      })
    }
    return route.fulfill({
      json: path.endsWith('/runs')
        ? {
            experiment,
            members: [
              {
                run_id: 'run-0',
                role: 'baseline',
                hypothesis: '',
                linked_at: experiment.created_at,
              },
            ],
            next_cursor: null,
            has_more: false,
          }
        : { experiments: deleted ? [] : [experiment], next_cursor: null, has_more: false },
    })
  })
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  const trigger = page.getByRole('button', { name: 'Delete experiment', exact: true })
  await trigger.click()
  const dialog = page.getByRole('alertdialog', { name: 'Delete experiment?' })
  await expect(dialog).toContainText('Saved runs, results, costs and artifacts will be kept.')
  await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
  expect(mutations).toEqual([])
  await trigger.click()
  await page.screenshot({ path: testInfo.outputPath('delete-experiment-desktop.png') })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: testInfo.outputPath('delete-experiment-mobile.png') })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await dialog.getByRole('button', { name: 'Delete experiment', exact: true }).click()
  await expect(dialog.getByRole('button', { name: 'Deleting…', exact: true })).toBeDisabled()
  await expect(dialog.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled()
  await expect.poll(() => mutations.length).toBe(1)
  release!()
  await expect(page).toHaveURL(/view=experiments$/)
  await expect(page.getByRole('heading', { name: 'Experiments', exact: true })).toBeVisible()
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Experiments', exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: /Balance recipe study/ })).toHaveCount(0)
  await page
    .getByRole('navigation', { name: 'Evaluation views' })
    .getByRole('button', { name: /^Runs/ })
    .click()
  await expect(page.getByRole('button', { name: 'Recipe 0', exact: true })).toBeVisible()
  expect(mutations).toEqual([`DELETE /api/sr-bench/v1/experiments/${id}`])
})

test('retains a lost deletion response and reconciles only the same experiment after explicit retry', async ({
  page,
}) => {
  const deletedIDs: string[] = []
  await setup(page, async (route) => {
    const path = new URL(route.request().url()).pathname
    if (route.request().method() === 'DELETE') {
      deletedIDs.push(path)
      if (deletedIDs.length === 1) return route.abort('failed')
      return route.fulfill({
        json: { id, deleted: true, unlinked_runs: 1, runs_deleted: 0, model_requests: 0 },
      })
    }
    return route.fulfill({
      json: path.endsWith('/runs')
        ? { experiment, members: [], next_cursor: null, has_more: false }
        : { experiments: [], next_cursor: null, has_more: false },
    })
  })
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await page.getByRole('button', { name: 'Delete experiment', exact: true }).click()
  const dialog = page.getByRole('alertdialog')
  await dialog.getByRole('button', { name: 'Delete experiment', exact: true }).click()
  await expect(dialog.getByRole('alert')).toContainText('Retry checks the same experiment.')
  expect(deletedIDs).toHaveLength(1)
  await expect(page).toHaveURL(new RegExp(`experiment=${id}`))
  await dialog.getByRole('button', { name: 'Retry deletion', exact: true }).click()
  await expect(page).toHaveURL(/view=experiments$/)
  expect(deletedIDs).toEqual(Array(2).fill(`/api/sr-bench/v1/experiments/${id}`))
})

test('refreshes an active-run rejection without repeating deletion and disables the action', async ({
  page,
}) => {
  let active = 0
  let deletions = 0
  await setup(page, async (route) => {
    if (route.request().method() === 'DELETE') {
      deletions += 1
      active = 1
      return route.fulfill({
        status: 409,
        json: {
          error: 'This experiment has active runs.',
          code: 'experiment_active_runs',
          active_run_count: 1,
        },
      })
    }
    return route.fulfill({
      json: {
        experiment: { ...experiment, active_run_count: active },
        members: [],
        next_cursor: null,
        has_more: false,
      },
    })
  })
  await page.goto(`/evaluation?view=experiments&experiment=${id}`)
  await page.getByRole('button', { name: 'Delete experiment', exact: true }).click()
  const dialog = page.getByRole('alertdialog')
  await dialog.getByRole('button', { name: 'Delete experiment', exact: true }).click()
  await expect(dialog.getByRole('alert')).toHaveText('This experiment has active runs.')
  await dialog.getByRole('button', { name: 'Refresh experiment', exact: true }).click()
  await expect(dialog).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Delete experiment', exact: true })).toBeDisabled()
  await expect(page.getByText('Wait for active runs to finish before deleting.')).toBeVisible()
  expect(deletions).toBe(1)
  await page.reload()
  await expect(page.getByRole('button', { name: 'Delete experiment', exact: true })).toBeDisabled()
  expect(deletions).toBe(1)
})

test('ignores a late deletion response after navigating to a different experiment', async ({
  page,
}) => {
  const second = { ...experiment, id: 'exp-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', name: 'Other study' }
  let release: (() => void) | undefined
  let responded = false
  await setup(page, async (route) => {
    const path = new URL(route.request().url()).pathname
    if (route.request().method() === 'DELETE') {
      await new Promise<void>((resolve) => {
        release = resolve
      })
      await route.fulfill({
        json: { id, deleted: true, unlinked_runs: 1, runs_deleted: 0, model_requests: 0 },
      })
      responded = true
      return
    }
    return route.fulfill({
      json: path.endsWith('/runs')
        ? {
            experiment: path.includes(second.id) ? second : experiment,
            members: [],
            next_cursor: null,
            has_more: false,
          }
        : { experiments: [experiment, second], next_cursor: null, has_more: false },
    })
  })
  await page.goto('/evaluation?view=experiments')
  await page.getByRole('button', { name: /Balance recipe study/ }).click()
  await page.getByRole('button', { name: 'Delete experiment', exact: true }).click()
  await page
    .getByRole('alertdialog')
    .getByRole('button', { name: 'Delete experiment', exact: true })
    .click()
  await expect.poll(() => Boolean(release)).toBe(true)
  await page.goBack()
  await page.getByRole('button', { name: /Other study/ }).click()
  release!()
  await expect.poll(() => responded).toBe(true)
  await expect(page).toHaveURL(new RegExp(`experiment=${second.id}`))
  await expect(page.getByRole('heading', { name: second.name, exact: true })).toBeVisible()
})
