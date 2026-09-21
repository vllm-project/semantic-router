import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const experimentID = 'exp-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
const created = '2026-01-01T00:00:00Z'
const run = (id: string, role = 'baseline', status = 'running') => ({
  id,
  role,
  status,
  created_at: created,
  updated_at: created,
  progress: { total: 36, completed: 3, failed: 1 },
  manifest: {
    version: 'sr-bench-1.0',
    name: `Saved ${role} ${id}`,
    mode: 'live',
    profile: 'quick',
    seed: 42,
    targets: [{ id: 'model', kind: role === 'baseline' ? 'single' : 'mom', model: 'model' }],
    sampling: { temperature: 1 },
    limits: {},
    ...(role === 'recovery' ? { recovery: { parent_run_id: 'run-reference' } } : {}),
  },
})

async function fixture(page: Page, savedRuns: ReturnType<typeof run>[], reader = false) {
  await mockAuthenticatedAppShell(
    page,
    reader
      ? {
          user: {
            id: 'reader',
            email: 'reader@example.com',
            name: 'Reader',
            role: 'read',
            permissions: ['evaluation.read'],
          },
        }
      : {},
  )
  const state = {
    runs: savedRuns,
    members: savedRuns.map((item) => item.id),
    splitMembership: false,
    membershipReads: 0,
    writes: [] as string[],
  }
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    if (request.method() !== 'GET') state.writes.push(`${request.method()} ${path}`)
    if (path === '/catalog') return route.fulfill({ json: { profiles: [], benchmarks: [] } })
    if (path === '/targets') return route.fulfill({ json: { targets: [] } })
    if (path === '/datasets') return route.fulfill({ json: { datasets: [] } })
    if (path === '/runs') return route.fulfill({ json: { runs: state.runs } })
    if (path === `/experiments/${experimentID}/runs`) {
      state.membershipReads++
      const firstPage = state.splitMembership && url.searchParams.get('after') !== '20'
      const members = state.splitMembership
        ? firstPage
          ? state.members.slice(0, 1)
          : state.members.slice(1)
        : state.members
      return route.fulfill({
        json: {
          experiment: {
            id: experimentID,
            name: 'Reference progress study',
            created_at: created,
            updated_at: created,
            run_count: state.members.length,
            active_run_count: state.runs.filter(
              (item) =>
                state.members.includes(item.id) && ['queued', 'running'].includes(item.status),
            ).length,
          },
          members: members.map((id) => ({
            run_id: id,
            role: state.runs.find((item) => item.id === id)!.role,
            hypothesis: '',
            linked_at: created,
          })),
          next_cursor: firstPage ? 20 : null,
          has_more: firstPage,
        },
      })
    }
    const selected = state.runs.find((item) => path.startsWith(`/runs/${item.id}`))
    if (selected && path === `/runs/${selected.id}`) return route.fulfill({ json: selected })
    if (selected && path.endsWith('/report'))
      return route.fulfill({
        json: {
          summary: { targets: [], wall_time_s: 60, total_spend_usd: null },
          benchmarks: [],
          limitations: [],
          provenance: {},
        },
      })
    if (selected && /\/(calls|results|events)$/.test(path))
      return route.fulfill({
        json: { [path.split('/').at(-1)!]: [], total: 0, next_cursor: null },
      })
    return route.fulfill({ status: 404, json: { error: 'Unexpected progress fixture read' } })
  })
  return state
}

for (const viewport of [
  { name: 'desktop', width: 1500, height: 1000, reader: false },
  { name: 'mobile reader', width: 390, height: 844, reader: true },
]) {
  test(`opens an existing active reference instead of promoting another baseline on ${viewport.name}`, async ({
    page,
  }, testInfo) => {
    await page.setViewportSize(viewport)
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    const state = await fixture(page, [run('run-reference')], viewport.reader)
    await page.goto(`/evaluation?view=experiments&experiment=${experimentID}`)
    const workspace = page.getByRole('region', { name: 'Evaluation experiments', exact: true })
    await expect(
      workspace.getByRole('heading', { name: 'Reference evaluation in progress' }),
    ).toBeVisible()
    const progress = workspace.getByRole('region', { name: 'Active evaluation', exact: true })
    await expect(progress).toContainText('3 / 36 attempts completed')
    await expect(progress).toContainText('1 failed')
    await expect(progress.getByText('running', { exact: true })).toBeVisible()
    await expect(workspace.locator('button[class*="_primary_"]')).toHaveAccessibleName('Open run')
    await expect(workspace.getByRole('button', { name: 'Run single models' })).toHaveCount(0)
    await expect(
      workspace.getByRole('heading', { name: 'Choose your reference results' }),
    ).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Create evaluation', exact: true })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath('active-reference.png') })
    await progress.getByRole('button', { name: 'Open run', exact: true }).click()
    await expect(page).toHaveURL(/view=runs.*run=run-reference/)
    expect(state.writes).toEqual([])
    expect(errors).toEqual([])
  })
}

for (const role of ['candidate', 'recovery']) {
  test(`prioritizes an active ${role} while retaining completed-reference actions`, async ({
    page,
  }) => {
    const completed = (id: string, role: string) => ({
      ...run(id, role, 'completed'),
      progress: { total: 36, completed: 36, failed: 0 },
    })
    const state = await fixture(page, [
      completed('run-reference', 'baseline'),
      completed('run-initial', 'initial'),
      run('run-active', role),
    ])
    await page.goto(`/evaluation?view=experiments&experiment=${experimentID}`)
    const workspace = page.getByRole('region', { name: 'Evaluation experiments', exact: true })
    await expect(
      workspace.getByRole('heading', { name: 'Evaluation in progress', exact: true }),
    ).toBeVisible()
    await expect(workspace.getByRole('region', { name: 'Active evaluation' })).toContainText(
      role === 'recovery' ? 'Recovery attempt' : 'Recipe version',
    )
    await expect(workspace.locator('button[class*="_primary_"]')).toHaveAccessibleName('Open run')
    for (const name of ['Compare results', 'Test another recipe', 'Check routing first'])
      await expect(workspace.getByRole('button', { name, exact: true })).toBeEnabled()
    await workspace.getByRole('button', { name: 'Test another recipe', exact: true }).click()
    await expect(page).toHaveURL(
      new RegExp(`view=new&baseline=run-reference&experiment=${experimentID}&role=candidate`),
    )
    expect(state.writes).toEqual([])
  })
}

test('uses complete membership and lets the user choose among active runs without including outsiders', async ({
  page,
}) => {
  const state = await fixture(page, [
    run('run-version', 'candidate'),
    run('run-reference'),
    run('run-recovery', 'recovery', 'queued'),
    run('run-outside'),
  ])
  state.members = ['run-version', 'run-reference', 'run-recovery']
  state.splitMembership = true
  await page.goto(`/evaluation?view=experiments&experiment=${experimentID}`)
  const progress = page.getByRole('region', { name: 'Active evaluation', exact: true })
  const select = progress.getByRole('combobox', { name: 'Active run', exact: true })
  await expect(select).toContainText('Saved baseline run-reference')
  await expect(progress).toContainText('3 active runs')
  await select.click()
  await expect(page.getByRole('option')).toHaveCount(3)
  await expect(page.getByRole('option', { name: /run-outside/ })).toHaveCount(0)
  await page.getByRole('option', { name: /Saved recovery run-recovery/ }).click()
  await expect(progress).toContainText('Recovery attempt')
  await expect(progress.getByText('queued', { exact: true })).toBeVisible()
  await progress.getByRole('button', { name: 'Open run', exact: true }).click()
  await expect(page).toHaveURL(/view=runs.*run=run-recovery/)
  expect(state.writes).toEqual([])
})

test('uses existing inventory polling to advance from active reference to recipe setup', async ({
  page,
}) => {
  const state = await fixture(page, [run('run-reference'), run('run-outside')])
  state.members = ['run-reference']
  await page.clock.install({ time: new Date(created) })
  await page.clock.pauseAt(new Date(created))
  await page.goto(`/evaluation?view=experiments&experiment=${experimentID}`)
  await expect(page.getByRole('region', { name: 'Active evaluation' })).toBeVisible()
  const membershipReads = state.membershipReads
  state.runs[0].status = 'completed'
  state.runs[0].progress = { total: 36, completed: 36, failed: 0 }
  await page.clock.fastForward(5000)
  await expect(
    page.getByRole('heading', { name: 'Test a recipe on the same questions' }),
  ).toBeVisible()
  await expect(page.getByRole('region', { name: 'Active evaluation' })).toHaveCount(0)
  await expect(page.getByText('1 saved run', { exact: true })).toBeVisible()
  await expect(page.getByText('1 saved run · 1 active', { exact: true })).toHaveCount(0)
  await expect(
    page.getByRole('button', { name: 'Evaluate starting recipe', exact: true }),
  ).toBeEnabled()
  expect(state.membershipReads).toBe(membershipReads)
  expect(state.writes).toEqual([])
})
