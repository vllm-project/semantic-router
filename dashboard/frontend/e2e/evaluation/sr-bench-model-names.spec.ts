import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const canonicalModel = 'provider/connected-model-2026'
const target = {
  id: 'short-alias',
  kind: 'single',
  model: canonicalModel,
  base_url: 'http://localhost:8000/v1',
}
const manifest = {
  version: 'sr-bench-1.0',
  name: 'Frozen naming fixture',
  mode: 'live',
  profile: 'quick',
  seed: 42,
  targets: [target],
  limits: {},
  sampling: {},
  cases: [
    {
      id: 'case-one',
      benchmark: 'mmlu-pro',
      messages: [{ role: 'user', content: 'Fixture question' }],
    },
  ],
}
const run = {
  id: 'naming-run',
  status: 'completed',
  manifest,
  created_at: '2026-09-19T00:00:00Z',
  updated_at: '2026-09-19T00:00:01Z',
  progress: { total: 1, completed: 1, failed: 0 },
}
const metric = {
  id: target.id,
  total: 1,
  completed: 1,
  correct: 1,
  failed: 0,
  accuracy: 1,
  macro_accuracy: 1,
  cost_usd: 0.01,
  tokens: 10,
  selected_models: { [canonicalModel]: 1 },
  decisions: { route: 1 },
}

async function mockNaming(page: Page) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const path = new URL(route.request().url()).pathname.replace('/api/sr-bench/v1', '')
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: ['smoke', 'quick', 'standard'].map((id) => ({ id, purpose: id })),
        benchmarks: [{ id: 'mmlu-pro', title: 'MMLU-Pro', kind: 'capability' }],
      }
    else if (path === '/targets') body = { targets: [target] }
    else if (path === '/datasets') body = { datasets: [] }
    else if (path === '/datasets/selection')
      body = { profile: 'quick', seed: 42, split: 'dev', benchmarks: [], model_requests: 0 }
    else if (path === '/experiments') body = { experiments: [], has_more: false, next_cursor: null }
    else if (path === '/runs') body = { runs: [run] }
    else if (path === `/runs/${run.id}`) body = run
    else if (path === `/runs/${run.id}/report`)
      body = {
        version: 'sr-bench-1.0',
        run_id: run.id,
        status: 'completed',
        summary: { targets: [metric], wall_time_s: 1 },
        benchmarks: [{ ...metric, benchmark: 'mmlu-pro' }],
        limitations: [],
        provenance: {},
      }
    else if (path === `/runs/${run.id}/results`)
      body = {
        results: [
          {
            case_id: 'case-one',
            target_id: target.id,
            benchmark: 'mmlu-pro',
            status: 'completed',
            correct: true,
          },
        ],
        total: 1,
        next_cursor: null,
        limit: 100,
      }
    else if (path === `/runs/${run.id}/calls`)
      body = {
        calls: [
          {
            id: 'call-one',
            case_id: 'case-one',
            target_id: target.id,
            role: 'subject',
            status: 'completed',
            selected_model: canonicalModel,
          },
        ],
        total: 1,
        next_cursor: null,
        limit: 100,
      }
    else if (path === `/runs/${run.id}/events`)
      body = { events: [], next_cursor: null, has_more: false }
    else
      return route.fulfill({
        status: 404,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Unexpected fixture path' }),
      })
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
  })
}

test('create uses the connected model name while keeping target ID as selection identity', async ({
  page,
}) => {
  await mockNaming(page)
  await page.goto('/evaluation?view=new')
  await page.getByRole('combobox', { name: 'Add configured target', exact: true }).click()
  const option = page.locator('[role="option"][data-value="short-alias"]')
  await expect(option).toContainText(canonicalModel)
  await expect(option).not.toContainText('short-alias')
  await option.click()
  await expect(page.getByRole('group', { name: canonicalModel, exact: true })).toBeVisible()
  await expect(
    page.getByRole('region', { name: `${canonicalModel} request profile`, exact: true }),
  ).toBeVisible()
  await expect(page.getByText('short-alias', { exact: true })).toHaveCount(0)
})

test('saved result and call labels resolve the frozen model rather than its target alias', async ({
  page,
}) => {
  await mockNaming(page)
  await page.goto(`/evaluation?view=runs&run=${run.id}`)
  const results = page.getByRole('tabpanel', { name: 'Results', exact: true })
  await expect(results.getByRole('rowheader', { name: canonicalModel, exact: true })).toBeVisible()
  await expect(results.getByText(`${canonicalModel}: route`, { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Questions', exact: true }).click()
  const questions = page.getByRole('tabpanel', { name: 'Questions', exact: true })
  await expect(questions.getByRole('cell', { name: canonicalModel, exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Calls', exact: true }).click()
  const calls = page.getByRole('tabpanel', { name: 'Calls', exact: true })
  await expect(
    calls.getByRole('cell', { name: `case-one / ${canonicalModel}`, exact: true }),
  ).toBeVisible()
  await expect(calls.getByText('short-alias', { exact: true })).toHaveCount(0)
})
