import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'
import type { Manifest, Target } from '../../src/components/sr-bench/types'

const mom: Target = {
  id: 'mom',
  kind: 'mom',
  model: 'recipe',
  base_url: 'http://localhost:8000/v1',
  capture_recipe: true,
  max_inference_calls: 1,
}
const dataset = {
  id: 'quick',
  path: '/prepared/quick.jsonl',
  sha256: 'a'.repeat(64),
  profile: 'quick',
  case_count: 1,
  seed: 42,
  benchmarks: ['mmlu-pro'],
}

async function setup(page: Page) {
  await mockAuthenticatedAppShell(page)
  const starts: Array<{ manifest: Manifest; idempotency_key: string }> = []
  let started: Record<string, unknown> | null = null
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname.replace('/api/sr-bench/v1', '')
    let body: unknown
    if (path === '/catalog')
      body = {
        version: 'sr-bench-1.0',
        profiles: ['smoke', 'quick', 'standard'].map((id) => ({ id, purpose: id })),
        benchmarks: [{ id: 'mmlu-pro', title: 'MMLU-Pro', kind: 'capability' }],
      }
    else if (path === '/datasets') body = { datasets: [dataset] }
    else if (path === '/datasets/selection')
      body = {
        profile: url.searchParams.get('profile'),
        seed: 42,
        split: 'dev',
        model_requests: 0,
        benchmarks: [
          {
            id: 'mmlu-pro',
            title: 'MMLU-Pro',
            eligible: true,
            case_count: 1,
            source_ids: [dataset.id],
            reason: null,
          },
        ],
      }
    else if (path === '/datasets/compose') body = { dataset }
    else if (path === '/targets') body = { targets: [mom] }
    else if (path === '/plans')
      body = {
        status: 'validated',
        total: 1,
        plan_sha256: 'b'.repeat(64),
        manifest: request.postDataJSON().manifest,
      }
    else if (path === '/runs' && request.method() === 'POST') {
      starts.push(request.postDataJSON())
      // A one-question local run can finish before its detail is first read.
      started = {
        id: 'fresh-run',
        status: 'completed',
        created_at: new Date().toISOString(),
        manifest: {
          ...request.postDataJSON().manifest,
          cases: [{ id: 'a', benchmark: 'mmlu-pro' }],
        },
        progress: { total: 1, completed: 1, failed: 0 },
      }
      body = started
    } else if (path === '/runs') body = { runs: started ? [started] : [] }
    else if (path === '/runs/fresh-run') body = started
    else if (path.endsWith('/report'))
      body = { summary: { targets: [] }, benchmarks: [], provenance: {}, limitations: [] }
    else if (path.endsWith('/results')) body = { results: [], total: 0, next_cursor: null }
    else if (path.endsWith('/calls')) body = { calls: [], total: 0, next_cursor: null }
    else if (path.endsWith('/events')) body = { events: [], next_cursor: null }
    else return route.fulfill({ status: 404, json: { error: 'Unexpected fixture request' } })
    await route.fulfill({ json: body })
  })
  return { starts }
}

test('a repeated Start click opens the new run instead of a re-run composer', async ({ page }) => {
  const state = await setup(page)
  await page.goto('/observe/evaluations?view=new')
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  await page.locator(`[data-target-id="${mom.id}"]`).click()
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  await page.getByRole('button', { name: 'Review plan', exact: true }).click()
  await expect(
    page.getByRole('heading', { name: 'Plan ready for review', exact: true }),
  ).toBeVisible()
  const box = (await page
    .getByRole('button', { name: 'Start evaluation', exact: true })
    .boundingBox())!
  const x = box.x + box.width / 2
  const y = box.y + box.height / 2
  await page.mouse.click(x, y)
  await expect(page).toHaveURL(/[?&]view=runs&run=fresh-run$/)
  const rerun = page.getByRole('button', { name: 'Re-run', exact: true })
  await expect(rerun).toBeVisible()
  // Re-run renders in the footer spot Start just left; a second click there
  // must not replace the new run with a composer for the same questions.
  await rerun.click()
  await page.waitForTimeout(300)
  await expect(page).toHaveURL(/[?&]view=runs&run=fresh-run$/)
  await expect(page.getByText('Re-run evaluation', { exact: true })).toHaveCount(0)
  expect(state.starts).toHaveLength(1)
  // The guard is brief; a deliberate Re-run still works afterwards.
  await page.waitForTimeout(1500)
  await rerun.click()
  await expect(page).toHaveURL(/[?&]rerun=fresh-run/)
  expect(state.starts).toHaveLength(1)
})
