import { expect, test } from '@playwright/test'

import {
  EVALUATION_BASELINE_MOM_TARGET_ID,
  EVALUATION_MOM_TARGET_ID,
  EVALUATION_RUN_IDS,
  evaluationRunID,
} from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import { defaultEvaluationRuns, evaluationRun } from './support/runFixtures'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Run ledger', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('loads the run ledger incrementally without hiding the server total', async ({ page }) => {
    const runs = Array.from({ length: 12 }, (_, index) =>
      evaluationRun(
        evaluationRunID(100 + index),
        `Evaluation ${index + 1}`,
        'completed',
        `2026-08-${String(29 - index).padStart(2, '0')}T00:00:00Z`,
      ),
    )
    await mockEvaluationPlane(page, runs, { runPageSize: 5 })
    await page.goto('/evaluation?view=runs')

    await expect(page.getByText(/5 matching runs.*5 of 12 loaded/)).toBeVisible()
    await expect(
      page.getByText(
        'Search and filters cover only the 5 loaded runs. Load older records to search and filter the full history.',
        { exact: true },
      ),
    ).toBeVisible()
    await page.getByRole('button', { name: 'Load more', exact: true }).click()
    await expect(page.getByText(/10 matching runs.*10 of 12 loaded/)).toBeVisible()
    await page.getByRole('button', { name: 'Load more', exact: true }).click()
    await expect(page.getByText(/12 matching runs.*12 of 12 loaded/)).toBeVisible()
    await expect(page.getByRole('button', { name: 'Load more', exact: true })).toHaveCount(0)
    await expect(page.getByText(/Search and filters cover only/)).toHaveCount(0)
  })

  test('resolves paginated run, report, and comparison deep links by exact identity', async ({
    page,
  }) => {
    const recent = Array.from({ length: 50 }, (_, index) =>
      evaluationRun(
        evaluationRunID(200 + index),
        `Recent evaluation ${index + 1}`,
        'completed',
        `2026-08-${String(29 - (index % 20)).padStart(2, '0')}T00:00:00Z`,
      ),
    )
    const olderBaseline = evaluationRun(
      EVALUATION_RUN_IDS.olderBaseline,
      'Older production baseline',
      'completed',
      '2026-07-01T00:00:00Z',
    )
    const olderCandidate = evaluationRun(
      EVALUATION_RUN_IDS.olderCandidate,
      'Older routed candidate',
      'completed',
      '2026-07-02T00:00:00Z',
      'recipe',
      { baseline_run_id: olderBaseline.id },
    )
    const state = await mockEvaluationPlane(page, [...recent, olderCandidate, olderBaseline], {
      runPageSize: 50,
    })

    await page.goto(`/evaluation?view=runs&run=${EVALUATION_RUN_IDS.olderCandidate}`)
    await expect(page.getByRole('heading', { name: 'Older routed candidate' })).toBeVisible()
    expect(state.runRequests).toContain(EVALUATION_RUN_IDS.olderCandidate)

    await page.goto(`/evaluation?view=reports&report=${EVALUATION_RUN_IDS.olderCandidate}`)
    await expect(page.getByRole('heading', { name: 'Older routed candidate' })).toBeVisible()
    await expect(page.getByLabel('Run')).toHaveValue(EVALUATION_RUN_IDS.olderCandidate)

    await page.goto(
      `/evaluation?view=compare&baseline=${EVALUATION_RUN_IDS.olderBaseline}&candidate=${EVALUATION_RUN_IDS.olderCandidate}`,
    )
    await expect(page.getByLabel('Comparison candidate', { exact: true })).toHaveValue(
      EVALUATION_RUN_IDS.olderCandidate,
    )
    await expect(page.getByText('Older production baseline', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Compare results' }).click()
    await expect.poll(() => state.comparisonRequests.length).toBe(1)
    expect(state.comparisonRequests[0]).toEqual({
      baselineRunID: EVALUATION_RUN_IDS.olderBaseline,
      candidateRunID: EVALUATION_RUN_IDS.olderCandidate,
    })
  })

  test('refreshes an off-page selected run directly when its terminal event arrives', async ({
    page,
  }) => {
    const recent = Array.from({ length: 50 }, (_, index) =>
      evaluationRun(
        evaluationRunID(400 + index),
        `Recent terminal refresh ${index + 1}`,
        'completed',
        `2026-08-${String(29 - (index % 20)).padStart(2, '0')}T00:00:00Z`,
      ),
    )
    const offPageRun = evaluationRun(
      evaluationRunID(499),
      'Off-page live evaluation',
      'running',
      '2026-07-01T00:00:00Z',
    )
    const state = await mockEvaluationPlane(page, [...recent, offPageRun], {
      runPageSize: 50,
      completeRunOnEventStream: offPageRun.id,
    })

    await page.goto(`/evaluation?view=runs&run=${offPageRun.id}`)
    await expect(page.getByRole('heading', { name: offPageRun.name })).toBeVisible()
    await expect(
      page.getByRole('button', { name: `Open report for ${offPageRun.name}` }),
    ).toBeVisible()
    await expect
      .poll(() => state.runRequests.filter((runID) => runID === offPageRun.id).length)
      .toBeGreaterThanOrEqual(2)
    await expect(page.getByText(/50 matching runs.*50 of 51 loaded/)).toBeVisible()
  })

  test('resumes first-page polling after a load-more request fails', async ({ page }) => {
    const runs = Array.from({ length: 6 }, (_, index) =>
      evaluationRun(
        evaluationRunID(300 + index),
        `Polling evaluation ${index + 1}`,
        'completed',
        `2026-08-${String(29 - index).padStart(2, '0')}T00:00:00Z`,
      ),
    )
    const state = await mockEvaluationPlane(page, runs, {
      runPageSize: 5,
      failFirstLoadMore: true,
    })
    await page.goto('/evaluation?view=runs')

    await page.getByRole('button', { name: 'Load more', exact: true }).click()
    const refreshIssue = page.getByRole('status').filter({
      has: page.getByText('Run history could not refresh. Showing the last loaded run state.', {
        exact: true,
      }),
    })
    await expect(refreshIssue).toBeVisible()
    const backendFailure = refreshIssue.getByText('temporary ledger page failure', {
      exact: true,
    })
    await expect(backendFailure).not.toBeVisible()
    await refreshIssue
      .locator('details[data-evaluation-technical-details="true"] > summary')
      .click()
    await expect(backendFailure).toBeVisible()
    const requestCountAfterFailure = state.getLedgerRequestCount()
    await expect
      .poll(() => state.getLedgerRequestCount(), { timeout: 7_000 })
      .toBeGreaterThan(requestCountAfterFailure)
    await expect(page.getByText(/temporary ledger page failure/)).toHaveCount(0)
    await page.getByRole('button', { name: 'Load more', exact: true }).click()
    await expect(page.getByText(/6 matching runs.*6 of 6 loaded/)).toBeVisible()
  })

  test('keeps completed evidence identity honest while the newest report is loading', async ({
    page,
  }) => {
    await mockEvaluationPlane(page, defaultEvaluationRuns, { reportDelayMs: 2_000 })
    await page.goto('/evaluation')

    await expect(page.getByText('Loading report summary…', { exact: true })).toBeVisible()
    await expect(page.locator('#evaluation-readiness-title')).toHaveText('Candidate recipe')
    await expect(page.locator('#latest-evidence-title')).toHaveText('Candidate recipe')
    await expect(
      page.getByText(
        'Loading the newest completed report. No decision is shown until the result is ready.',
        { exact: true },
      ),
    ).toBeVisible()
    await expect(
      page.getByText('Establish the first evaluation baseline', { exact: true }),
    ).toHaveCount(0)
    await expect(page.getByText('No completed report yet', { exact: true })).toHaveCount(0)

    await expect(page.getByText('Loading report summary…', { exact: true })).toHaveCount(0)
    await expect(
      page.getByText(
        'Headline results are verified by the evaluation service. Open the full report for every measured outcome.',
        { exact: true },
      ),
    ).toBeVisible()
  })

  test('supports keyboard navigation across the evaluation tabs', async ({ page }) => {
    await mockEvaluationPlane(page)
    await page.goto('/evaluation')

    const overview = page.getByRole('tab', { name: 'Overview', exact: true })
    await overview.focus()
    await overview.press('End')
    await expect(page.getByRole('tab', { name: 'Compare', exact: true })).toHaveAttribute(
      'aria-selected',
      'true',
    )
    await expect.poll(() => new URL(page.url()).searchParams.get('view')).toBe('compare')

    const compare = page.getByRole('tab', { name: 'Compare', exact: true })
    await compare.focus()
    await compare.press('Home')
    await expect(overview).toHaveAttribute('aria-selected', 'true')
    await expect.poll(() => new URL(page.url()).searchParams.get('view')).toBeNull()

    await overview.focus()
    await overview.press('ArrowRight')
    await expect(page.getByRole('tab', { name: 'New experiment', exact: true })).toHaveAttribute(
      'aria-selected',
      'true',
    )
  })

  test('keeps live Mixture deployment targets distinct while preserving their server IDs', async ({
    page,
  }) => {
    await mockEvaluationPlane(page)
    await page.goto('/evaluation?view=new')

    await page
      .getByRole('radio', {
        name: 'Live: evaluate a registered Mixture.',
        exact: true,
      })
      .check()
    const target = page.getByLabel('Mixture to evaluate')
    expect(await target.locator('option').allTextContents()).toEqual([
      'Select Mixture',
      'test-mom · Baseline',
      'test-mom · Candidate',
    ])
    await target.selectOption(EVALUATION_BASELINE_MOM_TARGET_ID)
    await expect(target).toHaveValue(EVALUATION_BASELINE_MOM_TARGET_ID)
    await target.selectOption(EVALUATION_MOM_TARGET_ID)
    await expect(target).toHaveValue(EVALUATION_MOM_TARGET_ID)
  })
})
