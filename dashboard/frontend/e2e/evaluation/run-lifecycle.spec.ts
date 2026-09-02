import { expect, test } from '@playwright/test'

import { EVALUATION_RUN_IDS, evaluationRunID } from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import {
  captureEvaluationSurface,
  expectDialogBottomReachable,
  expectNoHorizontalOverflow,
} from './support/pageAssertions'
import { defaultEvaluationRuns, evaluationRun } from './support/runFixtures'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Run lifecycle', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('mutates controlled-pair members only through their aggregate lifecycle', async ({
    page,
  }) => {
    const pairID = evaluationRunID(900)
    const baselineID = evaluationRunID(901)
    const candidateID = evaluationRunID(902)
    const createdAt = '2026-08-31T01:00:00Z'
    const baseline = evaluationRun(
      baselineID,
      'Controlled pair control',
      'running',
      createdAt,
      'recipe',
      { controlled_pair: { pair_id: pairID, role: 'baseline' } },
    )
    const candidate = evaluationRun(
      candidateID,
      'Controlled pair treatment',
      'running',
      createdAt,
      'recipe',
      {
        baseline_run_id: baseline.id,
        controlled_pair: { pair_id: pairID, role: 'candidate' },
      },
    )
    const state = await mockEvaluationPlane(page, [candidate, baseline, ...defaultEvaluationRuns], {
      mutationDelayMs: 300,
      failFirstControlledPairCancel: true,
    })
    await page.goto(`/evaluation?view=runs&run=${candidate.id}`)

    const cancelPair = page.getByRole('button', { name: 'Cancel controlled comparison' })
    await expect(cancelPair).toHaveText('Cancel comparison')
    const defaultInspectorText = await page
      .getByRole('complementary', { name: 'Selected evaluation run' })
      .innerText()
    expect(defaultInspectorText).not.toContain(candidate.id)
    expect(defaultInspectorText).not.toContain(baseline.id)
    expect(defaultInspectorText).not.toContain(candidate.suite_ids[0])
    const technicalDetails = page
      .getByRole('complementary', { name: 'Selected evaluation run' })
      .locator('details')
      .filter({ has: page.getByText('Run ID', { exact: true }) })
    await technicalDetails.locator(':scope > summary').click()
    await expect(technicalDetails.getByText(candidate.id, { exact: true })).toBeVisible()
    await expect(technicalDetails.getByText(baseline.id, { exact: true })).toBeVisible()
    await expect(
      technicalDetails.getByText(candidate.suite_ids.join(', '), { exact: true }),
    ).toBeVisible()
    await technicalDetails.locator(':scope > summary').click()
    await expect(page.getByRole('button', { name: `Cancel ${candidate.name}` })).toHaveCount(0)
    await expect(page.getByRole('button', { name: `Delete ${candidate.name}` })).toHaveCount(0)

    await cancelPair.click()
    let dialog = page.getByRole('alertdialog')
    await expect(
      dialog.getByRole('heading', { name: 'Cancel controlled comparison?' }),
    ).toBeVisible()
    await expect(dialog).toContainText('Both runs stop together.')
    await expect(dialog).not.toContainText(pairID)
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(dialog).toHaveCount(0)
    expect(state.controlledPairCancelRequests).toHaveLength(0)

    await cancelPair.click()
    dialog = page.getByRole('alertdialog')
    await dialog.getByRole('button', { name: 'Cancel comparison', exact: true }).click()
    await expect(dialog.getByRole('alert')).toContainText(
      'temporary controlled-pair cancellation failure',
    )
    expect(state.controlledPairCancelRequests).toEqual([
      `/api/evaluation/v1/controlled-pairs/${pairID}/cancel`,
    ])

    await dialog.getByRole('button', { name: 'Cancel comparison', exact: true }).click()
    await expect(dialog).toHaveAttribute('aria-busy', 'true')
    await expect(dialog.getByRole('button', { name: 'Cancelling comparison…' })).toBeDisabled()
    await expect(dialog).toHaveCount(0)
    expect(state.getCancelCount()).toBe(0)
    await expect(
      page
        .getByRole('button', { name: `Open ${baseline.name} details` })
        .getByText('Cancelled', { exact: true }),
    ).toBeVisible()
    await expect(
      page
        .getByRole('button', { name: `Open ${candidate.name} details` })
        .getByText('Cancelled', { exact: true }),
    ).toBeVisible()

    const deletePair = page.getByRole('button', { name: 'Delete controlled comparison' })
    await expect(deletePair).toHaveText('Delete comparison')
    await deletePair.click()
    dialog = page.getByRole('alertdialog')
    await expect(
      dialog.getByRole('heading', { name: 'Delete controlled comparison?' }),
    ).toBeVisible()
    await expect(dialog).toContainText(
      'This permanently removes both runs and their reports from Evaluation.',
    )
    await expect(dialog).not.toContainText(pairID)
    const confirmation = dialog.getByRole('textbox', {
      name: /Enter DELETE COMPARISON to confirm/,
    })
    await confirmation.fill('DELETE COMPARISON')
    const ledgerRequestsBeforeDelete = state.getLedgerRequestCount()
    await dialog.getByRole('button', { name: 'Delete comparison', exact: true }).click()
    await expect(dialog).toHaveAttribute('aria-busy', 'true')
    await expect(dialog.getByRole('button', { name: 'Deleting comparison…' })).toBeDisabled()
    await expect(dialog).toHaveCount(0)

    expect(state.controlledPairDeleteRequests).toEqual([
      `/api/evaluation/v1/controlled-pairs/${pairID}`,
    ])
    expect(state.getDeleteCount()).toBe(0)
    await expect(page.getByRole('button', { name: `Open ${baseline.name} details` })).toHaveCount(0)
    await expect(page.getByRole('button', { name: `Open ${candidate.name} details` })).toHaveCount(
      0,
    )
    await expect.poll(state.getLedgerRequestCount).toBeGreaterThan(ledgerRequestsBeforeDelete)
    await expect.poll(() => new URL(page.url()).searchParams.get('run')).toBeNull()
  })

  test('does not let a delayed member read roll back aggregate pair cancellation', async ({
    page,
  }) => {
    const pairID = evaluationRunID(922)
    const baselineID = evaluationRunID(923)
    const candidateID = evaluationRunID(924)
    const createdAt = '2026-08-31T02:20:00Z'
    const baseline = evaluationRun(
      baselineID,
      'Delayed pair baseline',
      'running',
      createdAt,
      'recipe',
      { controlled_pair: { pair_id: pairID, role: 'baseline' } },
    )
    const candidate = evaluationRun(
      candidateID,
      'Delayed pair candidate',
      'running',
      createdAt,
      'recipe',
      {
        baseline_run_id: baselineID,
        controlled_pair: { pair_id: pairID, role: 'candidate' },
      },
    )
    const state = await mockEvaluationPlane(page, [candidate, baseline, ...defaultEvaluationRuns], {
      runDelayMs: 800,
      mutationDelayMs: 100,
    })
    await page.goto(`/evaluation?view=runs&run=${candidate.id}`)
    const cancelPair = page.getByRole('button', { name: 'Cancel controlled comparison' })
    await expect(cancelPair).toBeVisible()

    await page.getByRole('button', { name: 'Refresh evaluation runs' }).click()
    await expect
      .poll(() => state.runRequests.filter((id) => id === candidate.id).length)
      .toBeGreaterThanOrEqual(1)
    await cancelPair.click()
    await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel comparison' }).click()
    await expect.poll(() => state.controlledPairCancelRequests.length).toBe(1)
    const deletePair = page.getByRole('button', { name: 'Delete controlled comparison' })
    await expect(deletePair).toBeVisible()

    await page.waitForTimeout(900)
    await expect(deletePair).toBeVisible()
    await expect(cancelPair).toHaveCount(0)
  })

  test('uses aggregate capabilities when the selected pair member is already terminal', async ({
    page,
  }) => {
    const pairID = evaluationRunID(910)
    const baselineID = evaluationRunID(911)
    const candidateID = evaluationRunID(912)
    const createdAt = '2026-08-31T01:20:00Z'
    const baseline = evaluationRun(
      baselineID,
      'Completed controlled baseline',
      'completed',
      createdAt,
      'recipe',
      {
        mode: 'live',
        controlled_pair: { pair_id: pairID, role: 'baseline' },
      },
    )
    const candidate = evaluationRun(
      candidateID,
      'Running controlled candidate',
      'running',
      createdAt,
      'recipe',
      {
        baseline_run_id: baseline.id,
        controlled_pair: { pair_id: pairID, role: 'candidate' },
      },
    )
    const state = await mockEvaluationPlane(page, [baseline, candidate, ...defaultEvaluationRuns], {
      controlledPairGetDelayMs: 2_000,
      failFirstControlledPairGet: true,
    })
    await page.goto(`/evaluation?view=runs&run=${baseline.id}`)

    await expect(
      page.getByRole('button', { name: `Open report for ${baseline.name}` }),
    ).toBeVisible()
    await expect(page.getByText('Loading comparison actions…')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Cancel controlled comparison' })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Delete controlled comparison' })).toHaveCount(0)

    const pairError = page.getByRole('alert').filter({
      has: page.getByText(
        'Comparison actions could not be loaded. Existing run evidence remains available.',
        { exact: true },
      ),
    })
    await expect(pairError).toBeVisible()
    const pairBackendFailure = pairError.getByText('temporary controlled-pair state failure', {
      exact: true,
    })
    await expect(pairBackendFailure).not.toBeVisible()
    await pairError.locator('details[data-evaluation-technical-details="true"] > summary').click()
    await expect(pairBackendFailure).toBeVisible()
    await expect(page.getByRole('button', { name: 'Delete controlled comparison' })).toHaveCount(0)
    await pairError.getByRole('button', { name: 'Retry comparison actions' }).click()

    const cancelPair = page.getByRole('button', { name: 'Cancel controlled comparison' })
    await expect(cancelPair).toHaveText('Cancel comparison')
    await expect(page.getByRole('button', { name: 'Delete controlled comparison' })).toHaveCount(0)
    await expect
      .poll(() => state.controlledPairGetRequests)
      .toEqual([
        `/api/evaluation/v1/controlled-pairs/${pairID}`,
        `/api/evaluation/v1/controlled-pairs/${pairID}`,
      ])

    await page.getByRole('button', { name: 'Refresh evaluation runs' }).click()
    await expect(page.getByText('Refreshing comparison actions…')).toBeVisible()
    await expect(cancelPair).toBeVisible()
    await expect(cancelPair).toBeDisabled()
    await expect(page.getByText('Loading evaluation run')).toHaveCount(0)
    await expect(cancelPair).toBeEnabled()
    await expect.poll(() => state.controlledPairGetRequests.length).toBeGreaterThanOrEqual(3)
  })

  test('requires typed delete confirmation and preserves pending dialog state', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns, { mutationDelayMs: 400 })
    await page.goto(`/evaluation?view=runs&run=${EVALUATION_RUN_IDS.failed}`)

    await page.getByRole('button', { name: 'Delete Failed diagnostic' }).click()
    const dialog = page.getByRole('alertdialog')
    const confirmation = dialog.getByRole('textbox', {
      name: /Enter Failed diagnostic to confirm/,
    })
    const deleteButton = dialog.getByRole('button', { name: 'Delete run' })
    await expect(confirmation).toBeFocused()
    await captureEvaluationSurface(page, 'delete-dialog')
    await page.setViewportSize({ width: 390, height: 844 })
    await expect(dialog).toBeVisible()
    await expectNoHorizontalOverflow(page)
    await expectDialogBottomReachable(page, dialog)
    await captureEvaluationSurface(page, 'delete-dialog-mobile')
    await expect(deleteButton).toBeDisabled()
    await confirmation.fill('Failed')
    await expect(deleteButton).toBeDisabled()
    await confirmation.fill('Failed diagnostic')
    await expect(deleteButton).toBeEnabled()
    await deleteButton.click()
    await expect(dialog).toHaveAttribute('aria-busy', 'true')
    await expect(dialog.getByRole('button', { name: 'Deleting…' })).toBeDisabled()
    await expect(dialog.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Delete Failed diagnostic' })).toBeDisabled()

    await expect.poll(state.getDeleteCount).toBe(1)
    await expect(dialog).toHaveCount(0)
    await expect(page.getByRole('tabpanel')).toBeFocused()
    await expect(page.getByRole('button', { name: 'Open Failed diagnostic details' })).toHaveCount(
      0,
    )
  })

  test('keeps one SSE subscription and one event across a run refresh', async ({ page }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns, { runDelayMs: 750 })
    await page.goto(`/evaluation?view=runs&run=${EVALUATION_RUN_IDS.live}`)

    await expect.poll(state.getEventStreamCount).toBe(1)
    await expect(page.getByText('Executing routing track from SSE')).toHaveCount(1)
    await captureEvaluationSurface(page, 'runs-desktop')
    const detailRequestsBeforeRefresh = state.runRequests.filter(
      (id) => id === EVALUATION_RUN_IDS.live,
    ).length
    await page.getByRole('button', { name: 'Refresh evaluation runs' }).click()
    await expect
      .poll(() => state.runRequests.filter((id) => id === EVALUATION_RUN_IDS.live).length)
      .toBeGreaterThan(detailRequestsBeforeRefresh)
    await expect(page.getByRole('heading', { name: 'Live AMD validation' })).toBeVisible()
    await expect(page.getByText('Loading evaluation run', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Cancel Live AMD validation' })).toBeVisible()

    expect(state.getEventStreamCount()).toBe(1)
    await expect(page.getByText('Executing routing track from SSE')).toHaveCount(1)
    await expect(page.getByText('Refreshing details…', { exact: true })).toHaveCount(0)
  })

  test('requires an explicit retry after a server-closed event stream', async ({ page }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns, {
      eventStreamCloseOnce: true,
    })
    await page.goto(`/evaluation?view=runs&run=${EVALUATION_RUN_IDS.live}`)

    await expect.poll(state.getEventStreamCount).toBe(1)
    await expect(page.getByText('Updates unavailable', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Reconnect', exact: true }).click()
    await expect.poll(state.getEventStreamCount).toBe(2)
    await expect(page.getByText('Executing routing track from SSE')).toHaveCount(1)
    await expect(
      page.getByText('Evaluation event stream was closed by the server.', { exact: true }),
    ).toHaveCount(0)
  })
})
