import { expect, test } from '@playwright/test'

import { EVALUATION_RUN_IDS, evaluationRunID } from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import { captureEvaluationSurface, expectDialogBottomReachable } from './support/pageAssertions'
import { defaultEvaluationRuns, evaluationRun } from './support/runFixtures'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Run resilience', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('keeps quarantined run evidence visible and blocks partial-ledger decisions', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns, {
      ledgerWarningCount: 3,
      ledgerWarnings: [
        {
          code: 'corrupt_run_bundle',
          evidence_id: 'bundle-entry-7f9d2a',
          evidence_file: 'status.json',
          message: 'Durable run status evidence is unreadable or invalid and has been quarantined.',
        },
      ],
    })
    await page.goto(
      `/evaluation?view=compare&baseline=${EVALUATION_RUN_IDS.baseline}&candidate=${EVALUATION_RUN_IDS.candidate}`,
    )

    await expect(page.getByText('Some saved runs could not be read', { exact: true })).toBeVisible()
    await expect(page.getByText(/3 saved runs are excluded/)).toBeVisible()
    await expect(
      page.getByText('Showing 1 of 3 warning details returned by run history.', { exact: true }),
    ).toBeVisible()
    await expect(page.getByText('bundle-entry-7f9d2a', { exact: true })).not.toBeVisible()
    const warningSummary = page.locator('details > summary').filter({
      has: page.getByText('Technical details · 1', { exact: true }),
    })
    await warningSummary.click()
    await expect(page.getByText('bundle-entry-7f9d2a', { exact: true })).toBeVisible()
    await expect(page.getByText(/status\.json: Durable run status evidence/)).toBeVisible()
    await expect(page.getByLabel('Comparison candidate', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Compare results' })).toHaveCount(0)
    await expect(page.getByText(/Baseline selection and comparison are paused/)).toBeVisible()
    expect(state.comparisonRequests).toHaveLength(0)

    await page.getByRole('tab', { name: 'New experiment', exact: true }).click()
    await expect(page.getByText('Some saved runs could not be read', { exact: true })).toBeVisible()
    await expect(page.getByLabel('Baseline run')).toBeDisabled()
    await expect(
      page.getByText('Baseline selection is paused until unreadable saved runs are repaired.', {
        exact: true,
      }),
    ).toBeVisible()
  })

  test('keeps cancellation modal and controls pending until the server responds', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns, {
      mutationDelayMs: 400,
      failFirstCancel: true,
    })
    await page.goto(`/evaluation?view=runs&run=${EVALUATION_RUN_IDS.live}`)

    const cancelTrigger = page.getByRole('button', { name: 'Cancel Live AMD validation' })
    await cancelTrigger.click()
    const dialog = page.getByRole('alertdialog')
    await expect(dialog).toContainText('Execution stops and no completed report is created.')
    await expect(dialog.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused()
    await page.keyboard.press('Shift+Tab')
    await expect(dialog.getByRole('button', { name: 'Cancel run' })).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(dialog.getByRole('button', { name: 'Cancel', exact: true })).toBeFocused()
    await page.keyboard.press('Escape')
    await expect(dialog).toHaveCount(0)
    await expect(cancelTrigger).toBeFocused()

    await cancelTrigger.click()
    await expectDialogBottomReachable(page, dialog)
    await captureEvaluationSurface(page, 'cancel-dialog')
    await dialog.getByRole('button', { name: 'Cancel run' }).click()
    const dialogError = dialog.getByRole('alert')
    await expect(dialogError).toContainText('temporary cancellation failure')
    await expect(page.locator('[role="alert"]')).toHaveCount(1)
    await expect(dialog.getByRole('button', { name: 'Cancel run' })).toBeEnabled()

    await dialog.getByRole('button', { name: 'Cancel run' }).click()
    await expect(dialog).toHaveAttribute('aria-busy', 'true')
    await expect(dialog.getByRole('button', { name: 'Cancelling…' })).toBeDisabled()
    await expect(dialog.getByRole('button', { name: 'Cancel', exact: true })).toBeDisabled()
    await expect(cancelTrigger).toBeDisabled()
    await page.keyboard.press('Escape')
    await expect(dialog).toBeVisible()

    await expect.poll(state.getCancelCount).toBe(1)
    await expect(dialog).toHaveCount(0)
    await expect(page.getByRole('tabpanel')).toBeFocused()
    const inspector = page.locator('aside').filter({
      has: page.getByRole('heading', { name: 'Live AMD validation' }),
    })
    await expect(inspector.getByText('Cancelled', { exact: true })).toBeVisible()
  })

  test('does not let a delayed detail read roll back a started run', async ({ page }) => {
    const pending = evaluationRun(
      evaluationRunID(920),
      'Delayed start fixture',
      'pending',
      '2026-08-31T02:00:00Z',
    )
    const state = await mockEvaluationPlane(page, [pending, ...defaultEvaluationRuns], {
      runDelayMs: 800,
      mutationDelayMs: 100,
    })
    await page.goto(`/evaluation?view=runs&run=${pending.id}`)

    await page.getByRole('button', { name: 'Refresh evaluation runs' }).click()
    await expect
      .poll(() => state.runRequests.filter((id) => id === pending.id).length)
      .toBeGreaterThanOrEqual(1)
    await page.getByRole('button', { name: `Start ${pending.name}` }).click()
    await expect.poll(state.getStartCount).toBe(1)
    const cancel = page.getByRole('button', { name: `Cancel ${pending.name}` })
    await expect(cancel).toBeVisible()

    await page.waitForTimeout(900)
    await expect(cancel).toBeVisible()
    await expect(page.getByRole('button', { name: `Start ${pending.name}` })).toHaveCount(0)
  })

  test('does not let a delayed detail read roll back a cancelled run', async ({ page }) => {
    const running = evaluationRun(
      evaluationRunID(921),
      'Delayed cancel fixture',
      'running',
      '2026-08-31T02:10:00Z',
    )
    const state = await mockEvaluationPlane(page, [running, ...defaultEvaluationRuns], {
      runDelayMs: 800,
      mutationDelayMs: 100,
    })
    await page.goto(`/evaluation?view=runs&run=${running.id}`)

    await page.getByRole('button', { name: 'Refresh evaluation runs' }).click()
    await expect
      .poll(() => state.runRequests.filter((id) => id === running.id).length)
      .toBeGreaterThanOrEqual(1)
    await page.getByRole('button', { name: `Cancel ${running.name}` }).click()
    await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel run' }).click()
    await expect.poll(state.getCancelCount).toBe(1)
    const deleteRun = page.getByRole('button', { name: `Delete ${running.name}` })
    await expect(deleteRun).toBeVisible()

    await page.waitForTimeout(900)
    await expect(deleteRun).toBeVisible()
    await expect(page.getByRole('button', { name: `Cancel ${running.name}` })).toHaveCount(0)
  })
})
