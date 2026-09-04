import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'
import { defaultEvaluationRuns, evaluationCatalog, mockEvaluationPlane } from './support/evaluation'

const evalUser = {
  id: 'user-eval-1',
  email: 'eval@example.com',
  name: 'Eval User',
  role: 'read',
  permissions: [
    'config.read',
    'evaluation.read',
    'evaluation.run',
    'evaluation.write',
    'logs.read',
    'topology.read',
  ],
}

test.describe('Evaluation Plane', () => {
  test.beforeEach(async ({ page }) => {
    await mockAuthenticatedAppShell(page, {
      user: evalUser,
      settings: { readonlyMode: false, serverReadonly: false },
    })
  })

  test('shows the complete information architecture and eight-track coverage map', async ({
    page,
  }) => {
    await mockEvaluationPlane(page)
    await page.goto('/evaluation')

    await expect(page.getByRole('heading', { name: 'Evaluation', exact: true })).toBeVisible()
    await expect(page.getByText('reproducible evidence and promotion gates')).toBeVisible()
    for (const tab of ['Overview', 'New experiment', 'Runs', 'Reports', 'Compare']) {
      await expect(page.getByRole('tab', { name: tab, exact: true })).toBeVisible()
    }
    for (const track of evaluationCatalog.tracks) {
      await expect(
        page.getByRole('heading', { name: track.name, exact: true }).first(),
      ).toBeVisible()
    }
    await expect(page.getByText('Contract evaluation.v1')).toBeVisible()
    await expect(page.getByText('evaluation-release-gates.v1')).toBeVisible()
    await expect(page.getByText('7 change profiles')).toBeVisible()
  })

  test('creates only a catalog-targeted run with all reproducibility controls', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page)
    await page.goto('/evaluation')
    await page.getByRole('tab', { name: 'New experiment' }).click()

    await expect(page.getByText(/cannot supply its own execution address/i)).toBeVisible()
    await page.getByLabel('Change profile').selectOption({ label: 'Routing recipe' })
    const gateMatrix = page.getByLabel('G0–G9 gate applicability')
    await expect(gateMatrix.locator('article')).toHaveCount(10)
    await expect(gateMatrix.getByText('G0', { exact: true })).toBeVisible()
    await expect(gateMatrix.getByText('G9', { exact: true })).toBeVisible()
    await expect(gateMatrix.getByText('Live fidelity', { exact: true })).toBeVisible()
    await page.getByLabel('Experiment name').fill('Recipe v4 candidate')
    await page.getByLabel('Description').fill('Validate the full evaluation surface.')
    await page.getByLabel('Sample limit').fill('64')
    await page.getByLabel('Concurrency').fill('8')
    await page.getByLabel('Seed').fill('7')
    await page.getByRole('button', { name: 'Create and start' }).click()

    await expect.poll(() => state.createdRequests.length).toBe(1)
    expect(state.createdRequests[0]).toEqual({
      name: 'Recipe v4 candidate',
      description: 'Validate the full evaluation surface.',
      suite_ids: ['evaluation-smoke'],
      track_ids: [...evaluationCatalog.suites[0].track_ids],
      mode: 'replay',
      target_id: 'fixture',
      change_profile: 'recipe',
      sample_limit: 64,
      concurrency: 8,
      seed: 7,
      auto_start: false,
    })
    await expect.poll(state.getStartCount).toBe(1)
    expect(state.createdRequests[0]).not.toHaveProperty('endpoint')
    expect(state.createdRequests[0]).not.toHaveProperty('url')
    await expect(page.getByRole('tab', { name: 'Runs' })).toHaveAttribute('aria-selected', 'true')
  })

  test('renders rich report evidence, three cost ledgers, gates, provenance, and artifacts', async ({
    page,
  }) => {
    await mockEvaluationPlane(page)
    await page.goto('/evaluation')
    await page.getByRole('tab', { name: 'Reports' }).click()

    await expect(page.getByText('Evidence report · evaluation.v1')).toBeVisible()
    await expect(page.getByText('Profile recipe', { exact: true })).toBeVisible()
    await expect(page.getByText('Gate contract evaluation-release-gates.v1')).toBeVisible()
    for (const metric of ['Quality', 'P95 latency', 'Runtime cost', 'Capacity TCO']) {
      await expect(page.getByText(metric, { exact: true }).first()).toBeVisible()
    }
    await expect(page.getByText('runtime', { exact: true })).toBeVisible()
    await expect(page.getByText('evaluation overhead', { exact: true })).toBeVisible()
    await expect(page.getByText('capacity tco', { exact: true })).toBeVisible()
    const promotionGates = page
      .getByRole('heading', { name: 'Promotion gates' })
      .locator('..')
      .locator('..')
      .locator('..')
    await expect(promotionGates).toBeVisible()
    await expect(
      promotionGates
        .getByText('Required gate is not satisfied: unavailable evidence never counts as pass.', {
          exact: true,
        })
        .first(),
    ).toBeVisible()
    await expect(promotionGates.getByText('N = 4', { exact: true }).first()).toBeVisible()
    await expect(promotionGates.getByText(/Coverage 4\/4 \(100\.0%\)/).first()).toBeVisible()
    await expect(promotionGates.getByText('records.jsonl', { exact: true }).first()).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Provenance' })).toBeVisible()
    await expect(page.getByText('Change profile', { exact: true })).toBeVisible()
    await expect(page.getByText('Gate contract', { exact: true })).toBeVisible()
    await expect(page.getByText('sha256:policy')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Artifacts' })).toBeVisible()
    await expect(page.getByText('sha256:report-html', { exact: true })).toBeVisible()
    await expect(page.getByRole('link', { name: 'Download report.html' })).toHaveCount(0)
    await expect(
      page.getByRole('link', { name: 'Download failure-summary.json' }),
    ).toHaveAttribute(
      'href',
      '/api/evaluation/v1/runs/candidate-run/artifacts/failure-summary-json',
    )
    await expect(page.getByRole('link', { name: 'Download run-manifest.json' })).toHaveCount(0)
    await expect(page.getByText(/Collect qualified robustness evidence/)).toBeVisible()
  })

  test('compares candidate and baseline through the versioned comparison endpoint', async ({
    page,
  }) => {
    await mockEvaluationPlane(page)
    await page.goto('/evaluation')
    await page.getByRole('tab', { name: 'Compare' }).click()
    await page.getByRole('button', { name: 'Compare runs' }).click()

    await expect(page.getByText(/required robustness evidence is unavailable/)).toBeVisible()
    await expect(page.getByText('Comparison gates')).toBeVisible()
    await expect(
      page.getByText('Collect qualified robustness evidence before a guarded live trial.'),
    ).toBeVisible()
  })

  test('confirms cancellation and keeps unavailable evidence explicit', async ({ page }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns)
    await page.goto('/evaluation')
    await page.getByRole('tab', { name: 'Runs' }).click()
    await page.getByRole('button', { name: 'Cancel', exact: true }).click()

    await expect(page.getByRole('alertdialog')).toContainText('Partial evidence remains explicit')
    await expect(page.getByRole('alertdialog')).toContainText(
      'unavailable gates will not count as passed',
    )
    await page.getByRole('button', { name: 'Cancel run' }).click()
    await expect.poll(state.getCancelCount).toBe(1)
    await expect(
      page
        .locator('article')
        .filter({ hasText: 'Live AMD validation' })
        .getByText('Cancelled', { exact: true }),
    ).toBeVisible()
  })

  test('keeps one SSE subscription and deduplicated event across run refresh', async ({ page }) => {
    const state = await mockEvaluationPlane(page)
    await page.goto('/evaluation')
    await page.getByRole('tab', { name: 'Runs' }).click()
    await page.getByRole('button', { name: /Live AMD validation live-run/ }).click()

    await expect.poll(state.getEventStreamCount).toBe(1)
    await expect(page.getByText('Executing routing track from SSE')).toHaveCount(1)
    await page.getByRole('button', { name: 'Refresh evaluation runs' }).click()
    await page.waitForTimeout(500)

    expect(state.getEventStreamCount()).toBe(1)
    await expect(page.getByText('Executing routing track from SSE')).toHaveCount(1)
  })
})
