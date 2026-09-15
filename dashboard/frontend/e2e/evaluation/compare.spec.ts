import { expect, test } from '@playwright/test'

import {
  EVALUATION_BASELINE_MOM_TARGET_ID,
  EVALUATION_MOM,
  EVALUATION_MOM_TARGET_ID,
  EVALUATION_RUN_IDS,
  evaluationRunID,
} from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import { captureEvaluationSurface, expectProductEvaluationLanguage } from './support/pageAssertions'
import {
  expectResponsiveEvaluationSurface,
  responsiveEvaluationSurfaces,
} from './support/responsiveAssertions'
import { defaultEvaluationRuns, evaluationRun } from './support/runFixtures'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Compare', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('changes a comparison candidate and its matching baseline atomically', async ({ page }) => {
    const secondBaseline = evaluationRun(
      EVALUATION_RUN_IDS.secondBaseline,
      'Second baseline',
      'completed',
      '2026-08-26T00:00:00Z',
    )
    const secondCandidate = evaluationRun(
      EVALUATION_RUN_IDS.secondCandidate,
      'Second candidate',
      'completed',
      '2026-08-29T12:00:00Z',
      'recipe',
      { baseline_run_id: secondBaseline.id },
    )
    await mockEvaluationPlane(page, [secondCandidate, secondBaseline, ...defaultEvaluationRuns])
    await page.goto(
      `/evaluation?view=compare&baseline=${EVALUATION_RUN_IDS.baseline}&candidate=${EVALUATION_RUN_IDS.candidate}`,
    )

    await page.getByRole('button', { name: 'Compare results' }).click()
    await expect(page.getByRole('table', { name: 'Paired comparison metrics' })).toBeVisible()
    await page.getByLabel('Comparison candidate', { exact: true }).selectOption(secondCandidate.id)

    await expect(page.getByRole('table', { name: 'Paired comparison metrics' })).toHaveCount(0)
    await expect(
      page.getByText('Choose a candidate, then calculate its paired comparison.'),
    ).toBeVisible()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('candidate'))
      .toBe(secondCandidate.id)
    await expect
      .poll(() => new URL(page.url()).searchParams.get('baseline'))
      .toBe(secondBaseline.id)
    await expect(page.getByText(secondBaseline.name, { exact: true })).toBeVisible()
  })

  test('rejects controlled-pair cohort order drift and missing Mixture identity', async ({
    page,
  }) => {
    const orderedPairID = evaluationRunID(950)
    const orderedBaseline = evaluationRun(
      evaluationRunID(951),
      'Ordered baseline',
      'completed',
      '2026-08-29T13:00:00Z',
      'recipe',
      {
        mode: 'live',
        target_id: EVALUATION_BASELINE_MOM_TARGET_ID,
        mixture: EVALUATION_MOM,
        suite_ids: ['live-mom-core', 'normalized-promotion-cohort'],
        track_ids: ['routing', 'joint'],
        controlled_pair: { pair_id: orderedPairID, role: 'baseline' },
      },
    )
    const reorderedCandidate = evaluationRun(
      evaluationRunID(952),
      'Reordered candidate',
      'completed',
      '2026-08-29T13:01:00Z',
      'recipe',
      {
        ...orderedBaseline,
        id: evaluationRunID(952),
        client_request_id: evaluationRunID(952),
        name: 'Reordered candidate',
        target_id: EVALUATION_MOM_TARGET_ID,
        baseline_run_id: orderedBaseline.id,
        suite_ids: [...orderedBaseline.suite_ids].reverse(),
        track_ids: [...orderedBaseline.track_ids].reverse(),
        controlled_pair: { pair_id: orderedPairID, role: 'candidate' },
      },
    )
    const missingMixturePairID = evaluationRunID(953)
    const missingMixtureBaseline = evaluationRun(
      evaluationRunID(954),
      'Missing Mixture baseline',
      'completed',
      '2026-08-29T14:00:00Z',
      'recipe',
      {
        mode: 'live',
        target_id: EVALUATION_BASELINE_MOM_TARGET_ID,
        mixture: undefined,
        controlled_pair: { pair_id: missingMixturePairID, role: 'baseline' },
      },
    )
    const missingMixtureCandidate = evaluationRun(
      evaluationRunID(955),
      'Missing Mixture candidate',
      'completed',
      '2026-08-29T14:01:00Z',
      'recipe',
      {
        ...missingMixtureBaseline,
        id: evaluationRunID(955),
        client_request_id: evaluationRunID(955),
        name: 'Missing Mixture candidate',
        target_id: EVALUATION_MOM_TARGET_ID,
        baseline_run_id: missingMixtureBaseline.id,
        controlled_pair: { pair_id: missingMixturePairID, role: 'candidate' },
      },
    )
    await mockEvaluationPlane(page, [
      reorderedCandidate,
      orderedBaseline,
      missingMixtureCandidate,
      missingMixtureBaseline,
    ])
    await page.goto('/evaluation?view=compare')

    await expect(page.getByLabel('Comparison candidate', { exact: true })).toHaveCount(0)
    for (const [baselineRunID, candidateRunID] of [
      [orderedBaseline.id, reorderedCandidate.id],
      [missingMixtureBaseline.id, missingMixtureCandidate.id],
    ]) {
      const status = await page.evaluate(
        async ({ baselineID, candidateID }) =>
          (
            await fetch(
              `/api/evaluation/v1/compare?baseline_run_id=${baselineID}&candidate_run_id=${candidateID}`,
            )
          ).status,
        { baselineID: baselineRunID, candidateID: candidateRunID },
      )
      expect(status).toBe(400)
    }
  })

  test('preserves comparison lineage and colors deltas according to metric direction', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page)
    await page.goto(
      `/evaluation?view=compare&baseline=${EVALUATION_RUN_IDS.baseline}&candidate=${EVALUATION_RUN_IDS.candidate}`,
    )

    await expect(
      page.getByRole('heading', { name: 'Compare a candidate with its baseline' }),
    ).toBeVisible()
    const candidates = page.getByLabel('Comparison candidate', { exact: true })
    expect(await candidates.locator('option').allTextContents()).toEqual([
      'Choose a compatible candidate',
      'Candidate recipe',
    ])
    await expect(page.getByText('Production baseline', { exact: true })).toBeVisible()
    await captureEvaluationSurface(page, 'comparison-setup-desktop')

    await page.getByRole('button', { name: 'Compare results' }).click()
    await expect.poll(() => state.comparisonRequests.length).toBe(1)
    expect(state.comparisonRequests[0]).toEqual({
      baselineRunID: EVALUATION_RUN_IDS.baseline,
      candidateRunID: EVALUATION_RUN_IDS.candidate,
    })

    const table = page.getByRole('table', { name: 'Paired comparison metrics' })
    const quality = table.locator('tr[data-metric-id="joint.realized_quality"]')
    await expect(quality).toContainText('Higher is better')
    await expect(quality.locator('strong[class*="delta_positive"]')).toHaveText('+3.0%')
    const latency = table.locator('tr[data-metric-id="capacity.latency_p95_ms"]')
    await expect(latency).toContainText('Lower is better')
    await expect(latency.locator('strong[class*="delta_positive"]')).toHaveText('−28 ms')
    const statistics = page.getByRole('table', { name: 'Paired outcome comparison' })
    const normalizedRegret = statistics.locator('tr[data-statistic-id="joint.normalized_regret"]')
    await expect(normalizedRegret).toContainText('Normalized quality gap')
    await expect(normalizedRegret).toContainText('Not estimable')
    await expect(normalizedRegret).toContainText(
      'Needs at least 20 independent case units; observed 4.',
    )
    const comparisonGates = page.locator(
      'section[aria-labelledby="evaluation-comparison-gates-title"]',
    )
    await expect(comparisonGates).toBeVisible()
    const valueComparison = comparisonGates.locator('article').filter({
      has: page.getByText('Controlled value comparison', { exact: true }),
    })
    await expect(
      valueComparison.getByText(/Release blocked · complete the required evaluation data/),
    ).toBeVisible()
    await expect(valueComparison.getByText('Passed', { exact: true })).toHaveCount(0)
    await table.scrollIntoViewIfNeeded()
    await captureEvaluationSurface(page, 'comparison-results-desktop')
  })

  test('offers one clear next step when no comparable candidate exists', async ({ page }) => {
    const standaloneBaseline = evaluationRun(
      evaluationRunID(998),
      'Standalone production baseline',
      'completed',
      '2026-08-20T00:00:00Z',
    )
    await mockEvaluationPlane(page, [standaloneBaseline])
    await page.goto('/evaluation?view=compare')

    await expect(page.getByLabel('Comparison candidate', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Compare results' })).toHaveCount(0)
    const panel = page.getByRole('tabpanel')
    await expect(panel.locator('[data-evaluation-action="true"]:visible')).toHaveCount(1)
    await expect(page.getByRole('button', { name: 'Create candidate run' })).toBeVisible()
    const releaseDecisionSummary = page.locator('details > summary').filter({
      has: page.getByText('Prepare a release decision', { exact: true }),
    })
    const releaseDecision = releaseDecisionSummary.locator('..')
    await expect(releaseDecision).not.toHaveAttribute('open', '')
    await expectProductEvaluationLanguage(page)
  })

  const responsiveViewports = [
    { name: 'mobile', width: 390, height: 844 },
    { name: 'tablet-compact', width: 768, height: 1024 },
    { name: 'tablet', width: 1024, height: 768 },
    { name: 'desktop', width: 1440, height: 900 },
  ] as const

  for (const viewport of responsiveViewports) {
    for (const surface of responsiveEvaluationSurfaces) {
      test(`keeps ${surface.capture} coherent at ${viewport.name} width`, async ({ page }) => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height })
        await mockEvaluationPlane(page)
        await expectResponsiveEvaluationSurface(page, surface, viewport.name)
      })
    }
  }
})
