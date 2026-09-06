import { expect, test } from '@playwright/test'

import {
  EVALUATION_MOM,
  EVALUATION_MOM_TARGET_ID,
  EVALUATION_RUN_IDS,
  evaluationRunID,
} from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import {
  captureEvaluationElement,
  captureEvaluationFullPage,
  captureEvaluationSurface,
  expectEvaluationBottomGutter,
  expectKeyboardScrollable,
  expectNoHorizontalOverflow,
  expectPageBottomReachable,
  expectScrollRegionsKeyboardReachable,
} from './support/pageAssertions'
import { defaultEvaluationRuns, evaluationRun } from './support/runFixtures'
import { expectEvaluationControlSystem } from './support/visualAssertions'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Reports', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('offers reports only for completed runs and returns 409 for terminal non-reports', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page)
    await page.goto('/evaluation?view=reports')

    const selector = page.getByLabel('Run')
    await expect(selector).toBeVisible()
    await expect(selector.locator('option')).toHaveCount(4)
    const options = await selector.locator('option').allTextContents()
    expect(options).toEqual([
      'Select a completed run',
      'Candidate recipe · Routing recipe · Replay · Diagnostic · 4 cases',
      'Production baseline · Routing recipe · Replay · Diagnostic · 4 cases',
      'Unpaired diagnostic · Routing recipe · Replay · Diagnostic · 4 cases',
    ])
    expect(options.join(' ')).not.toContain('Live AMD validation')
    expect(options.join(' ')).not.toContain('Failed diagnostic')
    expect(options.join(' ')).not.toContain('Cancelled diagnostic')

    const statuses = await page.evaluate(
      async ([failedRunID, cancelledRunID]) => {
        const [failed, cancelled] = await Promise.all([
          fetch(`/api/evaluation/v1/runs/${failedRunID}/report`),
          fetch(`/api/evaluation/v1/runs/${cancelledRunID}/report`),
        ])
        return [failed.status, cancelled.status]
      },
      [EVALUATION_RUN_IDS.failed, EVALUATION_RUN_IDS.cancelled],
    )
    expect(statuses).toEqual([409, 409])
    expect(state.reportRequests).toEqual(
      expect.arrayContaining([
        EVALUATION_RUN_IDS.candidate,
        EVALUATION_RUN_IDS.failed,
        EVALUATION_RUN_IDS.cancelled,
      ]),
    )
  })

  test('keeps workspace-owned route state isolated', async ({ page }) => {
    await mockEvaluationPlane(page)
    await page.goto(`/evaluation?report=${EVALUATION_RUN_IDS.unpaired}`)

    await expect(page.locator('#latest-evidence-title')).toHaveText('Candidate recipe')
    await page.getByRole('button', { name: 'Open full report' }).click()

    await expect.poll(() => new URL(page.url()).searchParams.get('view')).toBe('reports')
    await expect
      .poll(() => new URL(page.url()).searchParams.get('report'))
      .toBe(EVALUATION_RUN_IDS.candidate)
    await expect(page.getByRole('heading', { name: 'Candidate recipe' })).toBeVisible()
    await expect(
      page
        .locator('section[aria-labelledby="report-diagnostics-title"]')
        .getByText('Total records')
        .locator('..'),
    ).toContainText('32')
  })

  test('keeps the selected report explicit during a service outage', async ({ page }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns, {
      reportFailureIDs: [EVALUATION_RUN_IDS.candidate],
      reportFailureStatus: 503,
    })
    await page.goto('/evaluation')

    const latestReport = page.locator('section[aria-labelledby="latest-evidence-title"]')
    await expect(latestReport.getByText('Latest report could not be refreshed.')).toBeVisible()
    const technicalDetails = latestReport.locator(
      'details[data-evaluation-technical-details="true"]',
    )
    await expect(technicalDetails).not.toHaveAttribute('open', '')
    await expect(page.getByText('report storage is temporarily unavailable')).toBeHidden()
    await technicalDetails.getByText('Technical details', { exact: true }).click()
    await expect(page.getByText('report storage is temporarily unavailable')).toBeVisible()
    expect(state.reportRequests).toEqual([EVALUATION_RUN_IDS.candidate])
  })

  test('keeps diagnostic results distinct from release decisions', async ({ page }) => {
    await mockEvaluationPlane(page)
    await page.goto(`/evaluation?view=reports&report=${EVALUATION_RUN_IDS.candidate}`)

    await expect(
      page.getByText('Diagnostic run — no release recommendation', { exact: true }),
    ).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Diagnostic result only' })).toBeVisible()
    const rawServiceNote = page.getByText(
      'Treat these E0 observations as diagnostics, not a promotion claim.',
      { exact: true },
    )
    await expect(rawServiceNote).not.toBeVisible()
    const findingsSummary = page
      .locator('details > summary')
      .filter({ hasText: 'Next evaluation steps' })
    const findings = findingsSummary.locator('..')
    await findingsSummary.click()
    await expect(
      findings.getByText(
        'Use this diagnostic result to verify the evaluation setup; collect controlled or live results before making a release decision.',
        { exact: true },
      ),
    ).toBeVisible()
    await expect(rawServiceNote).not.toBeVisible()
    const technicalFindingsSummary = findings
      .locator(':scope > div > details > summary')
      .filter({ hasText: 'Technical details' })
    await technicalFindingsSummary.click()
    await expect(rawServiceNote).toBeVisible()

    const metrics = page.getByRole('table', { name: 'Evaluation metrics' })
    await expect(metrics.locator('tr[data-metric-id="joint.realized_quality"]')).toContainText(
      'System quality',
    )
    await expect(metrics.locator('tr[data-metric-id="capacity.latency_p95_ms"]')).toContainText(
      'P95 latency',
    )

    const diagnostics = page.locator('section[aria-labelledby="report-diagnostics-title"]')
    await expect(diagnostics.getByRole('heading', { name: 'Execution diagnostics' })).toBeVisible()
    await expect(diagnostics.getByText('Total records').locator('..')).toContainText('32')
    await expect(
      diagnostics.getByText('Succeeded', { exact: true }).first().locator('..'),
    ).toContainText('32')
    await page.getByRole('heading', { name: 'Diagnostic result only' }).scrollIntoViewIfNeeded()
    await captureEvaluationSurface(page, 'report-decision-desktop')

    const allGatesSummary = page.locator('details > summary').filter({
      has: page.getByText('All release checks', { exact: false }),
    })
    const allGates = allGatesSummary.locator('..')
    await allGatesSummary.click()
    await expect(allGates.getByText('Passed', { exact: true })).toHaveCount(2)
    for (const capability of [
      'Policy enforcement',
      'Controlled value comparison',
      'Workload-shift robustness',
      'Live consistency',
      'Fault recovery',
      'Cost, latency, and capacity',
      'Canary safety',
      'Online preference',
    ]) {
      const gate = allGates.locator('article').filter({
        has: page.getByText(capability, { exact: true }),
      })
      await expect(gate).toHaveCount(1)
      await expect(gate.getByText('Passed', { exact: true })).toHaveCount(0)
    }
    await captureEvaluationSurface(page, 'report-gates-desktop')
  })

  test('renders the live server-owned Routing Recipe report across desktop and compact mobile', async ({
    page,
  }) => {
    const liveReportRun = evaluationRun(
      evaluationRunID(91),
      'Live routing recipe evidence',
      'completed',
      '2026-08-31T01:00:00Z',
      'recipe',
      {
        mode: 'live',
        target_id: EVALUATION_MOM_TARGET_ID,
        mixture: EVALUATION_MOM,
        suite_ids: ['live-mom-core'],
        track_ids: ['routing'],
        evidence_level: 'E3',
        track_evidence_levels: { routing: 'E3' },
        completed_at: '2026-08-31T01:10:00Z',
      },
    )
    await mockEvaluationPlane(page, [liveReportRun, ...defaultEvaluationRuns])
    for (const viewport of [
      { name: 'desktop', width: 1440, height: 900 },
      { name: 'mobile-compact', width: 320, height: 568 },
    ] as const) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height })
      await page.goto(`/evaluation?view=reports&report=${liveReportRun.id}`)

      const routingRecipe = page.locator('section[aria-labelledby="routing-recipe-report-title"]')
      await expect(routingRecipe.getByRole('heading', { name: 'Routing Recipe' })).toBeVisible()
      await expect(routingRecipe.getByText('Decision coverage', { exact: true })).toBeVisible()
      await expect(routingRecipe.getByText('Eligibility complete', { exact: true })).toBeVisible()
      await expect(routingRecipe.getByText('Selected feasible', { exact: true })).toBeVisible()
      await expect(routingRecipe.getByRole('table', { name: 'Signal availability' })).toBeVisible()
      await expect(
        routingRecipe.getByRole('table', { name: 'Projection outcome calibration' }),
      ).toBeVisible()
      await expect(
        routingRecipe.getByText('Quality gap to the best feasible model', { exact: true }),
      ).toBeVisible()
      const technicalDetails = routingRecipe
        .locator('details[data-evaluation-technical-details="true"]')
        .filter({ has: page.getByText('Technical details', { exact: true }) })
      await expect(technicalDetails).not.toHaveAttribute('open', '')
      await expect(
        routingRecipe.getByText('insufficient_latency_samples', { exact: true }).first(),
      ).toBeHidden()
      await technicalDetails.locator(':scope > summary').click()
      await expect(
        routingRecipe.getByText('insufficient_latency_samples', { exact: true }).first(),
      ).toBeVisible()
      await expect(
        routingRecipe.getByText('insufficient_outcome_pairs', { exact: true }),
      ).toBeVisible()
      await expect(routingRecipe.getByText('oracle_outcome_missing', { exact: true })).toHaveCount(
        2,
      )
      await technicalDetails.locator(':scope > summary').click()
      const decision = page.locator('section[aria-labelledby="report-decision-title"]')
      await expect(decision).toBeVisible()
      await expect
        .poll(async () => {
          const decisionBox = await decision.boundingBox()
          const routingBox = await routingRecipe.boundingBox()
          return Boolean(decisionBox && routingBox && decisionBox.y < routingBox.y)
        })
        .toBe(true)
      if (viewport.width === 320) {
        await expectKeyboardScrollable(
          routingRecipe.getByRole('region', { name: 'Signal availability' }),
          'horizontal',
        )
        await expectKeyboardScrollable(
          routingRecipe.getByRole('region', { name: 'Projection outcome calibration' }),
          'horizontal',
        )
      }
      await expectNoHorizontalOverflow(page)
      await expectEvaluationControlSystem(page)
      await expectScrollRegionsKeyboardReachable(page)
      await captureEvaluationElement(routingRecipe, `routing-recipe-deep-dive-${viewport.name}`)
      await expectPageBottomReachable(page)
      await expectEvaluationBottomGutter(page)
      await captureEvaluationSurface(page, `routing-recipe-report-${viewport.name}`)
      await captureEvaluationFullPage(page, `routing-recipe-report-${viewport.name}-full`)
    }
  })

  test('keeps a long execution timeline named, focusable, and keyboard scrollable', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockEvaluationPlane(page, defaultEvaluationRuns, { eventStreamEventCount: 24 })
    await page.goto(`/evaluation?view=runs&run=${EVALUATION_RUN_IDS.live}`)

    const timeline = page.getByRole('region', { name: 'Execution timeline' })
    await expect(timeline).toBeVisible()
    await expect(timeline.locator('li')).toHaveCount(24)
    await expectKeyboardScrollable(timeline, 'vertical')
    await captureEvaluationElement(timeline, 'runs-long-timeline-keyboard-region')
  })

  test('pages dense metric reports and resets the page when filters change', async ({ page }) => {
    await mockEvaluationPlane(page, defaultEvaluationRuns, { reportMetricCount: 45 })
    await page.goto(`/evaluation?view=reports&report=${EVALUATION_RUN_IDS.candidate}`)

    const metrics = page.getByRole('table', { name: 'Evaluation metrics' })
    await expect(metrics.getByRole('row')).toHaveCount(21)
    await expect(page.getByText('1–20 of 45', { exact: true })).toBeVisible()
    await expect(page.getByText('Page 1 of 3', { exact: true })).toBeVisible()

    await page.getByRole('button', { name: 'Next', exact: true }).click()
    await expect(page.getByText('Page 2 of 3', { exact: true })).toBeVisible()
    await expect(page.getByText('21–40 of 45', { exact: true })).toBeVisible()

    await page.getByLabel('Find a metric').fill('metric 45')
    await expect(page.getByText('1–1 of 1 matching · 45 total', { exact: true })).toBeVisible()
    await expect(page.getByText('Page 2 of 3', { exact: true })).toHaveCount(0)
    await expect(metrics.getByRole('row')).toHaveCount(2)
  })

  test('isolates an invalid capacity diagnostic artifact without collapsing the report', async ({
    page,
  }) => {
    await mockEvaluationPlane(page, defaultEvaluationRuns, {
      diagnosticArtifactBodies: {
        capacityProfile:
          '{"schema_version":"evaluation.v1","kind":"bounded-concurrency-sweep","levels":null,"slo":null}',
      },
    })
    await page.goto(`/evaluation?view=reports&report=${EVALUATION_RUN_IDS.candidate}`)

    await expect(
      page.getByText('Diagnostic run — no release recommendation', { exact: true }),
    ).toBeVisible()
    await expect(page.getByRole('table', { name: 'Evaluation metrics' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Results by evaluation area' })).toBeVisible()

    const diagnostics = page.locator('section[aria-labelledby="report-diagnostics-title"]')
    const capacityIssue = diagnostics.getByRole('alert', {
      name: 'Capacity profile diagnostic error',
    })
    await expect(
      capacityIssue.getByText('Diagnostic could not be verified', { exact: true }),
    ).toBeVisible()
    await expect(
      capacityIssue.getByText(
        'This diagnostic is excluded because its saved evidence could not be verified. Other verified results remain available.',
        { exact: true },
      ),
    ).toBeVisible()
    const artifactPath = capacityIssue.getByText('capacity-profile.json', { exact: true })
    const serviceResponse = capacityIssue.getByText(
      'capacity-profile.json did not match the required evaluation.v1 diagnostic schema.',
      { exact: true },
    )
    await expect(artifactPath).not.toBeVisible()
    await expect(serviceResponse).not.toBeVisible()
    await capacityIssue
      .locator('details[data-evaluation-technical-details="true"] > summary')
      .click()
    await expect(artifactPath).toBeVisible()
    await expect(serviceResponse).toBeVisible()
    await expect(
      diagnostics.getByRole('table', { name: 'Outcome accounting by evaluation area' }),
    ).toBeVisible()
    await expect(
      diagnostics.getByRole('table', { name: 'Capacity observations by concurrency' }),
    ).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Report unavailable' })).toHaveCount(0)
  })
})
