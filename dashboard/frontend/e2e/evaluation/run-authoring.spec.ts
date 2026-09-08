import { expect, test } from '@playwright/test'

import { evaluationCatalog } from './support/catalog'
import { EVALUATION_MOM_TARGET_ID, EVALUATION_RUN_IDS } from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import { captureEvaluationSurface, expectNoHorizontalOverflow } from './support/pageAssertions'
import { defaultEvaluationRuns } from './support/runFixtures'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Run authoring', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('creates and starts a diagnostic run through separately authorized endpoints', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page, defaultEvaluationRuns, { mutationDelayMs: 250 })
    await page.goto('/evaluation?view=new')

    await expect(page.getByText('Evaluation scope · Diagnostic', { exact: true })).toBeVisible()
    await page.getByRole('radio', { name: /Replay/ }).check()
    await page.getByLabel('Evaluation target').selectOption('fixture')
    await page.getByRole('checkbox', { name: /Evaluation setup check/ }).check()
    await page.getByLabel('Change type').selectOption({ label: 'Routing recipe' })
    await page.getByLabel('Experiment name').fill('Recipe v4 candidate')
    await page.getByLabel('Description').fill('Validate the full evaluation surface.')
    await page.getByLabel('Maximum cases').fill('64')
    await page.getByLabel('Parallel requests').fill('8')
    await page.getByLabel('Repeatability key').fill('7')
    await page.getByRole('heading', { name: 'New evaluation experiment' }).scrollIntoViewIfNeeded()
    await captureEvaluationSurface(page, 'new-experiment-desktop')
    await page.getByRole('button', { name: 'Create and start' }).click()

    const form = page.locator('form[aria-busy]')
    await expect(form).toHaveAttribute('aria-busy', 'true')
    await expect(
      page.locator('fieldset[aria-label="Evaluation experiment fields"]'),
    ).toHaveAttribute('disabled', '')
    await expect(page.getByRole('button', { name: 'Creating…' })).toBeDisabled()

    await expect.poll(() => state.createdRequests.length).toBe(1)
    expect(state.createdRequests[0]).toMatchObject({
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
    })
    expect(state.createdRequests[0].client_request_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    )
    await expect.poll(state.getStartCount).toBe(1)
    expect(state.getRuns()[0].evidence_level).toBe('E0')
    await expect(page.getByRole('tab', { name: 'Runs' })).toHaveAttribute('aria-selected', 'true')

    const originalRequest = state.createdRequests[0]
    const originalRunID = state
      .getRuns()
      .find((run) => run.client_request_id === originalRequest.client_request_id)?.id
    const retry = await page.evaluate(async (request) => {
      const send = (body: typeof request) =>
        fetch('/api/evaluation/v1/runs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
      const repeated = await send(request)
      const repeatedRun = (await repeated.json()) as { id: string; client_request_id?: string }
      const conflicting = await send({ ...request, name: `${request.name} changed` })
      return {
        repeatedStatus: repeated.status,
        repeatedRun,
        conflictingStatus: conflicting.status,
      }
    }, originalRequest)
    expect(retry.repeatedStatus).toBe(201)
    expect(retry.repeatedRun).toMatchObject({
      id: originalRunID,
      client_request_id: originalRequest.client_request_id,
    })
    expect(retry.conflictingStatus).toBe(409)
    expect(state.createAttempts).toHaveLength(3)
    expect(
      state.getRuns().filter((run) => run.client_request_id === originalRequest.client_request_id),
    ).toHaveLength(1)
  })

  test('freezes the live Capacity SLO and repeated load protocol in the run', async ({ page }) => {
    const state = await mockEvaluationPlane(page)
    await page.goto('/evaluation?view=new')

    await page
      .getByRole('radio', {
        name: 'Live: evaluate a registered Mixture.',
        exact: true,
      })
      .check()
    await page.getByLabel('Mixture to evaluate').selectOption(EVALUATION_MOM_TARGET_ID)
    await expect(page.getByLabel('Mixture to evaluate')).toHaveValue(EVALUATION_MOM_TARGET_ID)
    await page.getByRole('checkbox', { name: /Routing and model-pool setup check/ }).uncheck()
    await page.getByRole('checkbox', { name: /Capacity setup check/ }).check()

    const capacity = page.locator('section').filter({
      has: page.getByRole('heading', { name: 'Capacity service objective' }),
    })
    const requiredConcurrency = capacity.getByRole('spinbutton', {
      name: /^Required concurrency/,
    })
    await expect(requiredConcurrency).toHaveValue('')
    await capacity.getByRole('button', { name: /Balanced service/ }).click()
    await expect(requiredConcurrency).toHaveValue('4')
    await expect(capacity.getByRole('spinbutton', { name: /^Maximum p95 latency/ })).toHaveValue(
      '750',
    )
    await expect(capacity.getByRole('spinbutton', { name: /^Maximum error rate/ })).toHaveValue(
      '0.02',
    )
    await expect(capacity.getByRole('spinbutton', { name: /^Minimum throughput/ })).toHaveValue(
      '10',
    )
    await expect(
      capacity.getByRole('spinbutton', { name: /^Minimum scaling efficiency/ }),
    ).toHaveValue('0.7')
    await expect(capacity.getByLabel('Recorded capacity load plan')).toContainText(
      '1 → 2 → 4 concurrent requests',
    )
    await expect(capacity.getByLabel('Recorded capacity load plan')).toContainText(
      '100 requests × 3 independent windows (minimum 3)',
    )

    await page.getByLabel('Experiment name').fill('Live capacity operating point')
    await requiredConcurrency.fill('5')
    await page.getByRole('button', { name: 'Create and start' }).click()
    await expect
      .poll(() => requiredConcurrency.evaluate((input: HTMLInputElement) => input.validity.valid))
      .toBe(false)
    expect(state.createdRequests).toHaveLength(0)

    await requiredConcurrency.fill('4')
    await page.setViewportSize({ width: 390, height: 844 })
    await expectNoHorizontalOverflow(page)
    await page.getByRole('button', { name: 'Create and start' }).click()

    await expect.poll(() => state.createdRequests.length).toBe(1)
    expect(state.createdRequests[0]).toMatchObject({
      name: 'Live capacity operating point',
      mode: 'live',
      target_id: EVALUATION_MOM_TARGET_ID,
      suite_ids: ['live-capacity'],
      track_ids: ['capacity'],
      concurrency: 4,
      capacity_slo: {
        schema_version: 'evaluation.v1',
        required_concurrency: 4,
        max_latency_p95_ms: 750,
        max_error_rate: 0.02,
        min_throughput_rps: 10,
        min_throughput_scaling_efficiency: 0.7,
      },
      capacity_load_protocol: {
        schema_version: 'evaluation.v1',
        kind: 'closed-loop',
        concurrency_levels: [1, 2, 4],
        warmup_request_multiplier: 2,
        measurement_requests_per_repetition: 100,
        repetitions_per_level: 3,
        minimum_measurement_clusters_per_level: 3,
        confidence_level: 0.95,
        max_error_rate_cluster_range: 0.05,
        max_throughput_cv: 0.2,
        max_latency_p95_cv: 0.2,
      },
    })
    const rejectedCapacityStatus = await page.evaluate(async (request) => {
      if (!request.capacity_load_protocol) return -1
      const response = await fetch('/api/evaluation/v1/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...request,
          client_request_id: '00000000-0000-4000-8000-000000000098',
          capacity_load_protocol: {
            ...request.capacity_load_protocol,
            concurrency_levels: [1, 4],
          },
        }),
      })
      return response.status
    }, state.createdRequests[0])
    expect(rejectedCapacityStatus).toBe(400)
    await expect.poll(state.getStartCount).toBe(1)
  })

  test('copies and locks the exact cohort when creating a candidate from a baseline', async ({
    page,
  }) => {
    const state = await mockEvaluationPlane(page)
    await page.goto('/evaluation?view=new')

    await page.getByLabel('Baseline run').selectOption(EVALUATION_RUN_IDS.baseline)
    await expect(
      page.getByText(
        'The comparison setup is copied and locked: change type, run type, Mixture, benchmarks, evaluation areas, sample size, parallel requests, performance goals, and repeatability key.',
        { exact: true },
      ),
    ).toBeVisible()
    await expect(page.getByLabel('Change type')).toHaveValue('recipe')
    await expect(page.getByLabel('Change type')).toBeDisabled()
    await expect(page.getByLabel('Evaluation source')).toHaveValue('fixture')
    await expect(page.getByLabel('Evaluation source')).toBeDisabled()
    await expect(page.getByRole('spinbutton', { name: 'Maximum cases', exact: true })).toHaveValue(
      '4',
    )
    await expect(
      page.getByRole('spinbutton', { name: 'Maximum cases', exact: true }),
    ).toBeDisabled()
    await expect(
      page.getByRole('spinbutton', { name: 'Parallel requests', exact: true }),
    ).toHaveValue('4')
    await expect(
      page.getByRole('spinbutton', { name: 'Parallel requests', exact: true }),
    ).toBeDisabled()
    await expect(page.getByRole('spinbutton', { name: /^Repeatability key/ })).toHaveValue('42')
    await expect(page.getByRole('spinbutton', { name: /^Repeatability key/ })).toBeDisabled()

    await page.getByLabel('Experiment name').fill('Paired recipe candidate')
    await page.getByLabel('Description').fill('Exact-cohort candidate for paired comparison.')
    await page.getByRole('checkbox', { name: /Start immediately/ }).uncheck()
    await page.getByRole('button', { name: 'Create draft' }).click()

    await expect.poll(() => state.createdRequests.length).toBe(1)
    expect(state.createdRequests[0]).toMatchObject({
      baseline_run_id: EVALUATION_RUN_IDS.baseline,
      mode: 'replay',
      target_id: 'fixture',
      change_profile: 'recipe',
      suite_ids: ['evaluation-smoke'],
      track_ids: [...evaluationCatalog.suites[0].track_ids],
      sample_limit: 4,
      concurrency: 4,
      seed: 42,
    })
  })
})
