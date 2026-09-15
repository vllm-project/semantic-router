import { expect, test } from '@playwright/test'

import {
  controlledPairSourceRuns,
  launchCampaignControlledPair,
  openReleaseDecisionInputs,
} from './support/campaignActions'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import { defaultEvaluationRuns, evaluationRun } from './support/runFixtures'
import { evaluationRunID } from './support/mixtureFixture'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Controlled-pair lifecycle', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('recovers a server-accepted controlled pair after the create response and page are lost', async ({
    page,
  }) => {
    const { baseline, candidate } = controlledPairSourceRuns()
    const state = await mockEvaluationPlane(page, [candidate, baseline, ...defaultEvaluationRuns], {
      abortControlledPairCreateResponseAfterAccept: true,
      ledgerDelayMs: 300,
    })
    await page.goto('/evaluation?view=compare')
    await launchCampaignControlledPair(page)

    await expect.poll(() => state.controlledPairRequests.length).toBe(1)
    const request = state.controlledPairRequests[0]
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair'))
      .toBe(request.client_request_id)
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBe('recipe')
    await page.reload()

    await openReleaseDecisionInputs(page)
    await expect(
      page.getByText(
        'Fresh baseline and candidate runs completed and are attached to the value comparison.',
      ),
    ).toBeVisible()
    const aggregatePath = `/api/evaluation/v1/controlled-pairs/${request.client_request_id}`
    await expect
      .poll(() => state.controlledPairGetRequests.filter((path) => path === aggregatePath).length)
      .toBeGreaterThanOrEqual(2)
    expect(state.runRequests).not.toContain(request.baseline_run_id)
    expect(state.runRequests).not.toContain(request.candidate_run_id)
    await expect.poll(() => new URL(page.url()).searchParams.get('controlled_pair')).toBeNull()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBeNull()
  })

  test('preserves an active controlled pair while navigating away from Compare and back', async ({
    page,
  }) => {
    const { baseline, candidate } = controlledPairSourceRuns()
    const state = await mockEvaluationPlane(page, [candidate, baseline, ...defaultEvaluationRuns])
    await page.goto('/evaluation?view=compare')
    await launchCampaignControlledPair(page)

    await expect.poll(() => state.controlledPairRequests.length).toBe(1)
    const request = state.controlledPairRequests[0]
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair'))
      .toBe(request.client_request_id)
    await page.getByRole('tab', { name: 'Runs', exact: true }).click()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair'))
      .toBe(request.client_request_id)
    await page.getByRole('tab', { name: 'Compare', exact: true }).click()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair'))
      .toBe(request.client_request_id)

    await openReleaseDecisionInputs(page)
    await expect(
      page.getByText(
        'Fresh baseline and candidate runs completed and are attached to the value comparison.',
      ),
    ).toBeVisible()
    expect(state.runRequests).not.toContain(request.baseline_run_id)
    expect(state.runRequests).not.toContain(request.candidate_run_id)
    await expect.poll(() => new URL(page.url()).searchParams.get('controlled_pair')).toBeNull()
  })

  test('restores a non-default controlled-pair profile across reload and workspace navigation', async ({
    page,
  }) => {
    const { baseline, candidate } = controlledPairSourceRuns('model_pool')
    const state = await mockEvaluationPlane(page, [candidate, baseline, ...defaultEvaluationRuns], {
      controlledPairGetDelayMs: 2_000,
      ledgerDelayMs: 250,
    })
    await page.goto('/evaluation?view=compare')
    await openReleaseDecisionInputs(page)
    await page.getByLabel('Release decision change type').selectOption('model_pool')
    await launchCampaignControlledPair(page)

    await expect.poll(() => state.controlledPairRequests.length).toBe(1)
    const request = state.controlledPairRequests[0]
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair'))
      .toBe(request.client_request_id)
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBe('model_pool')

    await page.reload()
    await openReleaseDecisionInputs(page)
    const profile = page.getByLabel('Release decision change type')
    await expect(profile).toHaveValue('model_pool')
    await expect(profile).toBeDisabled()
    await page.getByRole('tab', { name: 'Runs', exact: true }).click()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBe('model_pool')
    await page.getByRole('tab', { name: 'Compare', exact: true }).click()
    await openReleaseDecisionInputs(page)
    await expect(profile).toHaveValue('model_pool')
    await expect(profile).toBeDisabled()

    await expect(
      page.getByText(
        'Fresh baseline and candidate runs completed and are attached to the value comparison.',
      ),
    ).toBeVisible()
    await expect(page.getByLabel('Controlled comparison runs')).toContainText(
      /Controlled baseline AB\/BA.*Controlled candidate AB\/BA/,
    )
    await expect.poll(() => new URL(page.url()).searchParams.get('controlled_pair')).toBeNull()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBeNull()
  })

  test('restarts authoritative reconciliation after recovered pair polling fails', async ({
    page,
  }) => {
    const pairID = evaluationRunID(940)
    const baselineID = evaluationRunID(941)
    const candidateID = evaluationRunID(942)
    const createdAt = '2026-08-31T03:00:00Z'
    const baseline = evaluationRun(
      baselineID,
      'Recovered pair baseline',
      'running',
      createdAt,
      'model_pool',
      { controlled_pair: { pair_id: pairID, role: 'baseline' } },
    )
    const candidate = evaluationRun(
      candidateID,
      'Recovered pair candidate',
      'running',
      createdAt,
      'model_pool',
      {
        baseline_run_id: baselineID,
        controlled_pair: { pair_id: pairID, role: 'candidate' },
      },
    )
    const state = await mockEvaluationPlane(page, [candidate, baseline, ...defaultEvaluationRuns], {
      controlledPairGetDelayMs: 300,
      failControlledPairGetAt: 2,
    })
    await page.goto(
      `/evaluation?view=compare&controlled_pair=${pairID}&controlled_pair_profile=model_pool`,
    )

    await openReleaseDecisionInputs(page)
    const profile = page.getByLabel('Release decision change type')
    await expect(profile).toHaveValue('model_pool')
    await expect(profile).toBeDisabled()
    await expect(page.getByRole('alert')).toContainText('temporary controlled-pair state failure')
    await expect(profile).toBeDisabled()
    await expect.poll(() => state.controlledPairGetRequests.length).toBeGreaterThanOrEqual(2)

    await page.getByRole('button', { name: 'Retry comparison' }).click()
    await expect.poll(() => state.controlledPairGetRequests.length).toBeGreaterThanOrEqual(3)
    await expect(page.getByRole('alert')).toHaveCount(0)
    await expect(page.getByRole('button', { name: 'Comparison running…' })).toBeDisabled()
    await expect(profile).toBeDisabled()
  })

  test('rejects stale assignment when the profile changes during asynchronous handoff', async ({
    page,
  }) => {
    const { baseline, candidate } = controlledPairSourceRuns()
    await mockEvaluationPlane(page, [candidate, baseline, ...defaultEvaluationRuns], {
      ledgerDelayMs: 1_000,
    })
    await page.goto('/evaluation?view=compare')
    await launchCampaignControlledPair(page)

    const profile = page.getByLabel('Release decision change type')
    await expect(
      page.getByText(
        'Both runs completed. Refreshing run history before attaching the comparison.',
      ),
    ).toBeVisible()
    await expect(profile).toBeDisabled()
    await profile.evaluate((element) => {
      const select = element as HTMLSelectElement
      select.removeAttribute('disabled')
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set
      setter?.call(select, 'model_pool')
      select.dispatchEvent(new Event('change', { bubbles: true }))
    })

    await expect(profile).toHaveValue('recipe')
    await expect(page.getByLabel('Controlled comparison runs')).toContainText(
      /Controlled baseline AB\/BA.*Controlled candidate AB\/BA/,
    )
    await expect.poll(() => new URL(page.url()).searchParams.get('controlled_pair')).toBeNull()
  })

  test('fails closed for invalid or stale controlled-pair route identities', async ({ page }) => {
    const state = await mockEvaluationPlane(page)
    await page.goto(
      '/evaluation?view=compare&controlled_pair=not-a-canonical-id&controlled_pair_profile=recipe',
    )
    await expect(
      page.getByRole('heading', { name: 'Compare a candidate with its baseline' }),
    ).toBeVisible()
    expect(state.controlledPairGetRequests).toHaveLength(0)

    const stalePairID = evaluationRunID(990)
    await page.goto(
      `/evaluation?view=compare&controlled_pair=${stalePairID}&controlled_pair_profile=recipe`,
    )
    await openReleaseDecisionInputs(page)
    await expect(page.getByRole('alert')).toContainText('not found: controlled pair')
    await expect(page.getByLabel('Controlled comparison runs')).not.toContainText(stalePairID)
    await page.getByRole('button', { name: 'Clear saved comparison' }).click()
    await expect.poll(() => new URL(page.url()).searchParams.get('controlled_pair')).toBeNull()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBeNull()
  })
})
