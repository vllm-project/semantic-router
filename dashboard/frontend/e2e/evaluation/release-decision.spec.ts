import { expect, test } from '@playwright/test'

import { openReleaseDecisionInputs } from './support/campaignActions'
import { EVALUATION_RUN_IDS } from './support/mixtureFixture'
import { mockEvaluationPlane } from './support/mockEvaluationPlane'
import {
  captureEvaluationElement,
  captureEvaluationFullPage,
  captureEvaluationSurface,
  expectKeyboardScrollable,
  expectNoHorizontalOverflow,
  expectPageBottomReachable,
  expectProductEvaluationLanguage,
} from './support/pageAssertions'
import { releaseDecisionRuns } from './support/releaseDecisionFixtures'
import { defaultEvaluationRuns } from './support/runFixtures'
import {
  expectCompactVerticalFlow,
  expectEvaluationControlSystem,
} from './support/visualAssertions'
import { mockEvaluationUserSession } from './support/session'

test.describe('Evaluation Plane · Release decision', () => {
  test.beforeEach(async ({ page }) => {
    await mockEvaluationUserSession(page)
  })

  test('builds and reloads a verified release decision above diagnostic comparison', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write'])
    const state = await mockEvaluationPlane(
      page,
      [...releaseDecisionRuns(), ...defaultEvaluationRuns],
      { campaignGetDelayMs: 250, failFirstControlledPair: true, ledgerDelayMs: 750 },
    )
    await page.goto('/evaluation?view=compare')

    await expect(
      page.getByRole('heading', { name: 'Compare a candidate with its baseline' }),
    ).toBeVisible()
    const releaseDecisionSummary = page.locator('details > summary').filter({
      has: page.getByText('Prepare a release decision', { exact: true }),
    })
    const releaseDecision = releaseDecisionSummary.locator('..')
    await expect(releaseDecision).not.toHaveAttribute('open', '')
    await expect(
      page.getByLabel('Controlled comparison baseline run', { exact: true }),
    ).not.toBeVisible()
    const evidenceDisclosure = await openReleaseDecisionInputs(page)
    await expect(
      page.getByLabel('Controlled comparison baseline run', { exact: true }),
    ).toBeVisible()
    await expectEvaluationControlSystem(page)
    await evidenceDisclosure.locator(':scope > summary').press('Enter')
    await expect(evidenceDisclosure).not.toHaveAttribute('open', '')
    await evidenceDisclosure.locator(':scope > summary').press('Enter')
    await expect(evidenceDisclosure).toHaveAttribute('open', '')
    await page
      .getByLabel('Controlled comparison baseline run', { exact: true })
      .selectOption(EVALUATION_RUN_IDS.baselineLive)
    await page
      .getByLabel('Controlled comparison candidate run', { exact: true })
      .selectOption(EVALUATION_RUN_IDS.candidateLive)
    await page.getByRole('button', { name: 'Launch comparison' }).click()
    await expect(page.getByRole('alert')).toContainText('two worker slots are required')
    await page.getByRole('button', { name: 'Retry comparison' }).click()
    await expect.poll(() => state.controlledPairRequests.length).toBe(2)
    const controlledPairRequest = state.controlledPairRequests[1]
    expect(Object.keys(controlledPairRequest).sort()).toEqual([
      'baseline_run_id',
      'baseline_source_run_id',
      'candidate_run_id',
      'candidate_source_run_id',
      'client_request_id',
    ])
    expect(controlledPairRequest).toMatchObject({
      baseline_source_run_id: EVALUATION_RUN_IDS.baselineLive,
      candidate_source_run_id: EVALUATION_RUN_IDS.candidateLive,
    })
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair'))
      .toBe(controlledPairRequest.client_request_id)
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBe('recipe')
    const profileSelect = page.getByLabel('Release decision change type')
    await expect(profileSelect).toBeDisabled()
    await expect(
      page.getByText(/change type is locked while this controlled comparison/i),
    ).toBeVisible()
    const aggregatePath = `/api/evaluation/v1/controlled-pairs/${controlledPairRequest.client_request_id}`
    await expect
      .poll(() => state.controlledPairGetRequests.filter((path) => path === aggregatePath).length)
      .toBeGreaterThanOrEqual(1)
    await expect(
      page.getByText(
        'Fresh baseline and candidate runs completed and are attached to the value comparison.',
      ),
    ).toHaveCount(0)
    expect(state.runRequests).not.toContain(controlledPairRequest.baseline_run_id)
    expect(state.runRequests).not.toContain(controlledPairRequest.candidate_run_id)
    await expect
      .poll(() =>
        state
          .getRuns()
          .filter((run) => run.controlled_pair?.pair_id === controlledPairRequest.client_request_id)
          .map((run) => run.status),
      )
      .toEqual(['completed', 'completed'])
    await expect(profileSelect).toBeDisabled()
    await expect(
      page.getByText(
        'Fresh baseline and candidate runs completed and are attached to the value comparison.',
      ),
    ).toBeVisible()
    await expect
      .poll(() => state.controlledPairGetRequests.filter((path) => path === aggregatePath).length)
      .toBeGreaterThanOrEqual(2)
    expect(state.runRequests).not.toContain(controlledPairRequest.baseline_run_id)
    expect(state.runRequests).not.toContain(controlledPairRequest.candidate_run_id)
    await expect.poll(() => new URL(page.url()).searchParams.get('controlled_pair')).toBeNull()
    await expect
      .poll(() => new URL(page.url()).searchParams.get('controlled_pair_profile'))
      .toBeNull()
    const controlledComparison = page.getByLabel('Controlled comparison runs')
    await expect(controlledComparison).toContainText('Controlled baseline AB/BA')
    await expect(controlledComparison).toContainText('Controlled candidate AB/BA')
    const comparisonCandidate = page.getByLabel('Comparison candidate', { exact: true })
    await comparisonCandidate.selectOption(controlledPairRequest.candidate_run_id)
    const comparePanel = page
      .getByRole('heading', { name: 'Compare a candidate with its baseline' })
      .locator('xpath=ancestor::section[1]')
    await expect(comparePanel.getByText(/Controlled baseline AB\/BA/)).toBeVisible()
    await page.getByRole('button', { name: 'Compare results' }).click()
    await expect
      .poll(() => state.comparisonRequests.at(-1))
      .toEqual({
        baselineRunID: controlledPairRequest.baseline_run_id,
        candidateRunID: controlledPairRequest.candidate_run_id,
      })
    await expect(
      page.getByRole('heading', { name: 'Paired scientific statistics', exact: true }),
    ).toBeVisible()
    const controlledPairStatistics = page.getByRole('table', {
      name: 'Paired outcome comparison',
    })
    await expect(controlledPairStatistics).toBeVisible()
    await expect(
      controlledPairStatistics.locator('tr[data-statistic-id="joint.normalized_regret"]'),
    ).toContainText('Not estimable')
    await page.getByLabel('Policy enforcement run').selectOption(EVALUATION_RUN_IDS.campaignG2)
    await page
      .getByLabel('Workload-shift robustness run')
      .selectOption(EVALUATION_RUN_IDS.campaignG4)
    await page.getByLabel('Reference run').selectOption(EVALUATION_RUN_IDS.campaignG5Reference)
    await page
      .getByLabel('Candidate run', { exact: true })
      .selectOption(EVALUATION_RUN_IDS.campaignG5Live)
    const g7Evidence = page.getByLabel('Cost, latency, and capacity run')
    await expect(g7Evidence).toContainText('Option 1')
    await expect(g7Evidence).toContainText('Option 2')
    const g7OptionLabels = await g7Evidence.locator('option').allTextContents()
    expect(new Set(g7OptionLabels).size).toBe(g7OptionLabels.length)
    await g7Evidence.selectOption(EVALUATION_RUN_IDS.campaignG7)
    await page.getByLabel('Decision name').fill('Recipe v4 production review')
    await page
      .locator('details > summary')
      .filter({ has: page.getByText('Decision notes', { exact: true }) })
      .click()
    await page
      .getByLabel('Decision notes')
      .fill('Review the exact treatment after paired target and confirmation evidence.')
    await expectEvaluationControlSystem(page)
    await captureEvaluationSurface(page, 'campaign-builder-desktop')
    await page.getByRole('button', { name: 'Create release decision' }).click()

    await expect.poll(() => state.campaignRequests.length).toBe(1)
    const request = state.campaignRequests[0]
    expect(request).toMatchObject({
      name: 'Recipe v4 production review',
      change_profile: 'recipe',
      gate_bindings: {
        g2_run_id: EVALUATION_RUN_IDS.campaignG2,
        g3_controlled_pair: {
          baseline_run_id: controlledPairRequest.baseline_run_id,
          candidate_run_id: controlledPairRequest.candidate_run_id,
        },
        g4_run_id: EVALUATION_RUN_IDS.campaignG4,
        g5_fidelity: {
          reference_run_id: EVALUATION_RUN_IDS.campaignG5Reference,
          live_run_id: EVALUATION_RUN_IDS.campaignG5Live,
        },
        g7_run_id: EVALUATION_RUN_IDS.campaignG7,
      },
    })
    const rejectedCampaignStatus = await page.evaluate(async (campaignRequest) => {
      const response = await fetch('/api/evaluation/v1/campaigns', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...campaignRequest,
          client_request_id: '00000000-0000-4000-8000-000000000099',
          gate_bindings: { ...campaignRequest.gate_bindings, g2_run_id: undefined },
        }),
      })
      return response.status
    }, request)
    expect(rejectedCampaignStatus).toBe(400)
    await expect
      .poll(() => new URL(page.url()).searchParams.get('campaign'))
      .toBe(request.client_request_id)
    await expect(page.getByRole('heading', { name: 'Recipe v4 production review' })).toBeVisible()
    await expect(page.getByText('Verified release decision', { exact: true })).toBeVisible()
    await expect(page.getByLabel('Release decision summary')).toBeVisible()
    await expectEvaluationControlSystem(page)
    await captureEvaluationSurface(page, 'campaign-decision-desktop')
    const technicalDetails = page
      .locator('details[data-evaluation-technical-details="true"]')
      .filter({
        has: page.getByText('How this decision was verified and can be reproduced', {
          exact: true,
        }),
      })
    await expect(technicalDetails).not.toHaveAttribute('open', '')
    await expect(
      technicalDetails.getByText('1,000 samples · 95% confidence', { exact: true }),
    ).not.toBeVisible()
    const pairedLive = page.locator('section[aria-labelledby="campaign-paired-live-title"]')
    const pairedTableRegion = pairedLive.getByRole('region', {
      name: 'Paired live statistic matrix',
    })
    const pairedTable = pairedLive.getByRole('table', {
      name: 'Paired baseline and candidate statistics',
    })
    await expect(pairedTable.getByRole('row')).toHaveCount(11)
    await expect(
      pairedTable.getByRole('row', { name: /routing Candidate quality protection/i }),
    ).toContainText('+0.01')
    await expect(pairedTable.getByRole('row', { name: /routing Failure risk/i })).toContainText(
      'Passed',
    )
    await expect(
      pairedTable.getByRole('row', { name: /model pool All-model failure risk/i }),
    ).toContainText('Passed')
    const releaseMeasures = pairedLive.getByRole('table', { name: 'Release measures' })
    await expect(releaseMeasures.getByRole('row')).toHaveCount(6)
    await expect(releaseMeasures.getByRole('row', { name: /Pool availability/i })).toContainText(
      '≤ 20.0%',
    )
    const fidelity = page.locator('section[aria-labelledby="campaign-fidelity-title"]')
    await expect(fidelity.getByRole('heading', { name: 'Live consistency' })).toBeVisible()
    await expect(fidelity.getByText('59', { exact: true }).first()).toBeVisible()
    await expect(fidelity.getByText('Passed', { exact: true })).toBeVisible()
    await pairedTableRegion.focus()
    await expect(pairedTableRegion).toBeFocused()
    await expectCompactVerticalFlow(pairedLive)
    await expectNoHorizontalOverflow(page)
    await captureEvaluationElement(pairedLive, 'campaign-paired-live-desktop')
    await captureEvaluationFullPage(page, 'campaign-decision-desktop-full')
    await expectPageBottomReachable(page)
    await captureEvaluationSurface(page, 'campaign-decision-desktop-bottom')
    await page.evaluate(() => {
      const root = document.scrollingElement
      if (root) root.scrollTop = 0
    })
    const gates = page.locator('section[aria-labelledby="campaign-gates-title"]')
    await expect(gates.locator('article')).toHaveCount(10)
    await expect(
      gates
        .getByText('Verified evaluation result · End-to-end validation', { exact: true })
        .first(),
    ).toBeVisible()
    await expectProductEvaluationLanguage(page)
    await technicalDetails.locator(':scope > summary').click()
    await expect(
      technicalDetails.getByText('1,000 samples · 95% confidence', { exact: true }),
    ).toBeVisible()
    await expect(page.getByText('Evaluation receipt', { exact: true })).toBeVisible()
    await expect(page.getByText('Decision receipt', { exact: true })).toBeVisible()
    const anchors = page.locator('section[aria-labelledby="campaign-evidence-title"]')
    await expect(anchors.locator('article')).toHaveCount(7)
    expect(
      await anchors.locator('article').evaluateAll((elements) =>
        elements.map((element) => ({
          slot: element.getAttribute('data-slot-id'),
          role: element.getAttribute('data-binding-role'),
        })),
      ),
    ).toEqual([
      { slot: 'g2', role: 'evidence' },
      { slot: 'g3', role: 'baseline' },
      { slot: 'g3', role: 'candidate' },
      { slot: 'g4', role: 'evidence' },
      { slot: 'g5', role: 'reference' },
      { slot: 'g5', role: 'live' },
      { slot: 'g7', role: 'evidence' },
    ])
    await expect(anchors.getByText('Server execution receipt', { exact: true })).toHaveCount(7)
    const copyExecution = anchors
      .getByRole('button', { name: 'Copy server execution receipt' })
      .first()
    await copyExecution.click()
    await expect(
      anchors.getByRole('button', { name: 'Copied server execution receipt' }).first(),
    ).toBeVisible()
    await expect(
      page.getByRole('heading', { name: 'Compare a candidate with its baseline' }),
    ).toBeVisible()

    await expect
      .poll(
        () =>
          state.campaignGetRequests.filter((campaignID) => campaignID === request.client_request_id)
            .length,
      )
      .toBeGreaterThan(0)
    await page.waitForTimeout(300)
    state.rejectCampaignGets()
    await page.reload()
    await expect(page.getByRole('alert')).toBeVisible()
    const retryDecision = page.getByRole('button', { name: 'Retry decision' })
    state.allowCampaignGets()
    await retryDecision.click()
    await expect(page.getByRole('button', { name: 'Retrying decision…' })).toBeDisabled()
    await expect(page.getByRole('heading', { name: 'Recipe v4 production review' })).toBeVisible()
    expect(state.campaignGetRequests).toContain(request.client_request_id)

    await page.setViewportSize({ width: 1024, height: 768 })
    await page.evaluate(() => {
      const root = document.scrollingElement
      if (root) root.scrollTop = 0
    })
    await expectNoHorizontalOverflow(page)
    await expectCompactVerticalFlow(pairedLive)
    await captureEvaluationSurface(page, 'campaign-decision-tablet')
    await captureEvaluationElement(pairedLive, 'campaign-paired-live-tablet')
    await expectPageBottomReachable(page)
    await captureEvaluationSurface(page, 'campaign-decision-tablet-bottom')

    await page.setViewportSize({ width: 390, height: 844 })
    await page.evaluate(() => {
      const root = document.scrollingElement
      if (root) root.scrollTop = 0
    })
    await expect(page.getByRole('button', { name: 'Start another decision' })).toBeVisible()
    await expectNoHorizontalOverflow(page)
    await expectCompactVerticalFlow(pairedLive)
    await captureEvaluationSurface(page, 'campaign-decision-mobile')
    await expectKeyboardScrollable(pairedTableRegion, 'horizontal')
    await captureEvaluationElement(pairedLive, 'campaign-paired-live-mobile')
    await expectPageBottomReachable(page)
    await captureEvaluationSurface(page, 'campaign-decision-mobile-bottom')
    await page.evaluate(() => {
      const root = document.scrollingElement
      if (root) root.scrollTop = 0
    })

    await page.getByRole('button', { name: 'Start another decision' }).click()
    await expect(page.getByText('Release readiness', { exact: true })).toBeVisible()
    await expect(page.getByLabel('Decision name')).toHaveValue('')
    await expect.poll(() => new URL(page.url()).searchParams.get('campaign')).toBeNull()
    await expectNoHorizontalOverflow(page)
  })
})
