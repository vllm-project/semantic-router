import type { Page } from '@playwright/test'

import type { EvaluationChangeProfileId } from '../../../src/types/evaluationPlane'
import {
  EVALUATION_BASELINE_MOM_TARGET_ID,
  EVALUATION_MOM,
  EVALUATION_MOM_TARGET_ID,
  EVALUATION_RUN_IDS,
} from './mixtureFixture'
import { evaluationRun } from './runFixtures'

export function controlledPairSourceRuns(changeProfile: EvaluationChangeProfileId = 'recipe') {
  const trackIDs = ['routing', 'model_pool', 'joint'] as const
  const suiteIDs = ['live-mom-core']
  const shared = {
    mode: 'live' as const,
    suite_ids: suiteIDs,
    track_ids: [...trackIDs],
    evidence_level: 'E3' as const,
    track_evidence_levels: { routing: 'E3', model_pool: 'E4', joint: 'E5' } as const,
    sample_limit: 64,
    mixture: EVALUATION_MOM,
  }
  return {
    baseline: evaluationRun(
      EVALUATION_RUN_IDS.baselineLive,
      'Recipe live control',
      'completed',
      '2026-08-29T01:00:00Z',
      changeProfile,
      {
        ...shared,
        target_id: EVALUATION_BASELINE_MOM_TARGET_ID,
        completed_at: '2026-08-29T01:10:00Z',
      },
    ),
    candidate: evaluationRun(
      EVALUATION_RUN_IDS.candidateLive,
      'Recipe live treatment',
      'completed',
      '2026-08-29T02:00:00Z',
      changeProfile,
      {
        ...shared,
        target_id: EVALUATION_MOM_TARGET_ID,
        completed_at: '2026-08-29T02:10:00Z',
      },
    ),
  }
}

export async function openReleaseDecisionInputs(page: Page) {
  const releaseDecisionSummary = page.locator('details > summary').filter({
    has: page.getByText('Prepare a release decision', { exact: true }),
  })
  const releaseDecision = releaseDecisionSummary.locator('..')
  if (!(await releaseDecision.evaluate((element) => element.hasAttribute('open')))) {
    await releaseDecisionSummary.click()
  }
  const inputSummary = releaseDecision.locator('details > summary').filter({
    has: page.getByText('Review evaluation inputs', { exact: true }),
  })
  const inputs = inputSummary.locator('..')
  if (!(await inputs.evaluate((element) => element.hasAttribute('open')))) {
    await inputSummary.click()
  }
  return inputs
}

export async function launchCampaignControlledPair(page: Page) {
  await openReleaseDecisionInputs(page)
  await page
    .getByLabel('Controlled comparison baseline run', { exact: true })
    .selectOption(EVALUATION_RUN_IDS.baselineLive)
  await page
    .getByLabel('Controlled comparison candidate run', { exact: true })
    .selectOption(EVALUATION_RUN_IDS.candidateLive)
  await page.getByRole('button', { name: 'Launch comparison' }).click()
}
