import type { Page } from '@playwright/test'

import type { EvaluationRun } from '../../../src/types/evaluationPlane'
import { defaultEvaluationRuns } from './runFixtures'
import { registerCampaignRoutes } from './campaignRoutes'
import { evaluationCatalog } from './catalog'
import { registerControlledPairRoutes } from './controlledPairRoutes'
import { registerEvidenceRoutes } from './evidenceRoutes'
import { registerRunRoutes } from './runRoutes'
import { createEvaluationMockState, fulfillJSON, type MockEvaluationPlaneOptions } from './state'

export async function mockEvaluationPlane(
  page: Page,
  initialRuns = defaultEvaluationRuns,
  options: MockEvaluationPlaneOptions = {},
) {
  const state = createEvaluationMockState(initialRuns, options)

  // Route registration order is part of the browser contract: keep specific
  // endpoints ahead of the generic run collection and run detail patterns.
  await page.route('**/api/evaluation/v1/catalog', async (route) => {
    await new Promise<void>((resolve) => setTimeout(resolve, state.options.catalogDelayMs || 0))
    await fulfillJSON(route, 200, state.options.catalog ?? evaluationCatalog)
  })
  await registerControlledPairRoutes(page, state)
  await registerCampaignRoutes(page, state)
  await registerEvidenceRoutes(page, state)
  await registerRunRoutes(page, state)

  return {
    createAttempts: state.createAttempts,
    createdRequests: state.createdRequests,
    comparisonRequests: state.comparisonRequests,
    campaignRequests: state.campaignRequests,
    controlledPairRequests: state.controlledPairRequests,
    controlledPairGetRequests: state.controlledPairGetRequests,
    controlledPairCancelRequests: state.controlledPairCancelRequests,
    controlledPairDeleteRequests: state.controlledPairDeleteRequests,
    campaignGetRequests: state.campaignGetRequests,
    runRequests: state.runRequests,
    reportRequests: state.reportRequests,
    getCancelCount: () => state.cancelCount,
    getDeleteCount: () => state.deleteCount,
    getStartCount: () => state.startCount,
    getEventStreamCount: () => state.eventStreamCount,
    getLedgerRequestCount: () => state.ledgerRequestCount,
    getRuns: () => [...state.runs],
    addRun: (run: EvaluationRun) => {
      state.runs = [run, ...state.runs.filter((candidate) => candidate.id !== run.id)]
    },
    rejectCampaignGets: () => {
      state.rejectCampaignGets = true
    },
    allowCampaignGets: () => {
      state.rejectCampaignGets = false
    },
  }
}
