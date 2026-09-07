import type { Page } from '@playwright/test'

import type { CreateEvaluationCampaignPayload } from '../../../src/types/evaluationCampaign'
import { evaluationCatalog } from './catalog'
import { evaluationCampaign } from './campaignFixtures'
import {
  controlledPairCohortMatches,
  fulfillError,
  fulfillJSON,
  sameOrderedMembers,
  type EvaluationMockState,
} from './state'

export async function registerCampaignRoutes(
  page: Page,
  state: EvaluationMockState,
): Promise<void> {
  await page.route('**/api/evaluation/v1/campaign-readiness', async (route) => {
    if (route.request().method() !== 'POST') {
      await fulfillError(route, 405, 'method not allowed')
      return
    }
    const request = route.request().postDataJSON() as {
      change_profile?: string
      limit?: number
      cursor?: string
      controlled_pair_baseline_run_id?: string
      fidelity_reference_run_id?: string
    }
    const profile = evaluationCatalog.change_profiles.find(
      (candidate) => candidate.id === request.change_profile,
    )
    const limit = request.limit || 50
    const offset = request.cursor ? Number(request.cursor) : 0
    if (
      !profile ||
      !Number.isInteger(limit) ||
      limit < 1 ||
      limit > 200 ||
      !Number.isInteger(offset) ||
      offset < 0 ||
      offset > state.runs.length
    ) {
      await fulfillError(route, 400, 'invalid evaluation request: campaign readiness')
      return
    }
    const pageRuns = state.runs.slice(offset, offset + limit)
    const eligibleRuns = pageRuns.filter(
      (run) =>
        run.status === 'completed' &&
        run.change_profile === request.change_profile,
    )
    const allEligibleRuns = state.runs.filter(
      (run) => run.status === 'completed' && run.change_profile === request.change_profile,
    )
    const controlledBaseline = allEligibleRuns.find(
      (run) => run.id === request.controlled_pair_baseline_run_id,
    )
    const fidelityReference = allEligibleRuns.find(
      (run) => run.id === request.fidelity_reference_run_id,
    )
    const matchesFidelityReference = (live: (typeof state.runs)[number]) =>
      Boolean(
        fidelityReference &&
          fidelityReference.id !== live.id &&
          fidelityReference.mode === 'live' &&
          live.mode === 'live' &&
          fidelityReference.target_id === live.target_id &&
          fidelityReference.completed_at &&
          live.started_at &&
          Date.parse(live.started_at) > Date.parse(fidelityReference.completed_at) &&
          fidelityReference.sample_limit === live.sample_limit &&
          fidelityReference.seed === live.seed &&
          sameOrderedMembers(fidelityReference.suite_ids, live.suite_ids) &&
          sameOrderedMembers(fidelityReference.track_ids, live.track_ids),
      )
    const nextOffset = offset + pageRuns.length
    await fulfillJSON(route, 200, {
      schema_version: 'evaluation.v1',
      change_profile: profile.id,
      ...(nextOffset < state.runs.length ? { next_cursor: String(nextOffset) } : {}),
      total_runs: state.runs.length,
      slots: profile.campaign_slots.map((slot) => ({
        gate_id: slot.gate_id,
        binding_kind: slot.binding_kind,
        eligible_run_ids:
          slot.binding_kind === 'run' ? eligibleRuns.map((run) => run.id) : [],
        controlled_pair_source_run_ids:
          slot.binding_kind === 'controlled_pair'
            ? eligibleRuns.filter((run) => run.mode === 'live').map((run) => run.id)
            : [],
        controlled_pair_candidate_run_ids:
          slot.binding_kind === 'controlled_pair' && controlledBaseline
            ? eligibleRuns
                .filter(
                  (candidate) =>
                    candidate.id !== controlledBaseline.id &&
                    controlledPairCohortMatches(controlledBaseline, candidate),
                )
                .map((run) => run.id)
            : [],
        fidelity_reference_run_ids:
          slot.binding_kind === 'fidelity_pair'
            ? eligibleRuns.filter((run) => run.mode === 'live').map((run) => run.id)
            : [],
        fidelity_live_run_ids:
          slot.binding_kind === 'fidelity_pair'
            ? eligibleRuns.filter(matchesFidelityReference).map((run) => run.id)
            : [],
      })),
    })
  })
  await page.route(
    /\/api\/evaluation\/v1\/campaigns(?:\/[^/?]+(?:\/decision)?)?(?:\?.*)?$/,
    async (route) => {
      const url = new URL(route.request().url())
      const parts = url.pathname.split('/').filter(Boolean)
      const campaignIndex = parts.indexOf('campaigns')
      const id = campaignIndex >= 0 ? decodeURIComponent(parts[campaignIndex + 1] || '') : ''
      if (route.request().method() === 'GET') {
        state.campaignGetRequests.push(id)
        const shouldFail = state.rejectCampaignGets
        await new Promise<void>((resolve) =>
          setTimeout(resolve, state.options.campaignGetDelayMs || 0),
        )
        if (shouldFail) {
          await fulfillError(route, 503, 'temporary campaign read failure')
          return
        }
        const campaign = state.campaigns.get(id)
        if (!campaign) {
          await fulfillError(route, 404, 'not found: evaluation campaign')
          return
        }
        await fulfillJSON(
          route,
          200,
          parts[campaignIndex + 2] === 'decision' ? campaign.decision : campaign,
        )
        return
      }
      if (route.request().method() !== 'POST' || id) {
        await fulfillError(route, 405, 'method not allowed')
        return
      }
      const raw = route.request().postDataJSON() as Record<string, unknown>
      const request = raw as unknown as CreateEvaluationCampaignPayload
      state.campaignRequests.push(request)
      const allowed = new Set([
        'client_request_id',
        'name',
        'description',
        'change_profile',
        'gate_bindings',
      ])
      const profile = evaluationCatalog.change_profiles.find(
        (candidate) => candidate.id === request.change_profile,
      )
      const bindings = request.gate_bindings
      const bindingIDs = {
        G2: bindings?.g2_run_id ? [bindings.g2_run_id] : [],
        G3: bindings?.g3_controlled_pair
          ? [
              bindings.g3_controlled_pair.baseline_run_id,
              bindings.g3_controlled_pair.candidate_run_id,
            ]
          : [],
        G4: bindings?.g4_run_id ? [bindings.g4_run_id] : [],
        G5: bindings?.g5_fidelity
          ? [bindings.g5_fidelity.reference_run_id, bindings.g5_fidelity.live_run_id]
          : [],
        G6: bindings?.g6_run_id ? [bindings.g6_run_id] : [],
        G7: bindings?.g7_run_id ? [bindings.g7_run_id] : [],
        G8: bindings?.g8_run_id ? [bindings.g8_run_id] : [],
        G9: bindings?.g9_run_id ? [bindings.g9_run_id] : [],
      }
      const selectedIDs = Object.values(bindingIDs).flat()
      const selectedRuns = selectedIDs.map((runID) => state.runs.find((run) => run.id === runID))
      const baselineLive = bindings?.g3_controlled_pair
        ? state.runs.find((run) => run.id === bindings.g3_controlled_pair?.baseline_run_id)
        : undefined
      const candidateLive = bindings?.g3_controlled_pair
        ? state.runs.find((run) => run.id === bindings.g3_controlled_pair?.candidate_run_id)
        : undefined
      const fidelityReference = bindings?.g5_fidelity
        ? state.runs.find((run) => run.id === bindings.g5_fidelity?.reference_run_id)
        : undefined
      const fidelityLive = bindings?.g5_fidelity
        ? state.runs.find((run) => run.id === bindings.g5_fidelity?.live_run_id)
        : undefined
      const invalid =
        Object.keys(raw).some((key) => !allowed.has(key)) ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
          request.client_request_id || '',
        ) ||
        !request.name ||
        request.name.trim() !== request.name ||
        request.description.trim() !== request.description ||
        !profile ||
        !bindings ||
        profile?.campaign_slots.some(
          (slot) => slot.disposition === 'required' && bindingIDs[slot.gate_id].length === 0,
        ) ||
        selectedRuns.some(
          (run) =>
            !run || run.status !== 'completed' || run.change_profile !== request.change_profile,
        ) ||
        new Set(selectedIDs).size !== selectedIDs.length ||
        (bindings?.g3_controlled_pair !== undefined &&
          (!baselineLive ||
            !candidateLive ||
            baselineLive.id === candidateLive.id ||
            baselineLive.mode !== 'live' ||
            candidateLive.mode !== 'live' ||
            candidateLive.baseline_run_id !== baselineLive.id ||
            !controlledPairCohortMatches(baselineLive, candidateLive))) ||
        (bindings?.g5_fidelity !== undefined &&
          (!fidelityReference ||
            !fidelityLive ||
            fidelityReference.mode !== 'live' ||
            fidelityLive.mode !== 'live' ||
            !fidelityReference.completed_at ||
            !fidelityLive.started_at ||
            Date.parse(fidelityLive.started_at) <= Date.parse(fidelityReference.completed_at) ||
            fidelityReference.sample_limit !== fidelityLive.sample_limit ||
            fidelityReference.seed !== fidelityLive.seed ||
            !sameOrderedMembers(fidelityReference.suite_ids, fidelityLive.suite_ids) ||
            !sameOrderedMembers(fidelityReference.track_ids, fidelityLive.track_ids)))
      if (invalid) {
        await fulfillError(route, 400, 'invalid evaluation request: campaign contract rejected')
        return
      }
      if (state.ledgerWarningCount > 0) {
        await fulfillError(
          route,
          409,
          'conflict: evaluation run ledger is incomplete; repair quarantined evidence before deciding',
        )
        return
      }
      const existing = state.campaigns.get(request.client_request_id)
      if (existing) {
        await fulfillJSON(route, 201, existing)
        return
      }
      const campaign = evaluationCampaign(request)
      state.campaigns.set(campaign.id, campaign)
      await state.mutationDelay()
      await fulfillJSON(route, 201, campaign)
    },
  )
}
