import type { Page } from '@playwright/test'

import type { EvaluationRun } from '../../../src/types/evaluationPlane'
import type { CreateEvaluationControlledPairPayload } from '../../../src/types/evaluationControlledPair'
import { EVALUATION_RUN_IDS } from './mixtureFixture'
import {
  controlledPairCohortMatches,
  fulfillError,
  fulfillJSON,
  type EvaluationMockState,
} from './state'

export async function registerControlledPairRoutes(
  page: Page,
  state: EvaluationMockState,
): Promise<void> {
  await page.route('**/api/evaluation/v1/controlled-pairs', async (route) => {
    if (route.request().method() !== 'POST') {
      await fulfillError(route, 405, 'method not allowed')
      return
    }
    const raw = route.request().postDataJSON() as Record<string, unknown>
    const request = raw as unknown as CreateEvaluationControlledPairPayload
    state.controlledPairRequests.push(request)
    const allowed = [
      'client_request_id',
      'baseline_source_run_id',
      'candidate_source_run_id',
      'baseline_run_id',
      'candidate_run_id',
    ]
    const ids = allowed.map((field) => raw[field])
    const canonical = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
    const baselineSource = state.runs.find((run) => run.id === request.baseline_source_run_id)
    const candidateSource = state.runs.find((run) => run.id === request.candidate_source_run_id)
    if (
      Object.keys(raw).length !== allowed.length ||
      Object.keys(raw).some((field) => !allowed.includes(field)) ||
      ids.some((id) => typeof id !== 'string' || !canonical.test(id)) ||
      new Set(ids).size !== ids.length ||
      !baselineSource ||
      !candidateSource ||
      baselineSource.status !== 'completed' ||
      candidateSource.status !== 'completed' ||
      baselineSource.mode !== 'live' ||
      candidateSource.mode !== 'live' ||
      !controlledPairCohortMatches(baselineSource, candidateSource)
    ) {
      await fulfillError(
        route,
        400,
        'invalid evaluation request: controlled pair contract rejected',
      )
      return
    }
    if (state.firstControlledPairPending) {
      state.firstControlledPairPending = false
      await fulfillError(
        route,
        409,
        'controlled pairing is unavailable because two worker slots are required',
      )
      return
    }
    const controlledProgress = {
      percent: 35,
      completed: Math.max(1, Math.floor(baselineSource.track_ids.length / 2)),
      total: baselineSource.track_ids.length,
      message: 'AB/BA block admitted by server coordinator',
    }
    const baselineRun: EvaluationRun = {
      ...baselineSource,
      id: request.baseline_run_id,
      client_request_id: request.baseline_run_id,
      name: 'Controlled baseline AB/BA',
      description: 'Server-owned abba-interleaved.v1 execution',
      status: 'running',
      baseline_run_id: undefined,
      controlled_pair: { pair_id: request.client_request_id, role: 'baseline' },
      progress: controlledProgress,
      created_at: '2026-08-31T01:00:00Z',
      started_at: '2026-08-31T01:00:01Z',
      completed_at: undefined,
    }
    const candidateRun: EvaluationRun = {
      ...candidateSource,
      id: request.candidate_run_id,
      client_request_id: request.candidate_run_id,
      name: 'Controlled candidate AB/BA',
      description: 'Server-owned abba-interleaved.v1 execution',
      status: 'running',
      baseline_run_id: baselineRun.id,
      controlled_pair: { pair_id: request.client_request_id, role: 'candidate' },
      progress: controlledProgress,
      created_at: '2026-08-31T01:00:00Z',
      started_at: '2026-08-31T01:00:01Z',
      completed_at: undefined,
    }
    state.controlledPairRunIDs.add(baselineRun.id)
    state.controlledPairRunIDs.add(candidateRun.id)
    state.controlledPairSources.set(request.client_request_id, {
      baselineSourceRunID: request.baseline_source_run_id,
      candidateSourceRunID: request.candidate_source_run_id,
    })
    state.controlledPairStates.set(request.client_request_id, 'running')
    state.controlledPairAggregatePolls.set(request.client_request_id, 0)
    state.runs = [candidateRun, baselineRun, ...state.runs]
    if (state.abortControlledPairCreateResponsePending) {
      state.abortControlledPairCreateResponsePending = false
      await route.abort('failed')
      return
    }
    await fulfillJSON(route, 201, {
      schema_version: 'evaluation.v1',
      contract_version: 'evaluation-controlled-pair.v1',
      id: request.client_request_id,
      protocol: 'abba-interleaved.v1',
      baseline_source_run_id: request.baseline_source_run_id,
      candidate_source_run_id: request.candidate_source_run_id,
      baseline_run: baselineRun,
      candidate_run: candidateRun,
      state: 'running',
      capabilities: { can_cancel: true, can_delete: false },
    })
  })
  await page.route(
    /\/api\/evaluation\/v1\/controlled-pairs\/[^/?]+(?:\/cancel)?(?:\?.*)?$/,
    async (route) => {
      const url = new URL(route.request().url())
      const parts = url.pathname.split('/').filter(Boolean)
      const pairIndex = parts.indexOf('controlled-pairs')
      const pairID = decodeURIComponent(parts[pairIndex + 1] || '')
      const action = parts[pairIndex + 2] || ''
      const members = state.runs.filter((run) => run.controlled_pair?.pair_id === pairID)
      const baseline = members.find((run) => run.controlled_pair?.role === 'baseline')
      const candidate = members.find((run) => run.controlled_pair?.role === 'candidate')
      if (!baseline || !candidate) {
        await fulfillError(route, 404, 'not found: controlled pair')
        return
      }
      const sources = state.controlledPairSources.get(pairID) || {
        baselineSourceRunID: EVALUATION_RUN_IDS.baseline,
        candidateSourceRunID: EVALUATION_RUN_IDS.candidate,
      }
      if (route.request().method() === 'GET' && action === '') {
        state.controlledPairGetRequests.push(url.pathname)
        const pairGetAttempt = state.controlledPairGetRequests.filter(
          (path) => path === url.pathname,
        ).length
        await new Promise<void>((resolve) =>
          setTimeout(resolve, state.options.controlledPairGetDelayMs || 0),
        )
        if (
          state.firstControlledPairGetPending ||
          pairGetAttempt === state.options.failControlledPairGetAt
        ) {
          state.firstControlledPairGetPending = false
          await fulfillError(route, 503, 'temporary controlled-pair state failure')
          return
        }
        // Concurrent aggregate requests may finish after another request advances the
        // pair. Re-read the ledger so state and member snapshots remain coherent.
        const latestBaseline = state.runs.find((run) => run.id === baseline.id) || baseline
        const latestCandidate = state.runs.find((run) => run.id === candidate.id) || candidate
        let responseBaseline = latestBaseline
        let responseCandidate = latestCandidate
        let pairState = state.controlledPairStates.get(pairID)
        if (
          state.controlledPairRunIDs.has(latestBaseline.id) &&
          state.controlledPairRunIDs.has(latestCandidate.id)
        ) {
          const pollCount = (state.controlledPairAggregatePolls.get(pairID) || 0) + 1
          state.controlledPairAggregatePolls.set(pairID, pollCount)
          const completedAt = '2026-08-31T01:05:00Z'
          responseBaseline = {
            ...latestBaseline,
            status: 'completed',
            progress: {
              percent: 100,
              completed: latestBaseline.track_ids.length,
              total: latestBaseline.track_ids.length,
              message: 'Controlled AB/BA evidence complete',
            },
            completed_at: completedAt,
          }
          responseCandidate = {
            ...latestCandidate,
            status: 'completed',
            progress: {
              percent: 100,
              completed: latestCandidate.track_ids.length,
              total: latestCandidate.track_ids.length,
              message: 'Controlled AB/BA evidence complete',
            },
            completed_at: completedAt,
          }
          state.runs = state.runs.map((run) =>
            run.id === responseBaseline.id
              ? responseBaseline
              : run.id === responseCandidate.id
                ? responseCandidate
                : run,
          )
          if (pollCount >= 2) {
            pairState = 'terminal'
            state.controlledPairStates.set(pairID, pairState)
            state.controlledPairRunIDs.delete(baseline.id)
            state.controlledPairRunIDs.delete(candidate.id)
          }
        }
        if (!pairState) {
          const statuses = [responseBaseline.status, responseCandidate.status]
          pairState = statuses.every((status) => status === 'pending')
            ? 'pending'
            : statuses.every((status) => ['completed', 'failed', 'cancelled'].includes(status))
              ? 'terminal'
              : 'running'
        }
        const response = {
          schema_version: 'evaluation.v1',
          contract_version: 'evaluation-controlled-pair.v1',
          id: pairID,
          protocol: 'abba-interleaved.v1',
          baseline_source_run_id: sources.baselineSourceRunID,
          candidate_source_run_id: sources.candidateSourceRunID,
          baseline_run: responseBaseline,
          candidate_run: responseCandidate,
          state: pairState,
          capabilities:
            pairState === 'running'
              ? { can_cancel: true, can_delete: false }
              : { can_cancel: false, can_delete: true },
        }
        await fulfillJSON(route, 200, response)
        return
      }
      if (route.request().method() === 'POST' && action === 'cancel') {
        state.controlledPairCancelRequests.push(url.pathname)
        if (state.firstControlledPairCancelPending) {
          state.firstControlledPairCancelPending = false
          await fulfillError(route, 503, 'temporary controlled-pair cancellation failure')
          return
        }
        const aggregateState =
          state.controlledPairStates.get(pairID) ||
          ([baseline.status, candidate.status].every((status) => status === 'pending')
            ? 'pending'
            : [baseline.status, candidate.status].every((status) =>
                  ['completed', 'failed', 'cancelled'].includes(status),
                )
              ? 'terminal'
              : 'running')
        if (aggregateState !== 'running') {
          await fulfillError(route, 409, 'conflict: controlled pair is not running')
          return
        }
        await state.mutationDelay()
        const completedAt = '2026-08-31T01:04:00Z'
        state.runs = state.runs.map((run) =>
          run.controlled_pair?.pair_id === pairID
            ? {
                ...run,
                status: 'cancelled' as const,
                completed_at: completedAt,
                progress: { ...run.progress, message: 'Controlled pair cancelled' },
              }
            : run,
        )
        state.controlledPairRunIDs.delete(baseline.id)
        state.controlledPairRunIDs.delete(candidate.id)
        state.controlledPairStates.set(pairID, 'terminal')
        const cancelledBaseline = state.runs.find((run) => run.id === baseline.id)!
        const cancelledCandidate = state.runs.find((run) => run.id === candidate.id)!
        await fulfillJSON(route, 200, {
          schema_version: 'evaluation.v1',
          contract_version: 'evaluation-controlled-pair.v1',
          id: pairID,
          protocol: 'abba-interleaved.v1',
          baseline_source_run_id: sources.baselineSourceRunID,
          candidate_source_run_id: sources.candidateSourceRunID,
          baseline_run: cancelledBaseline,
          candidate_run: cancelledCandidate,
          state: 'terminal',
          capabilities: { can_cancel: false, can_delete: true },
        })
        return
      }
      if (route.request().method() === 'DELETE' && action === '') {
        state.controlledPairDeleteRequests.push(url.pathname)
        const aggregateState = state.controlledPairStates.get(pairID)
        if (
          aggregateState === 'running' ||
          (!aggregateState &&
            [baseline.status, candidate.status].some((status) =>
              ['running', 'sealing'].includes(status),
            ))
        ) {
          await fulfillError(route, 409, 'conflict: controlled pair is still running')
          return
        }
        await state.mutationDelay()
        state.runs = state.runs.filter((run) => run.controlled_pair?.pair_id !== pairID)
        state.controlledPairSources.delete(pairID)
        state.controlledPairStates.delete(pairID)
        state.controlledPairAggregatePolls.delete(pairID)
        await route.fulfill({ status: 204 })
        return
      }
      await fulfillError(route, 405, 'method not allowed')
    },
  )
}
