import type { Page } from '@playwright/test'

import type { CreateEvaluationRunPayload } from '../../../src/types/evaluationPlane'
import { evaluationCatalog } from './catalog'
import { evaluationRun } from './runFixtures'
import {
  createRequestMatchesRun,
  exactCohortMatches,
  fulfillError,
  fulfillJSON,
  validCapacityLoadProtocol,
  validCapacitySLO,
  type EvaluationMockState,
} from './state'

export async function registerRunRoutes(page: Page, state: EvaluationMockState): Promise<void> {
  await page.route('**/api/evaluation/v1/runs/*/cancel', async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 2] || '')
    const current = state.runs.find((run) => run.id === id)
    if (!current) {
      await fulfillError(route, 404, 'not found: evaluation run')
      return
    }
    if (current.controlled_pair) {
      await fulfillError(route, 409, 'conflict: controlled pair requires aggregate cancellation')
      return
    }
    if (current.status !== 'running') {
      await fulfillError(route, 409, `conflict: run cannot be cancelled from ${current.status}`)
      return
    }
    if (state.firstCancelPending) {
      state.firstCancelPending = false
      await fulfillError(route, 503, 'temporary cancellation failure')
      return
    }
    await state.mutationDelay()
    const cancelled = {
      ...current,
      status: 'cancelled' as const,
      completed_at: '2026-08-29T00:11:00Z',
      progress: { ...current.progress, message: 'Run cancelled' },
    }
    state.runs = state.runs.map((run) => (run.id === id ? cancelled : run))
    state.cancelCount += 1
    await fulfillJSON(route, 200, cancelled)
  })
  await page.route('**/api/evaluation/v1/runs/*/start', async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 2] || '')
    const current = state.runs.find((run) => run.id === id)
    if (!current) {
      await fulfillError(route, 404, 'not found: evaluation run')
      return
    }
    if (current.controlled_pair) {
      await fulfillError(route, 409, 'conflict: controlled pair members cannot be started directly')
      return
    }
    if (current.status !== 'pending') {
      await fulfillError(route, 409, `conflict: run cannot be started from ${current.status}`)
      return
    }
    await state.mutationDelay()
    const started = {
      ...current,
      status: 'running' as const,
      started_at: '2026-08-29T01:01:00Z',
      progress: { ...current.progress, message: 'Evaluation worker starting' },
    }
    state.runs = state.runs.map((run) => (run.id === id ? started : run))
    state.startCount += 1
    await fulfillJSON(route, 200, started)
  })
  await page.route(/\/api\/evaluation\/v1\/runs\/[^/?]+(?:\?.*)?$/, async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 1] || '')
    const current = state.runs.find((run) => run.id === id)
    if (!current) {
      await fulfillError(route, 404, 'not found: evaluation run')
      return
    }
    if (route.request().method() === 'GET') {
      state.runRequests.push(id)
      await new Promise<void>((resolve) => setTimeout(resolve, state.options.runDelayMs || 0))
      await fulfillJSON(route, 200, current)
      return
    }
    if (route.request().method() !== 'DELETE') {
      await fulfillError(route, 405, 'method not allowed')
      return
    }
    if (current.controlled_pair) {
      await fulfillError(route, 409, 'conflict: controlled pair requires aggregate deletion')
      return
    }
    if (current.status === 'running' || current.status === 'sealing') {
      await fulfillError(route, 409, 'conflict: evaluation execution is still active')
      return
    }
    await state.mutationDelay()
    state.runs = state.runs.filter((run) => run.id !== id)
    state.deleteCount += 1
    await route.fulfill({ status: 204 })
  })
  await page.route(/\/api\/evaluation\/v1\/runs(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'POST') {
      const rawRequest = route.request().postDataJSON() as Record<string, unknown>
      const request = rawRequest as unknown as CreateEvaluationRunPayload
      const allowedCreateFields = new Set([
        'client_request_id',
        'name',
        'description',
        'suite_ids',
        'track_ids',
        'mode',
        'target_id',
        'change_profile',
        'sample_limit',
        'concurrency',
        'capacity_slo',
        'capacity_load_protocol',
        'seed',
        'baseline_run_id',
      ])
      state.createAttempts.push(request)
      const target = evaluationCatalog.targets.find(
        (candidate) => candidate.id === request.target_id,
      )
      const suites = request.suite_ids.map((id) =>
        evaluationCatalog.suites.find((candidate) => candidate.id === id),
      )
      const suiteTrackIDs = new Set(suites.flatMap((suite) => suite?.track_ids || []))
      const capacitySLORequired = request.mode === 'live' && request.track_ids.includes('capacity')
      const capacitySLO = request.capacity_slo
      const capacityLoadProtocol = request.capacity_load_protocol
      const capacitySLOValid = validCapacitySLO(capacitySLO, request.concurrency)
      const utf8Length = (value: string) => new TextEncoder().encode(value).length
      const invalid =
        !request.name.trim() ||
        utf8Length(request.name.trim()) > 200 ||
        utf8Length(request.description.trim()) > 4000 ||
        !Number.isInteger(request.sample_limit) ||
        request.sample_limit < 1 ||
        request.sample_limit > 100000 ||
        !Number.isInteger(request.concurrency) ||
        request.concurrency < 1 ||
        request.concurrency > 128 ||
        !Number.isInteger(request.seed) ||
        request.seed < 0 ||
        request.seed > 4294967295 ||
        (capacitySLORequired
          ? !capacitySLOValid ||
            !validCapacityLoadProtocol(capacityLoadProtocol, request.concurrency)
          : capacitySLO !== undefined || capacityLoadProtocol !== undefined) ||
        !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
          request.client_request_id,
        ) ||
        Object.keys(rawRequest).some((field) => !allowedCreateFields.has(field)) ||
        (request.baseline_run_id !== undefined &&
          !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
            request.baseline_run_id,
          )) ||
        !evaluationCatalog.change_profiles.some(
          (profile) => profile.id === request.change_profile,
        ) ||
        !target ||
        target.healthy === false ||
        !target.modes.includes(request.mode) ||
        request.suite_ids.length === 0 ||
        request.track_ids.length === 0 ||
        suites.some(
          (suite) =>
            !suite ||
            !suite.modes.includes(request.mode) ||
            suite.track_ids.some((trackID) => !target.track_ids.includes(trackID)),
        ) ||
        request.track_ids.some(
          (trackID) =>
            !suiteTrackIDs.has(trackID) ||
            !target.track_ids.includes(trackID) ||
            !evaluationCatalog.tracks
              .find((track) => track.id === trackID)
              ?.modes.includes(request.mode),
        )
      if (invalid) {
        await fulfillError(route, 400, 'invalid evaluation request: create contract rejected')
        return
      }
      if (request.baseline_run_id && state.ledgerWarningCount > 0) {
        await fulfillError(
          route,
          409,
          'conflict: evaluation run ledger is incomplete; repair quarantined evidence before selecting a baseline',
        )
        return
      }
      const baseline = request.baseline_run_id
        ? state.runs.find((run) => run.id === request.baseline_run_id)
        : null
      if (request.baseline_run_id && (!baseline || baseline.status !== 'completed')) {
        await fulfillError(route, 400, 'invalid evaluation request: baseline run must be completed')
        return
      }
      const requestRun = evaluationRun(
        'request-cohort',
        request.name,
        'pending',
        '2026-08-29T01:00:00Z',
        request.change_profile,
        request,
      )
      if (baseline && !exactCohortMatches(baseline, requestRun)) {
        await fulfillError(
          route,
          400,
          'invalid evaluation request: candidate cohort must match the baseline',
        )
        return
      }
      const idempotentRun = state.runs.find(
        (run) => run.client_request_id === request.client_request_id,
      )
      if (idempotentRun) {
        if (!createRequestMatchesRun(request, idempotentRun)) {
          await fulfillError(
            route,
            409,
            'conflict: client_request_id was already used for a different evaluation run',
          )
          return
        }
        await state.mutationDelay()
        await fulfillJSON(route, 201, idempotentRun)
        return
      }
      state.createdRequests.push(request)
      const created = evaluationRun(
        request.client_request_id,
        request.name.trim(),
        'pending',
        '2026-08-29T01:00:00Z',
        request.change_profile,
        {
          description: request.description.trim(),
          mode: request.mode,
          target_id: request.target_id,
          suite_ids: request.suite_ids,
          track_ids: request.track_ids,
          sample_limit: request.sample_limit,
          concurrency: request.concurrency,
          capacity_slo: request.capacity_slo,
          capacity_load_protocol: request.capacity_load_protocol,
          seed: request.seed,
          baseline_run_id: request.baseline_run_id,
          evidence_level: 'E0',
        },
      )
      state.runs = [created, ...state.runs]
      await state.mutationDelay()
      await fulfillJSON(route, 201, created)
      return
    }
    if (route.request().method() !== 'GET') {
      await fulfillError(route, 405, 'method not allowed')
      return
    }
    const url = new URL(route.request().url())
    const offset = Number.parseInt(url.searchParams.get('cursor') || '0', 10)
    state.ledgerRequestCount += 1
    if (offset > 0 && state.firstLoadMorePending) {
      state.firstLoadMorePending = false
      await fulfillError(route, 503, 'temporary ledger page failure')
      return
    }
    const pageSize = state.options.runPageSize || 50
    const pageRuns = state.runs.slice(offset, offset + pageSize)
    const nextOffset = offset + pageRuns.length
    await new Promise<void>((resolve) => setTimeout(resolve, state.options.ledgerDelayMs || 0))
    await fulfillJSON(route, 200, {
      schema_version: 'evaluation.v1',
      runs: pageRuns,
      ...(nextOffset < state.runs.length ? { next_cursor: String(nextOffset) } : {}),
      total_runs: state.runs.length,
      ledger_complete: state.ledgerWarningCount === 0,
      warning_count: state.ledgerWarningCount,
      warnings: state.ledgerWarnings,
    })
  })
}
