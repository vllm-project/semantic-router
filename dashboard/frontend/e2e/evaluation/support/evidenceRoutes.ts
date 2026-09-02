import type { Page } from '@playwright/test'

import { failureSummary } from './campaignFixtures'
import { evaluationComparison, evaluationReport } from './reportFixtures'
import { denseReportMetric } from './reportMetricFixtures'
import {
  controlledPairCohortMatches,
  exactCohortMatches,
  fulfillError,
  fulfillJSON,
  type EvaluationMockState,
} from './state'

export async function registerEvidenceRoutes(
  page: Page,
  state: EvaluationMockState,
): Promise<void> {
  await page.route('**/api/evaluation/v1/compare?*', async (route) => {
    if (state.ledgerWarningCount > 0) {
      await fulfillError(
        route,
        409,
        'conflict: evaluation run ledger is incomplete; repair quarantined evidence before comparing runs',
      )
      return
    }
    const url = new URL(route.request().url())
    const baselineRunID = url.searchParams.get('baseline_run_id') || ''
    const candidateRunID = url.searchParams.get('candidate_run_id') || ''
    state.comparisonRequests.push({ baselineRunID, candidateRunID })
    const baseline = state.runs.find((run) => run.id === baselineRunID)
    const candidate = state.runs.find((run) => run.id === candidateRunID)
    if (!baseline || !candidate) {
      await fulfillError(route, 404, 'not found: evaluation run')
      return
    }
    if (baseline.status !== 'completed' || candidate.status !== 'completed') {
      await fulfillError(route, 409, 'conflict: comparison requires completed reports')
      return
    }
    if (candidate.baseline_run_id !== baseline.id) {
      await fulfillError(
        route,
        400,
        'invalid evaluation request: candidate baseline_run_id must identify the compared baseline',
      )
      return
    }
    const controlledPairMatches =
      baseline.controlled_pair?.role === 'baseline' &&
      candidate.controlled_pair?.role === 'candidate' &&
      baseline.controlled_pair.pair_id === candidate.controlled_pair.pair_id &&
      candidate.baseline_run_id === baseline.id &&
      baseline.mode === 'live' &&
      candidate.mode === 'live' &&
      controlledPairCohortMatches(baseline, candidate)
    if (!exactCohortMatches(baseline, candidate) && !controlledPairMatches) {
      await fulfillError(
        route,
        400,
        'invalid evaluation request: baseline and candidate report cohorts do not match',
      )
      return
    }
    await fulfillJSON(route, 200, {
      ...evaluationComparison,
      baseline_run_id: baseline.id,
      candidate_run_id: candidate.id,
      gates: evaluationComparison.gates.map((gate) =>
        gate.id === 'G3'
          ? {
              ...gate,
              sample_count: controlledPairMatches ? undefined : gate.sample_count,
              rationale: controlledPairMatches
                ? 'A run comparison cannot decide G3; use a campaign with controlled AB/BA paired-live outcomes.'
                : gate.rationale,
              evidence_refs: [
                'server-reduction:comparative-g3.v1',
                `run:baseline:${baseline.id}`,
                `run:candidate:${candidate.id}`,
                'comparison-statistic:joint.normalized_regret',
              ],
            }
          : gate,
      ),
    })
  })
  await page.route('**/api/evaluation/v1/runs/*/events', async (route) => {
    state.eventStreamCount += 1
    if (state.options.eventStreamCloseOnce && state.eventStreamCount === 1) {
      await route.fulfill({ status: 204 })
      return
    }
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 2] || '')
    const run = state.runs.find((candidate) => candidate.id === id)
    if (!run) {
      await fulfillError(route, 404, 'not found: evaluation run')
      return
    }
    const completesRun = state.options.completeRunOnEventStream === id
    const completedRun = completesRun
      ? {
          ...run,
          status: 'completed' as const,
          completed_at: '2026-08-29T00:06:00Z',
          progress: {
            percent: 100,
            completed: run.track_ids.length,
            total: run.track_ids.length,
            message: 'Evaluation completed',
          },
        }
      : null
    if (completedRun) {
      state.runs = state.runs.map((candidate) => (candidate.id === id ? completedRun : candidate))
    }
    const event = completesRun
      ? {
          id: '2',
          run_id: id,
          type: 'completed',
          timestamp: '2026-08-29T00:06:00Z',
          message: 'Evaluation completed',
          progress: completedRun?.progress,
        }
      : {
          id: '2',
          run_id: id,
          type: 'progress',
          timestamp: '2026-08-29T00:05:00Z',
          message: 'Executing routing track from SSE',
        }
    const eventCount = completesRun
      ? 1
      : Math.max(1, Math.min(50, state.options.eventStreamEventCount || 1))
    const eventBody = Array.from({ length: eventCount }, (_, index) => {
      const streamedEvent = {
        ...event,
        id: String(index + 2),
        ...(index === eventCount - 1
          ? {}
          : {
              timestamp: new Date(Date.parse(event.timestamp) - (eventCount - index) * 1_000)
                .toISOString()
                .replace('.000Z', 'Z'),
              message: `Durable evaluation progress ${index + 1}`,
            }),
      }
      return `id: ${streamedEvent.id}\nevent: ${streamedEvent.type}\ndata: ${JSON.stringify(streamedEvent)}\n\n`
    }).join('')
    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      headers: { 'Cache-Control': 'no-cache' },
      body: eventBody,
    })
  })
  await page.route('**/api/evaluation/v1/runs/*/report', async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const id = decodeURIComponent(parts[parts.length - 2] || '')
    const reportDelay = state.options.reportDelayMs ?? 0
    await new Promise<void>((resolve) => setTimeout(resolve, reportDelay))
    state.reportRequests.push(id)
    if (state.options.reportFailureIDs?.includes(id)) {
      await fulfillError(
        route,
        state.options.reportFailureStatus || 503,
        state.options.reportFailureStatus === 404
          ? 'not found: evaluation report'
          : 'report storage is temporarily unavailable',
      )
      return
    }
    const run = state.runs.find((candidate) => candidate.id === id)
    if (!run) {
      await fulfillError(route, 404, 'not found: evaluation run')
      return
    }
    if (run.status !== 'completed') {
      await fulfillError(
        route,
        409,
        'conflict: evaluation report is available only for completed runs',
      )
      return
    }
    const report = evaluationReport(run)
    if (state.options.reportMetricCount && report.metrics.length) {
      report.metrics = Array.from({ length: state.options.reportMetricCount }, (_, index) =>
        denseReportMetric(index),
      )
      report.tracks = report.tracks.map((track) => ({
        ...track,
        metrics: report.metrics.filter((metric) => metric.track_id === track.track_id),
      }))
    }
    if (typeof state.options.diagnosticArtifactBodies?.capacityProfile === 'string') {
      report.artifacts = [
        ...report.artifacts,
        {
          id: 'capacity-profile-json',
          name: 'capacity-profile.json',
          kind: 'json',
          uri: 'capacity-profile.json',
          digest: report.artifacts[0]?.digest,
          media_type: 'application/json',
          size_bytes: state.options.diagnosticArtifactBodies.capacityProfile.length,
        },
      ]
    }
    await fulfillJSON(route, 200, report)
  })
  await page.route('**/api/evaluation/v1/runs/*/artifacts/*', async (route) => {
    const parts = new URL(route.request().url()).pathname.split('/')
    const artifactID = decodeURIComponent(parts[parts.length - 1] || '')
    const id = decodeURIComponent(parts[parts.length - 3] || '')
    const run = state.runs.find((candidate) => candidate.id === id)
    if (!run) {
      await fulfillError(route, 404, 'not found: evaluation run')
      return
    }
    if (run.status !== 'completed') {
      await fulfillError(route, 409, 'conflict: evaluation evidence is not sealed')
      return
    }
    if (artifactID === 'failure-summary-json') {
      if (typeof state.options.diagnosticArtifactBodies?.failureSummary === 'string') {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: state.options.diagnosticArtifactBodies.failureSummary,
        })
        return
      }
      await fulfillJSON(route, 200, failureSummary(run))
      return
    }
    if (
      artifactID === 'capacity-profile-json' &&
      typeof state.options.diagnosticArtifactBodies?.capacityProfile === 'string'
    ) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: state.options.diagnosticArtifactBodies.capacityProfile,
      })
      return
    }
    if (['metrics-json', 'gates-json', 'provenance-json'].includes(artifactID)) {
      await fulfillJSON(route, 200, { schema_version: 'evaluation.v1' })
      return
    }
    if (artifactID === 'checksums-sha256') {
      await route.fulfill({
        status: 200,
        contentType: 'text/plain',
        body: '0123456789abcdef  metrics.json\n',
      })
      return
    }
    await fulfillError(route, 404, 'not found: evaluation artifact')
  })
}
