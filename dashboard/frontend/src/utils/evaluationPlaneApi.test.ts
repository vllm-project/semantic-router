import { afterEach, describe, expect, it, vi } from 'vitest'

import type { CreateEvaluationRunPayload, EvaluationCatalog } from '../types/evaluationPlane'
import type { CreateEvaluationCampaignPayload } from '../types/evaluationCampaign'
import type { CreateEvaluationControlledPairPayload } from '../types/evaluationControlledPair'
import { buildCreateRunPayload } from './evaluationCreateRunContract'
import {
  cancelEvaluationControlledPair,
  cancelEvaluationRun,
  compareEvaluationRuns,
  createEvaluationCampaign,
  createEvaluationControlledPair,
  createEvaluationRun,
  deleteEvaluationControlledPair,
  deleteEvaluationRun,
  getEvaluationCatalog,
  getEvaluationCampaignReadiness,
  getEvaluationCampaign,
  getEvaluationControlledPair,
  getEvaluationArtifactURL,
  getEvaluationReport,
  getEvaluationRun,
  isDownloadableEvaluationArtifact,
  listEvaluationRuns,
  startEvaluationRun,
} from './evaluationPlaneApi'
import {
  BASELINE_RUN_ID,
  CAMPAIGN_CONFIRMATION_ID,
  CAMPAIGN_ID,
  CAMPAIGN_LIVE_BASELINE_ID,
  CAMPAIGN_LIVE_ID,
  CANDIDATE_RUN_ID,
  catalog,
  completedRun,
  CREATE_RUN_ID,
  methodReportFixture,
  QUARANTINED_EVIDENCE_ID,
  reportFor,
  request,
  RUN_ID,
  run,
} from '../test/evaluationPlaneApiFixture'
import { decodeEvaluationReport } from './evaluationReportContract'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

afterEach(() => vi.unstubAllGlobals())

describe('Evaluation Plane API', () => {
  it('rejects non-contract create fields and catalog identities', () => {
    const untrusted = {
      ...request,
      endpoint: 'https://arbitrary.invalid/v1',
      url: 'https://arbitrary.invalid',
      hidden_label: 'must never cross the browser contract',
    } as CreateEvaluationRunPayload
    expect(() => buildCreateRunPayload(untrusted, catalog)).toThrow(/non-contract fields/i)
    expect(() =>
      buildCreateRunPayload({ ...request, target_id: 'https://arbitrary.invalid' }, catalog),
    ).toThrow(/available evaluation source/i)
    expect(() =>
      buildCreateRunPayload({ ...request, change_profile: 'selector' }, catalog),
    ).toThrow(/type of change/i)
    expect(() =>
      buildCreateRunPayload({ ...request, client_request_id: 'retry-me' }, catalog),
    ).toThrow(/canonical UUID/i)
    expect(() =>
      buildCreateRunPayload({ ...request, suite_ids: ['suite-routing', 'suite-routing'] }, catalog),
    ).toThrow(/benchmarks.*duplicates/i)
    expect(() => buildCreateRunPayload({ ...request, concurrency: 1.5 }, catalog)).toThrow(
      /concurrency must be an integer/i,
    )
  })

  it('preserves the client idempotency token in the create payload', () => {
    const payload = buildCreateRunPayload(
      { ...request, client_request_id: '4d0b4f2c-1fc5-40b0-b04e-876ad9d4d8e2' },
      catalog,
    )
    expect(payload.client_request_id).toBe('4d0b4f2c-1fc5-40b0-b04e-876ad9d4d8e2')
  })

  it('uses only the current campaign endpoints and canonical create body', async () => {
    const campaignRequest: CreateEvaluationCampaignPayload = {
      client_request_id: CAMPAIGN_ID,
      name: '  Recipe decision  ',
      description: '  Exact evidence roles.  ',
      change_profile: 'recipe',
      gate_bindings: {
        g2_run_id: CANDIDATE_RUN_ID,
        g3_controlled_pair: {
          baseline_run_id: CAMPAIGN_LIVE_BASELINE_ID,
          candidate_run_id: CAMPAIGN_LIVE_ID,
        },
        g5_fidelity: {
          reference_run_id: BASELINE_RUN_ID,
          live_run_id: CAMPAIGN_CONFIRMATION_ID,
        },
      },
    }
    const fetch = vi.fn().mockResolvedValue(jsonResponse({}, 201))
    vi.stubGlobal('fetch', fetch)

    await expect(createEvaluationCampaign(campaignRequest)).rejects.toThrow(
      /campaign response is incomplete/i,
    )
    expect(fetch).toHaveBeenCalledWith(
      '/api/evaluation/v1/campaigns',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          ...campaignRequest,
          name: 'Recipe decision',
          description: 'Exact evidence roles.',
        }),
      }),
    )

    fetch.mockResolvedValueOnce(jsonResponse({}))
    await expect(getEvaluationCampaign(CAMPAIGN_ID)).rejects.toThrow(
      /campaign response is incomplete/i,
    )
    expect(fetch).toHaveBeenLastCalledWith(`/api/evaluation/v1/campaigns/${CAMPAIGN_ID}`, {
      signal: undefined,
    })
  })

  it('posts only the five controlled-pair UUIDs to the server-owned execution endpoint', async () => {
    const controlledPairRequest: CreateEvaluationControlledPairPayload = {
      client_request_id: '88888888-8888-4888-8888-888888888888',
      baseline_source_run_id: CAMPAIGN_LIVE_BASELINE_ID,
      candidate_source_run_id: CAMPAIGN_LIVE_ID,
      baseline_run_id: '99999999-9999-4999-8999-999999999999',
      candidate_run_id: 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
    }
    const fetch = vi.fn().mockResolvedValue(jsonResponse({}, 201))
    vi.stubGlobal('fetch', fetch)

    await expect(createEvaluationControlledPair(controlledPairRequest)).rejects.toThrow(
      /controlled pair response/i,
    )
    expect(fetch).toHaveBeenCalledWith(
      '/api/evaluation/v1/controlled-pairs',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(controlledPairRequest),
      }),
    )
    expect(Object.keys(JSON.parse(fetch.mock.calls[0]?.[1]?.body as string)).sort()).toEqual([
      'baseline_run_id',
      'baseline_source_run_id',
      'candidate_run_id',
      'candidate_source_run_id',
      'client_request_id',
    ])
  })

  it('uses the authoritative aggregate resource and lifecycle endpoints for controlled pairs', async () => {
    const pairID = '88888888-8888-4888-8888-888888888888'
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValueOnce(jsonResponse({}))
      .mockResolvedValueOnce(jsonResponse({}))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
    vi.stubGlobal('fetch', fetch)

    await expect(getEvaluationControlledPair(pairID)).rejects.toThrow(/controlled pair response/i)
    await expect(cancelEvaluationControlledPair(pairID)).rejects.toThrow(
      /controlled pair response/i,
    )
    await expect(deleteEvaluationControlledPair(pairID)).resolves.toBeUndefined()

    expect(fetch.mock.calls).toEqual([
      [`/api/evaluation/v1/controlled-pairs/${pairID}`, { signal: undefined }],
      [`/api/evaluation/v1/controlled-pairs/${pairID}/cancel`, { method: 'POST' }],
      [`/api/evaluation/v1/controlled-pairs/${pairID}`, { method: 'DELETE' }],
    ])
  })

  it('rejects partially supported suites and tracks before creating a run', () => {
    const expandedCatalog: EvaluationCatalog = {
      ...catalog,
      tracks: [...catalog.tracks],
      suites: [
        ...catalog.suites,
        {
          id: 'suite-partial',
          executors: { replay: 'fixture-replay.v1' },
          name: 'Partially supported suite',
          description: 'Requires routing and agentic evidence',
          track_ids: ['routing', 'agentic'],
          modes: ['replay'],
          evidence_level: 'E2',
          revision: 'suite-partial.v1',
          tags: ['fixture'],
          methods: [
            {
              id: 'fixture.partial-routing.v1',
              track_id: 'routing',
              qualified_gate_ids: [],
              evidence_source: 'diagnostic_fixture',
              status: 'configured',
            },
            {
              id: 'fixture.partial-agentic.v1',
              track_id: 'agentic',
              qualified_gate_ids: [],
              evidence_source: 'diagnostic_fixture',
              status: 'configured',
            },
          ],
        },
      ],
    }

    expect(
      buildCreateRunPayload(
        {
          ...request,
          suite_ids: ['suite-partial'],
          track_ids: ['routing'],
        },
        expandedCatalog,
      ),
    ).toMatchObject({ suite_ids: ['suite-partial'], track_ids: ['routing'] })
    expect(() =>
      buildCreateRunPayload(
        {
          ...request,
          track_ids: ['agentic'],
        },
        expandedCatalog,
      ),
    ).toThrow(/evaluation area/i)
  })

  it('links only backend-allowlisted report artifacts and never the run manifest', () => {
    for (const name of [
      'capacity-profile.json',
      'metrics.json',
      'gates.json',
      'failure-summary.json',
      'provenance.json',
      'checksums.sha256',
    ]) {
      expect(isDownloadableEvaluationArtifact({ id: name, name, kind: 'evidence' })).toBe(true)
    }
    expect(
      isDownloadableEvaluationArtifact({
        id: 'routing-traces',
        name: 'routing-traces.jsonl',
        kind: 'jsonl',
      }),
    ).toBe(false)
    expect(
      isDownloadableEvaluationArtifact({ id: 'comparison', name: 'comparison.json', kind: 'json' }),
    ).toBe(false)
    expect(
      isDownloadableEvaluationArtifact({
        id: 'records.jsonl',
        name: 'Case records',
        kind: 'records',
      }),
    ).toBe(false)
    expect(
      isDownloadableEvaluationArtifact({
        id: 'failure-summary-json',
        name: 'failure-summary.json',
        kind: 'json',
      }),
    ).toBe(true)
    expect(
      isDownloadableEvaluationArtifact({
        id: 'run-manifest.json',
        name: 'Run manifest',
        kind: 'manifest',
      }),
    ).toBe(false)
    expect(getEvaluationArtifactURL(RUN_ID, 'records/visible.jsonl')).toBe(
      `/api/evaluation/v1/runs/${RUN_ID}/artifacts/records%2Fvisible.jsonl`,
    )
  })

  it('uses only the versioned catalog and run lifecycle endpoints', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(catalog))
      .mockResolvedValueOnce(
        jsonResponse({
          schema_version: 'evaluation.v1',
          runs: [run],
          total_runs: 1,
          ledger_complete: true,
          warning_count: 0,
          warnings: [],
        }),
      )
      .mockResolvedValueOnce(jsonResponse(run))
      .mockResolvedValueOnce(
        jsonResponse({ ...run, id: CREATE_RUN_ID, client_request_id: CREATE_RUN_ID }, 201),
      )
      .mockResolvedValueOnce(jsonResponse({ ...run, status: 'running' }))
      .mockResolvedValueOnce(jsonResponse({ ...run, status: 'cancelled' }))
      .mockResolvedValueOnce(new Response(null, { status: 204 }))
      .mockResolvedValueOnce(jsonResponse(reportFor(completedRun)))
      .mockResolvedValueOnce(
        jsonResponse({
          schema_version: 'evaluation.v1',
          attestation_revision: 'evaluation-server-attestation.v2',
          baseline_run_id: BASELINE_RUN_ID,
          candidate_run_id: CANDIDATE_RUN_ID,
          verdict: 'unavailable',
          summary: 'No paired evidence is available.',
          metrics: [],
          statistics: [],
          gates: Array.from({ length: 10 }, (_, index) => ({
            id: `G${index}`,
            name: `Gate ${index}`,
            disposition: 'required',
            verdict: index === 3 ? 'unavailable' : 'pass',
            change_profile: 'recipe',
            contract_version: 'evaluation-release-gates.v2',
            evidence_refs:
              index === 3
                ? [
                    'server-reduction:comparative-g3.v1',
                    `run:baseline:${BASELINE_RUN_ID}`,
                    `run:candidate:${CANDIDATE_RUN_ID}`,
                    'comparison-statistic:joint.normalized_regret',
                  ]
                : [`gate:G${index}`],
            evidence_level: index === 3 ? 'E0' : 'E5',
            ...(index === 3 ? { owner: 'recipe-and-model-pool' } : {}),
          })),
          recommendations: [],
          created_at: '2026-08-29T00:00:02Z',
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await getEvaluationCatalog()
    await expect(listEvaluationRuns()).resolves.toEqual({
      schema_version: 'evaluation.v1',
      runs: [run],
      total_runs: 1,
      ledger_complete: true,
      warning_count: 0,
      warnings: [],
    })
    await getEvaluationRun(RUN_ID)
    await createEvaluationRun(request, catalog)
    await startEvaluationRun(RUN_ID)
    await cancelEvaluationRun(RUN_ID)
    await deleteEvaluationRun(RUN_ID)
    await getEvaluationReport(RUN_ID)
    await compareEvaluationRuns(BASELINE_RUN_ID, CANDIDATE_RUN_ID)

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/api/evaluation/v1/catalog',
      '/api/evaluation/v1/runs',
      `/api/evaluation/v1/runs/${RUN_ID}`,
      '/api/evaluation/v1/runs',
      `/api/evaluation/v1/runs/${RUN_ID}/start`,
      `/api/evaluation/v1/runs/${RUN_ID}/cancel`,
      `/api/evaluation/v1/runs/${RUN_ID}`,
      `/api/evaluation/v1/runs/${RUN_ID}/report`,
      `/api/evaluation/v1/compare?baseline_run_id=${BASELINE_RUN_ID}&candidate_run_id=${CANDIDATE_RUN_ID}`,
    ])
    expect(fetchMock.mock.calls[3]?.[1]).toMatchObject({ method: 'POST' })
    const createBody = JSON.parse(String(fetchMock.mock.calls[3]?.[1]?.body))
    expect(createBody).toMatchObject({ change_profile: 'recipe' })
    expect(createBody).not.toHaveProperty('auto_start')
    expect(fetchMock.mock.calls[5]?.[1]).toMatchObject({ method: 'POST' })
    expect(fetchMock.mock.calls[6]?.[1]).toMatchObject({ method: 'DELETE' })
  })

  it('page-merges readiness across histories larger than one server page', async () => {
    const profile = catalog.change_profiles[0]
    const runIDs = Array.from(
      { length: 201 },
      (_, index) => `aaaaaaaa-aaaa-4aaa-8aaa-${String(index).padStart(12, '0')}`,
    )
    const readinessPage = (pageRunIDs: string[], nextCursor?: string) => ({
      schema_version: 'evaluation.v1',
      change_profile: profile.id,
      ...(nextCursor ? { next_cursor: nextCursor } : {}),
      total_runs: runIDs.length,
      slots: profile.campaign_slots.map((slot) => ({
        gate_id: slot.gate_id,
        binding_kind: slot.binding_kind,
        eligible_run_ids: slot.binding_kind === 'run' ? pageRunIDs : [],
        controlled_pair_source_run_ids: slot.binding_kind === 'controlled_pair' ? pageRunIDs : [],
        controlled_pair_candidate_run_ids: [],
        fidelity_reference_run_ids: slot.binding_kind === 'fidelity_pair' ? pageRunIDs : [],
        fidelity_live_run_ids: [],
      })),
    })
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse(readinessPage(runIDs.slice(0, 200), 'page-2')))
      .mockResolvedValueOnce(jsonResponse(readinessPage(runIDs.slice(200))))
    vi.stubGlobal('fetch', fetchMock)

    const readiness = await getEvaluationCampaignReadiness(profile)
    expect(readiness.total_runs).toBe(201)
    expect(
      readiness.slots.find((slot) => slot.binding_kind === 'run')?.eligible_run_ids,
    ).toHaveLength(201)
    expect(
      readiness.slots.find((slot) => slot.binding_kind === 'controlled_pair')
        ?.controlled_pair_source_run_ids,
    ).toHaveLength(201)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body))).toMatchObject({
      cursor: 'page-2',
      limit: 200,
    })
  })

  it('rejects malformed or duplicate catalog members', async () => {
    const duplicateTrack = { ...catalog.tracks[0] }
    const duplicateSuite = { ...catalog.suites[0] }
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ ...catalog, tracks: catalog.tracks.slice(0, -1) }))
      .mockResolvedValueOnce(
        jsonResponse({
          ...catalog,
          tracks: [...catalog.tracks.slice(0, -1), duplicateTrack],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          ...catalog,
          suites: [{ ...catalog.suites[0], methods: [] }, ...catalog.suites.slice(1)],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({ ...catalog, suites: [...catalog.suites, duplicateSuite] }),
      )
    vi.stubGlobal('fetch', fetchMock)

    for (let attempt = 0; attempt < 4; attempt += 1) {
      await expect(getEvaluationCatalog()).rejects.toThrow(/catalog response is incomplete/i)
    }
  })

  it('requires explicit and internally consistent ledger integrity metadata', async () => {
    const warning = {
      code: 'corrupt_run_bundle',
      evidence_id: QUARANTINED_EVIDENCE_ID,
      evidence_file: 'status.json',
      message: 'Durable run status evidence is unreadable or invalid and has been quarantined.',
    }
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        jsonResponse({
          schema_version: 'evaluation.v1',
          runs: [run],
          total_runs: 1,
          ledger_complete: false,
          warning_count: 1,
          warnings: [warning],
        }),
      )
      .mockResolvedValueOnce(jsonResponse([run]))
      .mockResolvedValueOnce(
        jsonResponse({
          schema_version: 'evaluation.v1',
          runs: [run],
          total_runs: 1,
          ledger_complete: true,
          warning_count: 1,
          warnings: [warning],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          schema_version: 'evaluation.v1',
          runs: [],
          next_cursor: 'cursor-1',
          total_runs: 1,
          ledger_complete: true,
          warning_count: 0,
          warnings: [],
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await expect(listEvaluationRuns()).resolves.toMatchObject({
      ledger_complete: false,
      warning_count: 1,
      warnings: [warning],
    })
    await expect(listEvaluationRuns()).rejects.toThrow(/ledger response is invalid or incomplete/i)
    await expect(listEvaluationRuns()).rejects.toThrow(/ledger response is invalid or incomplete/i)
    await expect(listEvaluationRuns({ cursor: 'cursor-1' })).rejects.toThrow(
      /ledger response is invalid or incomplete/i,
    )
  })

  it('rejects unknown fields and contract revisions in otherwise current resources', async () => {
    const comparison = {
      schema_version: 'evaluation.v1',
      attestation_revision: 'evaluation-server-attestation.v2',
      baseline_run_id: BASELINE_RUN_ID,
      candidate_run_id: CANDIDATE_RUN_ID,
      verdict: 'unavailable',
      summary: 'No paired evidence is available.',
      metrics: [],
      gates: [],
      recommendations: [],
      created_at: '2026-08-29T00:00:02Z',
    }
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ ...catalog, retired_contract: true }))
      .mockResolvedValueOnce(jsonResponse({ ...run, retired_status: 'ready' }))
      .mockResolvedValueOnce(
        jsonResponse({
          ...reportFor(completedRun),
          summary: { ...reportFor(completedRun).summary, retired_score: 1 },
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ ...comparison, retired_verdict: 'pass' }))
      .mockResolvedValueOnce(
        jsonResponse({ ...catalog, gate_contract_version: 'evaluation-release-gates.v3' }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await expect(getEvaluationCatalog()).rejects.toThrow(/catalog response is incomplete/i)
    await expect(getEvaluationRun(RUN_ID)).rejects.toThrow(/run response is incomplete/i)
    await expect(getEvaluationReport(RUN_ID)).rejects.toThrow(/report response is incomplete/i)
    await expect(compareEvaluationRuns(BASELINE_RUN_ID, CANDIDATE_RUN_ID)).rejects.toThrow(
      /requested pair/i,
    )
    await expect(getEvaluationCatalog()).rejects.toThrow(/catalog response is incomplete/i)
  })

  it('rejects malformed sealed method reports before rendering', () => {
    const report = reportFor(completedRun)
    const valid = methodReportFixture()
    const malformedReports = [
      { ...valid, audc: -0.01 },
      { ...valid, nauc: 1.01 },
      { ...valid, action_refs: [...valid.action_refs, valid.action_refs[0]] },
      { ...valid, slice_refs: [...valid.slice_refs, valid.slice_refs[0]] },
      { ...valid, analysis_plan: { ...valid.analysis_plan, cluster_unit: 'attempt' } },
      {
        ...valid,
        raw_shared_domain_curve: [
          ...valid.raw_shared_domain_curve,
          valid.raw_shared_domain_curve[0],
        ],
      },
      {
        ...valid,
        raw_shared_domain_curve: [
          {
            ...valid.raw_shared_domain_curve[0],
            action: { schema_version: 'evaluation-method.v2', id: 'unknown' },
          },
        ],
      },
    ]
    for (const malformed of malformedReports) {
      expect(() =>
        decodeEvaluationReport({ ...report, method_reports: [malformed] }, RUN_ID),
      ).toThrow(/report response is incomplete/i)
    }
    expect(
      decodeEvaluationReport({ ...report, method_reports: [valid] }, RUN_ID).method_reports,
    ).toHaveLength(1)
  })

  it('requires server-only fields on every published report', () => {
    const report = reportFor(completedRun)
    const withoutAttestation = { ...report }
    delete (withoutAttestation as { attestation_revision?: unknown }).attestation_revision
    expect(() => decodeEvaluationReport(withoutAttestation, RUN_ID)).toThrow(
      /current server contract/i,
    )

    const withoutMethodReductions = { ...report }
    delete (withoutMethodReductions as { method_reports?: unknown }).method_reports
    expect(() => decodeEvaluationReport(withoutMethodReductions, RUN_ID)).toThrow(
      /report response is incomplete/i,
    )

    const withoutRoutingReduction = { ...report }
    delete (withoutRoutingReduction as { routing_recipe_report?: unknown }).routing_recipe_report
    expect(() => decodeEvaluationReport(withoutRoutingReduction, RUN_ID)).toThrow(
      /server-owned routing recipe field/i,
    )

    expect(() =>
      decodeEvaluationReport({ ...report, routing_recipe_report: undefined }, RUN_ID),
    ).toThrow(/explicit null/i)
  })

  it('rejects forged published report envelopes before rendering', () => {
    const report = reportFor(completedRun)
    const withoutPublishedGateFields = {
      ...report,
      gates: report.gates.map((gate, index) =>
        index === 0 ? { ...gate, sample_count: undefined } : gate,
      ),
    }
    expect(() => decodeEvaluationReport(withoutPublishedGateFields, RUN_ID)).toThrow(
      /report response is incomplete/i,
    )

    const withoutEvidenceReferences = {
      ...report,
      gates: report.gates.map((gate, index) =>
        index === 0 ? { ...gate, evidence_refs: [] } : gate,
      ),
    }
    expect(() => decodeEvaluationReport(withoutEvidenceReferences, RUN_ID)).toThrow(
      /report response is incomplete/i,
    )

    const duplicateEvidenceReferences = {
      ...report,
      gates: report.gates.map((gate, index) =>
        index === 0
          ? { ...gate, evidence_refs: [gate.evidence_refs[0], gate.evidence_refs[0]] }
          : gate,
      ),
    }
    expect(() => decodeEvaluationReport(duplicateEvidenceReferences, RUN_ID)).toThrow(
      /report response is incomplete/i,
    )

    expect(() =>
      decodeEvaluationReport({ ...report, gates: report.gates.slice(0, -1) }, RUN_ID),
    ).toThrow(/report response is incomplete/i)

    const wrongProfile = {
      ...report,
      gates: report.gates.map((gate, index) =>
        index === 0 ? { ...gate, change_profile: 'selector' } : gate,
      ),
    }
    expect(() => decodeEvaluationReport(wrongProfile, RUN_ID)).toThrow(
      /report response is incomplete/i,
    )

    expect(() => decodeEvaluationReport({ ...report, tracks: [] }, RUN_ID)).toThrow(
      /report response is incomplete/i,
    )
    expect(() =>
      decodeEvaluationReport(
        {
          ...report,
          tracks: [{ ...report.tracks[0], gates: [] }],
        },
        RUN_ID,
      ),
    ).toThrow(/report response is incomplete/i)
    expect(() =>
      decodeEvaluationReport(
        {
          ...report,
          summary: { ...report.summary, unavailable_gates: 9 },
        },
        RUN_ID,
      ),
    ).toThrow(/report response is incomplete/i)
    expect(() =>
      decodeEvaluationReport(
        { ...report, summary: { ...report.summary, verdict: 'pass' } },
        RUN_ID,
      ),
    ).toThrow(/report response is incomplete/i)
    expect(() =>
      decodeEvaluationReport(
        { ...report, summary: { ...report.summary, verdict: 'waived' } },
        RUN_ID,
      ),
    ).toThrow(/report response is incomplete/i)
    expect(() =>
      decodeEvaluationReport(
        {
          ...report,
          gates: report.gates.map((gate, index) =>
            index === 0 ? { ...gate, disposition: 'waived', verdict: 'waived' } : gate,
          ),
        },
        RUN_ID,
      ),
    ).toThrow(/report response is incomplete/i)
  })

  it('rejects responses outside the current run and attestation contracts', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(jsonResponse({ run }))
      .mockResolvedValueOnce(
        jsonResponse({
          schema_version: 'evaluation.v1',
          run,
          metrics: [],
          gates: [],
        }),
      )
    vi.stubGlobal('fetch', fetchMock)

    await expect(getEvaluationRun(RUN_ID)).rejects.toThrow(/evaluation\.v1 contract/i)
    await expect(getEvaluationReport(RUN_ID)).rejects.toThrow(/current server contract/i)
  })
})
