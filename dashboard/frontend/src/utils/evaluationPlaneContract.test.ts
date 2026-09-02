import { describe, expect, it } from 'vitest'

import {
  EVALUATION_CAMPAIGN_COHORT_SCHEMA_VERSION,
  EVALUATION_GATE_DISPOSITIONS,
  EVALUATION_GATE_VERDICTS,
  EVALUATION_METHOD_EVIDENCE_SOURCE,
  EVALUATION_METHOD_EVIDENCE_SOURCES,
  EVALUATION_SUMMARY_VERDICTS,
  isEvaluationMethodEvidenceSource,
  type EvaluationCatalogCampaignSlot,
  type EvaluationCatalogSuite,
} from '../types/evaluationPlane'
import { decodeEvaluationCatalog } from './evaluationCatalogContract'
import {
  catalogWithUnavailableMixture,
  canonicalCampaignSlots,
  evaluationCatalogFixture,
  mixture,
  RUN_ID,
  strictContractRun as run,
  unavailableMixture,
} from '../test/evaluationPlaneApiFixture'
import {
  decodeEvaluationRun,
  decodeEvaluationRunEvent,
  decodeEvaluationRunLedger,
  isCanonicalEvaluationRunID,
} from './evaluationRunContract'

describe('evaluation current-contract codec', () => {
  it('keeps release check and summary verdict inventories exact', () => {
    expect(EVALUATION_GATE_DISPOSITIONS).toEqual(['required', 'advisory', 'not_applicable'])
    expect(EVALUATION_GATE_VERDICTS).toEqual(['pass', 'fail', 'unavailable', 'not_applicable'])
    expect(EVALUATION_SUMMARY_VERDICTS).toEqual(['pass', 'fail', 'unavailable'])

    const catalog = evaluationCatalogFixture()
    const profile = catalog.change_profiles[0]
    expect(() =>
      decodeEvaluationCatalog({
        ...catalog,
        change_profiles: [
          {
            ...profile,
            campaign_slots: profile.campaign_slots.map((slot, index) =>
              index === 0 ? { ...slot, disposition: 'waived' } : slot,
            ),
          },
        ],
      }),
    ).toThrow(/catalog response is incomplete/i)
  })

  it('derives strict method evidence-source validation from one typed inventory', () => {
    expect(EVALUATION_METHOD_EVIDENCE_SOURCES).toEqual(
      Object.values(EVALUATION_METHOD_EVIDENCE_SOURCE),
    )
    expect(new Set(EVALUATION_METHOD_EVIDENCE_SOURCES).size).toBe(
      EVALUATION_METHOD_EVIDENCE_SOURCES.length,
    )
    for (const source of EVALUATION_METHOD_EVIDENCE_SOURCES) {
      expect(isEvaluationMethodEvidenceSource(source)).toBe(true)
    }
    expect(isEvaluationMethodEvidenceSource('unknown_source')).toBe(false)

    const catalog = evaluationCatalogFixture()
    for (const source of EVALUATION_METHOD_EVIDENCE_SOURCES) {
      const suite = catalog.suites[0]
      const method = suite.methods[0]
      const decoded = decodeEvaluationCatalog({
        ...catalog,
        suites: [
          {
            ...suite,
            evidence_level: 'E0',
            methods: [
              {
                ...method,
                qualified_gate_ids:
                  source === EVALUATION_METHOD_EVIDENCE_SOURCE.SERVER_BROKERED_LIVE ? ['G4'] : [],
                evidence_source: source,
              },
            ],
          },
          ...catalog.suites.slice(1),
        ],
      })
      expect(decoded.suites[0]?.methods[0]?.evidence_source).toBe(source)
    }

    const suites = catalog.suites.map((suite, index) =>
      index === 0
        ? {
            ...suite,
            methods: [
              {
                ...suite.methods[0],
                evidence_source: 'unknown_source',
              },
            ],
          }
        : suite,
    )
    expect(() => decodeEvaluationCatalog({ ...catalog, suites })).toThrow(
      /catalog response is incomplete/i,
    )
  })

  it('validates campaign protocol shape without replaying server admission semantics', () => {
    const catalog = evaluationCatalogFixture()
    const campaignProtocol = {
      schema_version: EVALUATION_CAMPAIGN_COHORT_SCHEMA_VERSION,
      minimum_cases: 23,
    } as const
    const campaignSuite: EvaluationCatalogSuite = {
      ...catalog.suites[1],
      executors: { replay: 'mom-cohort-replay.v1', live: 'live-runtime.v1' },
      track_ids: ['routing', 'model_pool', 'joint'],
      modes: ['replay', 'live'],
      evidence_level: 'E0',
      case_count: 80,
      campaign_protocol: campaignProtocol,
      methods: [
        {
          id: 'routing.live.v1',
          track_id: 'routing',
          qualified_gate_ids: [],
          evidence_source: 'live_runtime',
          status: 'configured',
        },
        {
          id: 'model-pool.live.v1',
          track_id: 'model_pool',
          qualified_gate_ids: [],
          evidence_source: 'live_runtime',
          status: 'configured',
        },
        {
          id: 'joint.live.v1',
          track_id: 'joint',
          qualified_gate_ids: [],
          evidence_source: 'live_runtime',
          status: 'configured',
        },
      ],
    }
    const withCampaignSuite = (suite: unknown) => ({
      ...catalog,
      suites: catalog.suites.map((item) => (item.id === campaignSuite.id ? suite : item)),
    })
    expect(decodeEvaluationCatalog(withCampaignSuite(campaignSuite)).suites[1]).toMatchObject({
      campaign_protocol: {
        schema_version: EVALUATION_CAMPAIGN_COHORT_SCHEMA_VERSION,
        minimum_cases: 23,
      },
    })
    const serverConfiguredProtocolSuite = {
      ...campaignSuite,
      executors: { live: 'next-campaign-executor.v2' },
      track_ids: ['agentic'],
      modes: ['live'],
      evidence_level: 'E4',
      methods: [
        {
          ...campaignSuite.methods[0],
          id: 'agentic.next-campaign.v2',
          track_id: 'agentic',
          evidence_source: 'live_runtime',
        },
      ],
    }
    expect(
      decodeEvaluationCatalog(withCampaignSuite(serverConfiguredProtocolSuite)).suites[1],
    ).toMatchObject({
      executors: { live: 'next-campaign-executor.v2' },
      track_ids: ['agentic'],
      evidence_level: 'E4',
      campaign_protocol: campaignProtocol,
    })

    for (const suite of [
      {
        ...campaignSuite,
        campaign_protocol: { ...campaignProtocol, minimum_cases: 0 },
      },
      {
        ...campaignSuite,
        campaign_protocol: { ...campaignProtocol, minimum_cases: 81 },
      },
      {
        ...campaignSuite,
        campaign_protocol: {
          ...campaignProtocol,
          schema_version: 'evaluation-campaign-cohort.v0',
        },
      },
      { ...campaignSuite, case_count: undefined },
      { ...campaignSuite, campaign_protocol: { ...campaignProtocol, extra: true } },
      { ...campaignSuite, campaign_protocol: undefined },
    ]) {
      expect(() => decodeEvaluationCatalog(withCampaignSuite(suite))).toThrow(
        /catalog response is incomplete/i,
      )
    }

    for (const legacySuite of [
      { ...campaignSuite, campaign_eligible: true },
      { ...campaignSuite, campaign_minimum_cases: 23 },
    ]) {
      expect(() => decodeEvaluationCatalog(withCampaignSuite(legacySuite))).toThrow(
        /catalog response is incomplete/i,
      )
    }
  })

  it('accepts an inspectable zero-arm Mixture only as an unavailable catalog target', () => {
    expect(decodeEvaluationCatalog(catalogWithUnavailableMixture).targets[0]).toMatchObject({
      id: 'mom-unavailable',
      healthy: false,
      track_ids: [],
    })
  })

  it('keeps the unavailable catalog exception narrow and digest-bound', () => {
    const unavailableTarget = catalogWithUnavailableMixture.targets[0]
    const invalidTargets = [
      { ...unavailableTarget, healthy: true },
      { ...unavailableTarget, healthy: undefined },
      { ...unavailableTarget, track_ids: ['routing'] },
      { ...unavailableTarget, kind: 'provider-runtime', mixture: undefined },
      {
        ...unavailableTarget,
        mixture: {
          ...unavailableMixture,
          routing_recipe_plan: {
            ...unavailableMixture.routing_recipe_plan,
            target_snapshot_digest: `sha256:${'a'.repeat(64)}`,
          },
        },
      },
      {
        ...unavailableTarget,
        mixture: {
          ...unavailableMixture,
          routing_recipe_plan: {
            ...unavailableMixture.routing_recipe_plan,
            plan_digest: `sha256:${'b'.repeat(64)}`,
          },
        },
      },
    ]
    for (const target of invalidTargets) {
      expect(() =>
        decodeEvaluationCatalog({ ...catalogWithUnavailableMixture, targets: [target] }),
      ).toThrow(/catalog response is incomplete/i)
    }
  })

  it('keeps zero-arm Mixtures outside the executable run contract', () => {
    expect(() =>
      decodeEvaluationRun({
        ...run,
        mode: 'live',
        target_id: 'mom-unavailable',
        mixture: unavailableMixture,
      }),
    ).toThrow(/run response is incomplete/i)
  })

  it('requires one explicit executor per advertised suite mode', () => {
    const installedSuite: EvaluationCatalogSuite = {
      id: 'installed-routing',
      executors: {
        replay: 'normalized-suite-replay.v1',
        live: 'normalized-suite-live.v1',
      },
      name: 'Installed routing',
      description: '',
      track_ids: ['routing'],
      modes: ['replay', 'live'],
      evidence_level: 'E0',
      revision: 'sha256:revision',
      tags: [],
      methods: [
        {
          id: 'routing.normalized-replay-live.v1',
          track_id: 'routing',
          qualified_gate_ids: [],
          evidence_source: 'normalized_import',
          status: 'configured',
        },
      ],
    }
    const catalog = evaluationCatalogFixture({
      change_profiles: [
        {
          id: 'recipe',
          name: 'Recipe',
          description: '',
          campaign_slots: canonicalCampaignSlots,
        },
      ],
      suites: [installedSuite],
      targets: [
        {
          id: 'benchmark-source',
          name: 'Benchmark source',
          description: '',
          kind: 'benchmark-source',
          track_ids: ['routing'],
          modes: ['replay'],
          accepted_executors: { replay: ['normalized-suite-replay.v1'] },
        },
      ],
    })

    expect(
      decodeEvaluationCatalog(catalog).suites.find((suite) => suite.id === installedSuite.id)
        ?.executors,
    ).toEqual({
      replay: 'normalized-suite-replay.v1',
      live: 'normalized-suite-live.v1',
    })
    const brokeredSuite: EvaluationCatalogSuite = {
      ...installedSuite,
      methods: [
        {
          id: 'routing.declared-shift-live.v1',
          track_id: 'routing',
          qualified_gate_ids: ['G4'],
          evidence_source: 'server_brokered_live',
          status: 'configured',
        },
      ],
    }
    const brokeredCatalog = {
      ...catalog,
      suites: [brokeredSuite],
    }
    expect(
      decodeEvaluationCatalog(brokeredCatalog).suites.find((suite) => suite.id === brokeredSuite.id)
        ?.methods[0],
    ).toMatchObject({ evidence_source: 'server_brokered_live', qualified_gate_ids: ['G4'] })
    expect(
      decodeEvaluationCatalog({
        ...brokeredCatalog,
        suites: [
          {
            ...brokeredSuite,
            methods: [{ ...brokeredSuite.methods[0], qualified_gate_ids: ['G3'] }],
          },
        ],
      }).suites[0].methods[0],
    ).toMatchObject({ qualified_gate_ids: ['G3'] })
    for (const method of [{ ...brokeredSuite.methods[0], status: 'qualified' }]) {
      expect(() =>
        decodeEvaluationCatalog({
          ...brokeredCatalog,
          suites: [{ ...brokeredSuite, methods: [method] }],
        }),
      ).toThrow(/catalog response is incomplete/i)
    }
    expect(() =>
      decodeEvaluationCatalog({
        ...catalog,
        suites: [{ ...installedSuite, executors: { replay: 'normalized-suite-replay.v1' } }],
      }),
    ).toThrow(/catalog response is incomplete/i)
    expect(() =>
      decodeEvaluationCatalog({
        ...catalog,
        suites: [
          {
            ...installedSuite,
            executor_id: 'retired-universal-executor',
          },
        ],
      }),
    ).toThrow(/catalog response is incomplete/i)
  })

  it('accepts server-owned profile and slot semantics while rejecting broken references', () => {
    const serverSlots = [
      {
        gate_id: 'G4',
        name: 'Server-defined release check',
        description: 'A future release check owned by the evaluation service.',
        disposition: 'required',
        binding_kind: 'run',
        track_id: 'multimodal',
        mode: 'live',
        minimum_evidence_level: 'E4',
        accepted_executor_ids: ['next-campaign-executor.v2'],
      },
    ] satisfies EvaluationCatalogCampaignSlot[]
    const catalog = evaluationCatalogFixture({
      change_profiles: [
        {
          id: 'future_multimodal_rollout',
          name: 'Future multimodal rollout',
          description: '',
          campaign_slots: serverSlots,
        },
      ],
      suites: [
        {
          id: 'future-multimodal-suite',
          executors: { live: 'next-campaign-executor.v2' },
          name: 'Future multimodal suite',
          description: '',
          track_ids: ['multimodal'],
          modes: ['live'],
          evidence_level: 'E4',
          revision: 'future-multimodal-suite.v1',
          tags: [],
          methods: [
            {
              id: 'multimodal.future-live.v1',
              track_id: 'multimodal',
              qualified_gate_ids: ['G4'],
              evidence_source: 'live_runtime',
              status: 'configured',
            },
          ],
        },
      ],
    })
    expect(decodeEvaluationCatalog(catalog).change_profiles[0]?.campaign_slots[0]).toMatchObject({
      gate_id: 'G4',
      track_id: 'multimodal',
      minimum_evidence_level: 'E4',
    })

    expect(() =>
      decodeEvaluationCatalog({
        ...catalog,
        suites: [
          {
            ...catalog.suites[0],
            methods: [{ ...catalog.suites[0].methods[0], qualified_gate_ids: ['G404'] }],
          },
        ],
      }),
    ).toThrow(/catalog response is incomplete/i)
    expect(() =>
      decodeEvaluationCatalog({
        ...catalog,
        change_profiles: [
          {
            ...catalog.change_profiles[0],
            campaign_slots: [serverSlots[0], { ...serverSlots[0] }],
          },
        ],
      }),
    ).toThrow(/catalog response is incomplete/i)
    for (const campaignSlot of [
      { ...serverSlots[0], track_id: undefined },
      { ...serverSlots[0], mode: undefined },
      { ...serverSlots[0], accepted_executor_ids: [] },
    ]) {
      expect(() =>
        decodeEvaluationCatalog({
          ...catalog,
          change_profiles: [{ ...catalog.change_profiles[0], campaign_slots: [campaignSlot] }],
        }),
      ).toThrow(/catalog response is incomplete/i)
    }
    expect(() => decodeEvaluationCatalog({ ...catalog, change_profiles: [] })).toThrow(
      /catalog response is incomplete/i,
    )
  })

  it('accepts only canonical run identities and exact run fields', () => {
    expect(isCanonicalEvaluationRunID(RUN_ID)).toBe(true)
    expect(isCanonicalEvaluationRunID('run-1')).toBe(false)
    expect(decodeEvaluationRun(run, RUN_ID)).toEqual(run)
    expect(
      decodeEvaluationRun(
        {
          ...run,
          status: 'sealing',
          started_at: '2026-08-30T00:00:01Z',
          completed_at: undefined,
          progress: {
            percent: 100,
            completed: 1,
            total: 1,
            message: 'Sealing evaluation evidence',
          },
        },
        RUN_ID,
      ).status,
    ).toBe('sealing')
    expect(() => decodeEvaluationRun({ ...run, retired_status: 'ready' }, RUN_ID)).toThrow(
      /run response is incomplete/i,
    )
  })

  it('requires live Mixture snapshots without inferring replay semantics from suite IDs', () => {
    const cohortReplay = {
      ...run,
      target_id: mixture.id,
      suite_ids: ['live-mom-core'],
      mixture,
    }
    const liveRun = { ...cohortReplay, mode: 'live' }
    expect(decodeEvaluationRun(liveRun, RUN_ID)).toEqual(liveRun)
    expect(
      decodeEvaluationRunLedger({
        schema_version: 'evaluation.v1',
        runs: [cohortReplay],
        total_runs: 1,
        ledger_complete: true,
        warning_count: 0,
        warnings: [],
      }).runs,
    ).toEqual([cohortReplay])

    expect(decodeEvaluationRun({ ...cohortReplay, mixture: undefined }, RUN_ID)).toEqual({
      ...cohortReplay,
      mixture: undefined,
    })
    expect(() => decodeEvaluationRun({ ...liveRun, mixture: undefined }, RUN_ID)).toThrow(
      /live run does not bind its Mixture snapshot/i,
    )
    expect(decodeEvaluationRun({ ...run, target_id: mixture.id, mixture }, RUN_ID)).toEqual({
      ...run,
      target_id: mixture.id,
      mixture,
    })
    expect(
      decodeEvaluationRun(
        { ...cohortReplay, suite_ids: ['live-mom-core', 'evaluation-smoke'] },
        RUN_ID,
      ),
    ).toEqual({ ...cohortReplay, suite_ids: ['live-mom-core', 'evaluation-smoke'] })
  })

  it('requires live, role-correct lineage for every controlled-pair ledger member', () => {
    const pairID = '22222222-2222-4222-8222-222222222222'
    const baselineID = '33333333-3333-4333-8333-333333333333'
    const candidateID = '44444444-4444-4444-8444-444444444444'
    const liveMember = {
      ...run,
      id: baselineID,
      client_request_id: baselineID,
      mode: 'live' as const,
      target_id: mixture.id,
      suite_ids: ['live-mom-core'],
      concurrency: 2,
      mixture,
      controlled_pair: { pair_id: pairID, role: 'baseline' as const },
    }
    const candidate = {
      ...liveMember,
      id: candidateID,
      client_request_id: candidateID,
      baseline_run_id: baselineID,
      controlled_pair: { pair_id: pairID, role: 'candidate' as const },
    }

    expect(decodeEvaluationRun(liveMember, baselineID)).toEqual(liveMember)
    expect(decodeEvaluationRun(candidate, candidateID)).toEqual(candidate)
    expect(() => decodeEvaluationRun({ ...liveMember, mode: 'replay' }, baselineID)).toThrow(
      /only valid for live execution/i,
    )
    expect(() =>
      decodeEvaluationRun({ ...liveMember, baseline_run_id: candidateID }, baselineID),
    ).toThrow(/baseline member cannot declare a baseline run/i)
    expect(() =>
      decodeEvaluationRun({ ...candidate, baseline_run_id: undefined }, candidateID),
    ).toThrow(/candidate member must reference a distinct canonical baseline run/i)
    expect(() =>
      decodeEvaluationRun({ ...candidate, baseline_run_id: candidateID }, candidateID),
    ).toThrow(/candidate member must reference a distinct canonical baseline run/i)

    expect(() =>
      decodeEvaluationRunLedger({
        schema_version: 'evaluation.v1',
        runs: [{ ...liveMember, mode: 'replay' }],
        total_runs: 1,
        ledger_complete: true,
        warning_count: 0,
        warnings: [],
      }),
    ).toThrow(/ledger response is invalid or incomplete/i)
  })

  it('keeps ledger warning evidence identities opaque and non-navigable', () => {
    const ledger = {
      schema_version: 'evaluation.v1',
      runs: [run],
      total_runs: 1,
      ledger_complete: false,
      warning_count: 1,
      warnings: [
        {
          code: 'unexpected_entry',
          evidence_id: `sha256:${'a'.repeat(64)}`,
          evidence_file: '',
          message: 'Unexpected durable entry was quarantined.',
        },
      ],
    }

    expect(decodeEvaluationRunLedger(ledger).warnings[0]?.evidence_id).toMatch(/^sha256:/)
    expect(() =>
      decodeEvaluationRunLedger({
        ...ledger,
        warnings: [{ ...ledger.warnings[0], run_id: 'retired-run-identity' }],
      }),
    ).toThrow(/ledger response is invalid or incomplete/i)
  })

  it('preserves the typed record count for track evidence events', () => {
    const event = decodeEvaluationRunEvent(
      {
        id: '2',
        run_id: RUN_ID,
        type: 'track',
        timestamp: '2026-08-30T00:00:30Z',
        message: 'Evaluation track evidence collected',
        track_id: 'routing',
        progress: {
          percent: 100,
          completed: 1,
          total: 1,
          current_track_id: 'routing',
          message: 'Evaluation track evidence collected',
        },
        payload: { record_count: 17 },
      },
      run,
    )

    expect(event.type).toBe('track')
    if (event.type !== 'track') throw new Error('Expected a typed track event.')
    expect(event.payload.record_count).toBe(17)
  })

  it('rejects unknown or event-mismatched durable SSE payloads', () => {
    const base = {
      id: '3',
      run_id: RUN_ID,
      timestamp: '2026-08-30T00:00:45Z',
      message: 'Evaluation progress updated',
    }
    const trackProgress = {
      percent: 100,
      completed: 1,
      total: 1,
      current_track_id: 'routing',
      message: 'Evaluation track evidence collected',
    }

    expect(() =>
      decodeEvaluationRunEvent(
        {
          ...base,
          type: 'track',
          track_id: 'routing',
          progress: trackProgress,
          payload: { record_count: 1, retired_detail: 'not public' },
        },
        run,
      ),
    ).toThrow(/invalid event/i)
    expect(() =>
      decodeEvaluationRunEvent(
        {
          ...base,
          type: 'track',
          track_id: 'routing',
          progress: trackProgress,
          payload: { record_count: -1 },
        },
        run,
      ),
    ).toThrow(/invalid event/i)
    expect(() =>
      decodeEvaluationRunEvent(
        {
          ...base,
          type: 'track',
          track_id: 'routing',
          progress: { ...trackProgress, current_track_id: 'capacity' },
          payload: { record_count: 1 },
        },
        run,
      ),
    ).toThrow(/invalid event/i)
    expect(() =>
      decodeEvaluationRunEvent(
        {
          ...base,
          type: 'track',
          track_id: 'routing',
          progress: trackProgress,
          payload: { verdict: 'pass' },
        },
        run,
      ),
    ).toThrow(/invalid event/i)
    expect(() =>
      decodeEvaluationRunEvent({ ...base, type: 'progress', payload: { record_count: 1 } }, run),
    ).toThrow(/invalid event/i)
    expect(() =>
      decodeEvaluationRunEvent({ ...base, type: 'completed', payload: { verdict: 'pass' } }, run),
    ).toThrow(/invalid event/i)
  })

  it('enforces durable event identity, message, and run-bound progress semantics', () => {
    const base = {
      id: '4',
      run_id: RUN_ID,
      type: 'progress' as const,
      timestamp: '2026-08-30T00:00:45Z',
      message: 'Evaluation progress updated',
      progress: { percent: 50, completed: 0, total: 1 },
    }

    expect(decodeEvaluationRunEvent(base, run)).toEqual(base)
    for (const id of ['', '0', '01', 'event-4', '18446744073709551616']) {
      expect(() => decodeEvaluationRunEvent({ ...base, id }, run)).toThrow(/invalid event/i)
    }
    for (const message of ['', ' not trimmed', 'not trimmed ', '界'.repeat(171)]) {
      expect(() => decodeEvaluationRunEvent({ ...base, message }, run)).toThrow(/invalid event/i)
    }
    expect(() =>
      decodeEvaluationRunEvent({ ...base, progress: { ...base.progress, total: 2 } }, run),
    ).toThrow(/invalid event/i)
    expect(() =>
      decodeEvaluationRunEvent(
        {
          ...base,
          progress: { ...base.progress, message: ' malformed ' },
        },
        run,
      ),
    ).toThrow(/invalid event/i)
  })

  it('requires server-owned terminal progress and exact completed semantics', () => {
    const base = {
      id: '5',
      run_id: RUN_ID,
      timestamp: '2026-08-30T00:01:00Z',
      message: 'Evaluation completed',
    }
    const completed = {
      ...base,
      type: 'completed' as const,
      progress: { percent: 100, completed: 1, total: 1, message: 'Evaluation completed' },
    }

    expect(decodeEvaluationRunEvent(completed, run)).toEqual(completed)
    expect(() =>
      decodeEvaluationRunEvent(
        { ...completed, progress: { ...completed.progress, percent: 99 } },
        run,
      ),
    ).toThrow(/invalid event/i)
    expect(() => decodeEvaluationRunEvent({ ...base, type: 'failed' }, run)).toThrow(
      /invalid event/i,
    )
    expect(() => decodeEvaluationRunEvent({ ...base, type: 'cancelled' }, run)).toThrow(
      /invalid event/i,
    )
  })
})
