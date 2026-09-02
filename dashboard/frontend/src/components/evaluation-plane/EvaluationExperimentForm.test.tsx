import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationCatalog, EvaluationRun } from '../../types/evaluationPlane'
import { canonicalCampaignSlots } from '../../test/evaluationPlaneApiFixture'
import EvaluationExperimentForm from './EvaluationExperimentForm'
import { buildEvaluationRoutingRecipePlan } from '../../test/evaluationRoutingRecipeFixture'
import {
  compatibleEvaluationSuites,
  exactCohortFromRun,
  minimumCatalogEvidenceClass,
  reconcileEvaluationScope,
  selectedSuiteTracks,
  toggleEvaluationSuite,
} from './evaluationExperiment'
import { baselineCohortIssue, validateEvaluationDraft } from './evaluationExperimentValidation'
import { newEvaluationClientRequestID } from '../../utils/evaluationIdentity'

const catalog: EvaluationCatalog = {
  schema_version: 'evaluation.v1',
  gate_contract_version: 'evaluation-release-gates.v2',
  generated_at: '2026-08-30T00:00:00Z',
  change_profiles: [
    {
      id: 'recipe',
      name: 'Recipe',
      description: 'Recipe change',
      campaign_slots: canonicalCampaignSlots,
    },
  ],
  tracks: [
    {
      id: 'routing',
      name: 'Routing',
      description: 'Routing evidence',
      modes: ['replay'],
      metrics: ['routing.accuracy'],
      evidence_levels: ['E4'],
    },
    {
      id: 'joint',
      name: 'Joint',
      description: 'Joint evidence',
      modes: ['replay'],
      metrics: ['joint.quality'],
      evidence_levels: ['E2', 'E4'],
    },
    {
      id: 'model_pool',
      name: 'Model pool',
      description: 'Pool evidence',
      modes: ['replay'],
      metrics: ['model_pool.oracle_gain'],
      evidence_levels: ['E2'],
    },
  ],
  suites: [
    {
      id: 'routing-suite',
      executors: { replay: 'fixture-replay.v1' },
      name: 'Routing suite',
      description: 'Routing and joint cases',
      track_ids: ['routing', 'joint'],
      modes: ['replay'],
      evidence_level: 'E4',
      revision: 'routing-suite.v1',
      tags: ['fixture'],
      methods: [
        {
          id: 'fixture.routing.v1',
          track_id: 'routing',
          qualified_gate_ids: [],
          evidence_source: 'diagnostic_fixture',
          status: 'configured',
        },
        {
          id: 'fixture.joint.v1',
          track_id: 'joint',
          qualified_gate_ids: [],
          evidence_source: 'diagnostic_fixture',
          status: 'configured',
        },
      ],
    },
    {
      id: 'pool-suite',
      executors: { replay: 'fixture-replay.v1' },
      name: 'Pool suite',
      description: 'Joint and pool cases',
      track_ids: ['joint', 'model_pool'],
      modes: ['replay'],
      evidence_level: 'E2',
      revision: 'pool-suite.v1',
      tags: ['fixture'],
      methods: [
        {
          id: 'fixture.pool.v1',
          track_id: 'model_pool',
          qualified_gate_ids: [],
          evidence_source: 'diagnostic_fixture',
          status: 'configured',
        },
        {
          id: 'fixture.pool-joint.v1',
          track_id: 'joint',
          qualified_gate_ids: [],
          evidence_source: 'diagnostic_fixture',
          status: 'configured',
        },
      ],
    },
    {
      id: 'live-suite',
      executors: { live: 'live-runtime.v1' },
      name: 'Live suite',
      description: 'Not compatible with the replay target',
      track_ids: ['routing'],
      modes: ['live'],
      evidence_level: 'E5',
      revision: 'live-suite.v1',
      tags: [],
      methods: [
        {
          id: 'live.routing.v1',
          track_id: 'routing',
          qualified_gate_ids: [],
          evidence_source: 'live_runtime',
          status: 'configured',
        },
      ],
    },
  ],
  targets: [
    {
      id: 'fixture',
      name: 'Fixture',
      description: 'Replay fixture',
      kind: 'fixture',
      track_ids: ['routing', 'joint', 'model_pool'],
      modes: ['replay'],
      accepted_executors: { replay: ['fixture-replay.v1'] },
      healthy: true,
    },
  ],
}

const baseline: EvaluationRun = {
  schema_version: 'evaluation.v1',
  id: 'baseline-1',
  client_request_id: 'baseline-1',
  name: 'Baseline',
  description: 'Reference cohort',
  status: 'completed',
  mode: 'replay',
  evidence_level: 'E2',
  track_evidence_levels: { routing: 'E2', joint: 'E2', model_pool: 'E2' },
  target_id: 'fixture',
  change_profile: 'recipe',
  suite_ids: ['routing-suite', 'pool-suite'],
  track_ids: ['routing', 'joint', 'model_pool'],
  sample_limit: 100,
  concurrency: 4,
  seed: 42,
  progress: { percent: 100, completed: 3, total: 3 },
  created_at: '2026-08-30T00:00:00Z',
  completed_at: '2026-08-30T00:01:00Z',
}

function visibleText(markup: string): string {
  return markup
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function expectProductLanguage(markup: string): void {
  const text = visibleText(markup)
  expect(text).not.toMatch(/\bE[0-5]\b/)
  expect(text).not.toMatch(/\bG[0-9]\b/)
  expect(text).not.toContain(catalog.schema_version)
  expect(text).not.toContain(catalog.gate_contract_version)
}

describe('evaluation experiment cohort helpers', () => {
  it('uses the least-selected catalog evidence class without promising a run claim', () => {
    expect(minimumCatalogEvidenceClass(catalog, ['routing-suite', 'pool-suite'])).toBe('E2')
    expect(minimumCatalogEvidenceClass(catalog, ['routing-suite'])).toBe('E4')
    expect(minimumCatalogEvidenceClass(catalog, ['missing-suite'])).toBeNull()
  })

  it('creates backend-valid, collision-resistant idempotency tokens', () => {
    const first = newEvaluationClientRequestID()
    const second = newEvaluationClientRequestID()
    expect(first).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
    expect(second).not.toBe(first)
  })

  it('prunes tracks orphaned by removing a suite while preserving shared tracks', () => {
    const next = toggleEvaluationSuite(
      catalog,
      'fixture',
      'replay',
      ['routing-suite', 'pool-suite'],
      ['routing', 'joint', 'model_pool'],
      'routing-suite',
    )
    expect(next).toEqual({
      suiteIDs: ['pool-suite'],
      trackIDs: ['joint', 'model_pool'],
    })
    expect(selectedSuiteTracks(catalog, 'fixture', 'replay', next.suiteIDs)).toEqual([
      'joint',
      'model_pool',
    ])
  })

  it('does not add a suite that the target and mode cannot execute', () => {
    expect(
      compatibleEvaluationSuites(catalog, 'fixture', 'replay').map((suite) => suite.id),
    ).toEqual(['routing-suite', 'pool-suite'])
    expect(toggleEvaluationSuite(catalog, 'fixture', 'replay', [], [], 'live-suite')).toEqual({
      suiteIDs: [],
      trackIDs: [],
    })
  })

  it('offers the executable subset of a suite for a partial-capability target', () => {
    const partialCatalog: EvaluationCatalog = {
      ...catalog,
      targets: [{ ...catalog.targets[0], track_ids: ['routing'] }],
    }

    expect(
      compatibleEvaluationSuites(partialCatalog, 'fixture', 'replay').map((suite) => suite.id),
    ).toEqual(['routing-suite'])
    expect(
      toggleEvaluationSuite(partialCatalog, 'fixture', 'replay', [], [], 'routing-suite'),
    ).toEqual({ suiteIDs: ['routing-suite'], trackIDs: ['routing'] })
  })

  it('keeps one target-approved executor cohort and switches cohorts deliberately', () => {
    const mixedCatalog: EvaluationCatalog = {
      ...catalog,
      suites: [
        ...catalog.suites,
        {
          id: 'normalized-routing',
          executors: { replay: 'normalized-suite-replay.v1' },
          name: 'Normalized routing',
          description: 'Pinned exploratory import; native execution is not attested',
          track_ids: ['routing'],
          modes: ['replay'],
          evidence_level: 'E0',
          revision: 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
          tags: ['normalized'],
          methods: [
            {
              id: 'normalized.routing.v1',
              track_id: 'routing',
              qualified_gate_ids: [],
              evidence_source: 'normalized_import',
              status: 'configured',
            },
          ],
        },
      ],
      targets: [
        {
          ...catalog.targets[0],
          accepted_executors: {
            replay: ['fixture-replay.v1', 'normalized-suite-replay.v1'],
          },
        },
      ],
    }
    expect(
      compatibleEvaluationSuites(mixedCatalog, 'fixture', 'replay').map((suite) => suite.id),
    ).toEqual(['routing-suite', 'pool-suite', 'normalized-routing'])
    expect(
      reconcileEvaluationScope(
        mixedCatalog,
        'fixture',
        'replay',
        ['routing-suite', 'normalized-routing'],
        ['routing'],
      ),
    ).toEqual({ suiteIDs: ['routing-suite'], trackIDs: ['routing'] })
    expect(
      toggleEvaluationSuite(
        mixedCatalog,
        'fixture',
        'replay',
        ['routing-suite'],
        ['routing', 'joint'],
        'normalized-routing',
      ),
    ).toEqual({ suiteIDs: ['normalized-routing'], trackIDs: ['routing'] })
  })

  it('copies all eight exact cohort dimensions and rejects unavailable baselines', () => {
    expect(exactCohortFromRun(baseline)).toEqual({
      mode: 'replay',
      targetID: 'fixture',
      changeProfile: 'recipe',
      suiteIDs: ['routing-suite', 'pool-suite'],
      trackIDs: ['routing', 'joint', 'model_pool'],
      sampleLimit: 100,
      concurrency: 4,
      seed: 42,
    })
    expect(baselineCohortIssue(catalog, baseline)).toBeNull()
    expect(baselineCohortIssue(catalog, { ...baseline, concurrency: 129 })).toContain(
      'parallel request count',
    )
    expect(baselineCohortIssue(catalog, { ...baseline, suite_ids: ['removed-suite'] })).toContain(
      'no longer reproducible',
    )
  })

  it('validates backend byte and numeric bounds and exact baseline matching', () => {
    const validDraft = {
      name: 'Candidate',
      description: 'Comparable change',
      ...exactCohortFromRun(baseline),
      baselineRunID: baseline.id,
    }
    const unpairedDraft = { ...validDraft, baselineRunID: '' }
    expect(validateEvaluationDraft(catalog, [baseline], validDraft)).toBeNull()
    expect(
      validateEvaluationDraft(catalog, [baseline], {
        ...unpairedDraft,
        name: 'x'.repeat(200),
        description: 'x'.repeat(4000),
        concurrency: 128,
        seed: 4294967295,
      }),
    ).toBeNull()
    expect(
      validateEvaluationDraft(catalog, [baseline], {
        ...unpairedDraft,
        description: 'x'.repeat(4001),
      }),
    ).toBe('Description must be at most 4000 bytes.')
    expect(
      validateEvaluationDraft(catalog, [baseline], { ...unpairedDraft, concurrency: 129 }),
    ).toBe('Concurrency must be an integer between 1 and 128.')
    expect(
      validateEvaluationDraft(catalog, [baseline], { ...unpairedDraft, seed: 4294967296 }),
    ).toBe('Repeatability key must be an integer between 0 and 4294967295.')
    expect(
      validateEvaluationDraft(catalog, [baseline], {
        ...unpairedDraft,
        suiteIDs: ['routing-suite'],
        trackIDs: ['model_pool'],
      }),
    ).toBe('The selected benchmarks and evaluation areas do not support this Mixture and run type.')
    expect(validateEvaluationDraft(catalog, [baseline], { ...validDraft, seed: 43 })).toBe(
      'The candidate must use the same comparison setup as the selected baseline.',
    )
    expect(
      validateEvaluationDraft(catalog, [baseline], { ...unpairedDraft, name: '界'.repeat(67) }),
    ).toBe('Experiment name must be at most 200 bytes.')
  })
})

describe('EvaluationExperimentForm contract', () => {
  it('prefers a healthy live Mixture and opens it as a frozen routing, pool, and joint cohort', () => {
    const mixtureBase = {
      id: 'mom-live',
      entrypoint_model: 'vllm-sr/auto',
      aliases: ['smart-model'],
      recipe_name: 'balanced',
      recipe_description: 'Balanced routing.',
      recipe_digest: `sha256:${'1'.repeat(64)}`,
      pool_digest: `sha256:${'2'.repeat(64)}`,
      selector_policy_digest: `sha256:${'4'.repeat(64)}`,
      selector_digest: `sha256:${'5'.repeat(64)}`,
      adaptation_digest: `sha256:${'6'.repeat(64)}`,
      binding_digest: `sha256:${'3'.repeat(64)}`,
      model_arms: [
        {
          id: 'fast',
          model: 'models/fast',
          provider_model_id_digest: `sha256:${'4'.repeat(64)}`,
          input_cost_per_million_tokens_usd: 0.1,
          output_cost_per_million_tokens_usd: 0.2,
        },
        {
          id: 'strong',
          model: 'models/strong',
          provider_model_id_digest: `sha256:${'5'.repeat(64)}`,
          input_cost_per_million_tokens_usd: 0.4,
          output_cost_per_million_tokens_usd: 0.8,
        },
      ],
      support_models: [],
      fallback_arm_id: 'fast',
      decisions: [{ name: 'reasoning', algorithm: 'confidence', arm_ids: ['fast', 'strong'] }],
    }
    const mixture = {
      ...mixtureBase,
      routing_recipe_plan: buildEvaluationRoutingRecipePlan(mixtureBase),
    }
    const liveCatalog: EvaluationCatalog = {
      ...catalog,
      tracks: catalog.tracks.map((track) => ({ ...track, modes: ['replay', 'live'] })),
      suites: [
        {
          ...catalog.suites[0],
          id: 'live-mom-core',
          executors: { replay: 'mom-cohort-replay.v1', live: 'live-runtime.v1' },
          track_ids: ['routing', 'model_pool', 'joint'],
          modes: ['replay', 'live'],
          evidence_level: 'E0',
          case_count: 64,
          campaign_protocol: {
            schema_version: 'evaluation-campaign-cohort.v1',
            minimum_cases: 59,
          },
          revision: 'mom-campaign-cohort-v1',
          tags: ['campaign', 'mom', 'hidden-label', 'paired-live'],
        },
      ],
      targets: [
        {
          id: 'baseline--mom-live',
          name: 'Baseline · vllm-sr/auto',
          description: 'Balanced routing.',
          kind: 'mixture-of-models',
          track_ids: ['routing', 'model_pool', 'joint'],
          modes: ['replay', 'live'],
          accepted_executors: {
            replay: ['mom-cohort-replay.v1'],
            live: ['live-runtime.v1'],
          },
          healthy: true,
          mixture,
        },
        {
          id: 'candidate--mom-live',
          name: 'Candidate · vllm-sr/auto',
          description: 'Balanced routing candidate.',
          kind: 'mixture-of-models',
          track_ids: ['routing', 'model_pool', 'joint'],
          modes: ['replay', 'live'],
          accepted_executors: {
            replay: ['mom-cohort-replay.v1'],
            live: ['live-runtime.v1'],
          },
          healthy: true,
          mixture,
        },
      ],
    }
    const markup = renderToStaticMarkup(
      createElement(EvaluationExperimentForm, {
        catalog: liveCatalog,
        runs: [],
        totalRuns: 0,
        canCreate: true,
        canAutoStart: true,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        pending: false,
        onLoadMoreRuns: () => undefined,
        onSubmit: async () => true,
      }),
    )
    expect(markup).toContain('checked="" value="live"')
    expect(markup).toContain(
      '<option value="baseline--mom-live" selected="">Baseline · vllm-sr/auto</option>',
    )
    expect(markup).toContain(
      '<option value="candidate--mom-live">Candidate · vllm-sr/auto</option>',
    )
    expect(markup).toContain('Selected Mixture-of-Models')
    expect(markup).toContain('vllm-sr/auto')
    expect(markup).toContain('balanced')
    expect(markup).toContain('2 pool models')
    expect(markup).toContain('3 areas')
  })

  it('falls back to the healthy replay diagnostics target when no live Mixture exists', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationExperimentForm, {
        catalog,
        runs: [],
        totalRuns: 0,
        canCreate: true,
        canAutoStart: true,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        pending: false,
        onLoadMoreRuns: () => undefined,
        onSubmit: async () => true,
      }),
    )
    expect(markup).toContain('checked="" value="replay"')
    expect(markup).toContain('<option value="fixture" selected="">Fixture</option>')
  })

  it('renders only the executable suite and selected track for a partial-capability target', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationExperimentForm, {
        catalog: {
          ...catalog,
          targets: [{ ...catalog.targets[0], track_ids: ['routing'] }],
        },
        runs: [],
        totalRuns: 0,
        canCreate: true,
        canAutoStart: true,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        pending: false,
        onLoadMoreRuns: () => undefined,
        onSubmit: async () => true,
      }),
    )

    expect(markup).toContain('Routing suite')
    expect(markup).not.toContain('Pool suite')
    expect(markup).toContain('1 area')
    expect(markup).toContain('Not supported for replay on this target')
    expect(markup).not.toContain('2 areas')
  })

  it('does not substitute replay evidence for a missing Mixture deep link', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationExperimentForm, {
        catalog,
        runs: [],
        totalRuns: 0,
        canCreate: true,
        canAutoStart: true,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        pending: false,
        initialEntrypoint: 'removed-mixture',
        onLoadMoreRuns: () => undefined,
        onSubmit: async () => true,
      }),
    )
    expect(markup).toContain('checked="" value="live"')
    expect(markup).toContain('Requested Mixture is not registered for evaluation')
    expect(markup).toContain('removed-mixture')
    expect(markup).toContain('<option value="" selected="">Select Mixture</option>')
    expect(markup).not.toContain('Replay fixture</small>')
  })

  it('locks the complete form while pending and exposes backend-aligned input bounds', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationExperimentForm, {
        catalog,
        runs: [baseline],
        totalRuns: 1,
        canCreate: true,
        canAutoStart: true,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        pending: true,
        onLoadMoreRuns: () => undefined,
        onSubmit: async () => true,
      }),
    )
    expect(markup).toContain('aria-busy="true"')
    expect(markup).toContain('<fieldset disabled=""')
    expect(markup).toContain('maxLength="200"')
    expect(markup).toContain('maxLength="4000"')
    expect(markup).toContain('max="128"')
    expect(markup).toContain('max="4294967295"')
    expect(markup).toContain('Evaluation scope · Model-pool validation')
    expect(markup).toContain('<details')
    expect(markup).toContain('Review release checks')
    expect(markup.indexOf('Experiment setup')).toBeLessThan(markup.indexOf('Release readiness'))
    expect(markup.indexOf('Release readiness')).toBeLessThan(markup.indexOf('Benchmarks'))
    expect(markup.indexOf('Benchmarks')).toBeLessThan(markup.indexOf('Evaluation areas'))
    expect(markup.indexOf('Evaluation areas')).toBeLessThan(
      markup.indexOf('Budget and reproducibility'),
    )
    expectProductLanguage(markup)
  })

  it('explains when the selected target has no compatible suites or tracks', () => {
    const markup = renderToStaticMarkup(
      createElement(EvaluationExperimentForm, {
        catalog: {
          ...catalog,
          targets: [
            {
              ...catalog.targets[0],
              modes: ['live'],
            },
          ],
        },
        runs: [],
        totalRuns: 0,
        canCreate: true,
        canAutoStart: false,
        runLedgerAvailable: true,
        runLedgerComplete: true,
        hasMoreRuns: false,
        loadingMoreRuns: false,
        pending: false,
        onLoadMoreRuns: () => undefined,
        onSubmit: async () => true,
      }),
    )
    expect(markup).toContain(
      'Select an available Mixture that supports replay, or choose another run type.',
    )
    expect(markup).toContain('Select a compatible benchmark to see the areas it can measure.')
    expect(markup).toContain('Choose benchmarks to set the scope')
    expectProductLanguage(markup)
  })
})
