import { createElement, type ComponentProps } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { EvaluationCatalog, EvaluationCatalogCampaignSlot } from '../../types/evaluationPlane'
import EvaluationCampaign from './EvaluationCampaign'

const OPTIONAL_CHECK_NAMES = {
  G4: 'Shift robustness',
  G6: 'Fault recovery',
  G7: 'Cost, latency, and capacity',
  G8: 'Canary safety',
  G9: 'Online preference',
} as const

const catalog: EvaluationCatalog = {
  schema_version: 'evaluation.v1',
  gate_contract_version: 'evaluation-release-gates.v2',
  generated_at: '2026-08-30T00:00:00Z',
  change_profiles: [
    {
      id: 'recipe',
      name: 'Routing recipe',
      description: 'A recipe-only promotion boundary.',
      campaign_slots: (
        [
          {
            gate_id: 'G2',
            name: 'Hard policy',
            description: 'Policy evidence.',
            disposition: 'advisory',
            binding_kind: 'run',
            track_id: 'safety',
            mode: 'live',
            minimum_evidence_level: 'E0',
            accepted_executor_ids: ['live-runtime.v1'],
          },
          {
            gate_id: 'G3',
            name: 'Controlled value comparison',
            description: 'Controlled paired value evidence.',
            disposition: 'required',
            binding_kind: 'controlled_pair',
            mode: 'live',
            track_id: 'joint',
            minimum_evidence_level: 'E4',
            accepted_executor_ids: ['live-runtime.v1'],
          },
          ...(['G4', 'G6', 'G7', 'G8', 'G9'] as const).map((gate_id) => ({
            gate_id,
            name: OPTIONAL_CHECK_NAMES[gate_id],
            description: 'Optional evidence.',
            disposition: 'advisory' as const,
            binding_kind: 'run' as const,
            track_id: 'joint' as const,
            mode: 'live' as const,
            minimum_evidence_level: 'E0' as const,
            accepted_executor_ids: ['live-runtime.v1'],
          })),
          {
            gate_id: 'G5',
            name: 'Live fidelity',
            description: 'Reference and live fidelity evidence.',
            disposition: 'advisory',
            binding_kind: 'fidelity_pair',
            track_id: 'joint',
            mode: 'live',
            minimum_evidence_level: 'E5',
            accepted_executor_ids: ['normalized-suite-live.v1', 'live-runtime.v1'],
          },
        ] as EvaluationCatalogCampaignSlot[]
      ).sort((left, right) => left.gate_id.localeCompare(right.gate_id)),
    },
  ],
  tracks: [],
  suites: [],
  targets: [],
}

function visibleText(markup: string): string {
  return markup
    .replace(/<[^>]*>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

const campaignProps: ComponentProps<typeof EvaluationCampaign> = {
  catalog,
  runs: [],
  totalRuns: 0,
  runLedgerAvailable: true,
  runLedgerComplete: true,
  allRunsLoaded: true,
  loadingAllRuns: false,
  canCreate: true,
  createPending: false,
  createError: null,
  campaign: null,
  campaignLoading: false,
  campaignError: null,
  activeControlledPairID: null,
  activeControlledPairProfileID: null,
  onLoadAllRuns: () => undefined,
  onRefreshRuns: () => true,
  onControlledPairIdentityChange: () => undefined,
  onCreate: async () => null,
  onClearCreateError: () => undefined,
  onRetryCampaign: () => undefined,
  onClearCampaign: () => undefined,
}

function renderCampaign(overrides: Partial<ComponentProps<typeof EvaluationCampaign>> = {}) {
  return renderToStaticMarkup(createElement(EvaluationCampaign, { ...campaignProps, ...overrides }))
}

function expectTechnicalOnly(markup: string, rawText: string) {
  const rawIndex = markup.indexOf(rawText)
  const detailsIndex = markup.lastIndexOf('<details', rawIndex)
  expect(rawIndex).toBeGreaterThan(-1)
  expect(detailsIndex).toBeGreaterThan(-1)
  expect(markup.slice(detailsIndex, rawIndex)).toContain('data-evaluation-technical-details="true"')
}

describe('EvaluationCampaignBuilder workspace', () => {
  it('keeps expert evidence mapping progressively disclosed behind a readiness summary', () => {
    const markup = renderCampaign()

    expect(markup).toContain('aria-label="Release decision inputs"')
    expect(markup).toContain('aria-label="Release readiness summary"')
    expect(markup).toContain('Review evaluation inputs')
    expect(markup).toContain('1 required check still needs a run')
    expect(markup).not.toContain('<details open=""')
    expect(markup).toContain('<table')
    expect(markup).toContain('<th scope="col">Release check</th>')
    expect(markup).toContain('<td data-label="Status">')
    expect(markup).toContain('<td data-label="Selected runs">')
    expect(markup).not.toContain('aria-label="Release decision inputs" tabindex="0"')
    expect(markup).toContain('Controlled value comparison')
    expect(markup).toContain('Optional')
    expect(markup).toContain('<details')
    expect(markup).toContain('Decision notes')
    expect(markup).toContain('No completed live Mixture run is available')
    expect(markup).not.toContain('Only five UUIDs cross the wire')
    expect(markup.match(/type="submit"/g)).toHaveLength(1)
    expect(markup).toContain('Create release decision')
    expect(markup).not.toContain('Prepare a production decision')

    const text = visibleText(markup)
    expect(text).not.toMatch(/\bE[0-5]\b/)
    expect(text).not.toMatch(/\bG[0-9]\b/)
    expect(text).not.toContain(catalog.schema_version)
    expect(text).not.toContain(catalog.gate_contract_version)
  })

  it('keeps decision read and create failures behind product error boundaries', () => {
    const readError = 'backend://campaign-read G8 private-stack'
    const readMarkup = renderCampaign({ campaignError: readError })
    expect(readMarkup).toContain('Release decision could not be loaded')
    expectTechnicalOnly(readMarkup, readError)

    const createError = 'worker://campaign-create G3 private-ledger-response'
    const createMarkup = renderCampaign({ createError })
    expect(createMarkup).toContain('Release decision could not be created')
    expectTechnicalOnly(createMarkup, createError)
    expect(createMarkup).not.toContain('<details open')
  })
})
