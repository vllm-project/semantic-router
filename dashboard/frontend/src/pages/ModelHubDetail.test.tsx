import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import generatedCatalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { EvaluationCard, ModelAccess, ModelDetail, VirtualPool } from './ModelHubDetail'
import { providerProtocolOperations } from './modelHubProviderOperations'
import { modelHubRows, type ModelHubFilters } from './modelHubSupport'

const catalog = generatedCatalog as unknown as BuiltInModelCatalog
const virtualFilters: ModelHubFilters = {
  query: '',
  kind: 'virtual',
  distribution: 'all',
  lifecycle: 'supported',
  publisher: 'all',
  provider: 'all',
  capability: 'all',
  sort: 'name',
}

describe('model hub detail', () => {
  it('labels published evaluation conditions without inventing a configurable effort', () => {
    const model = catalog.models.find((candidate) => candidate.id === 'ai21/jamba-reasoning-3b')!
    const evaluation = catalog.evaluations.find(
      (candidate) => candidate.model === model.id && candidate.reasoning_effort === 'enabled',
    )!
    const benchmark = catalog.benchmarks.find((candidate) => candidate.id === evaluation.benchmark)
    const markup = renderToStaticMarkup(
      <EvaluationCard model={model} evaluation={evaluation} benchmark={benchmark} />,
    )

    expect(markup).toContain('Reasoning enabled')
    expect(markup).not.toContain('enabled effort')
  })

  it('shows the measurement date before the catalog observation date', () => {
    const model = catalog.models.find((candidate) => candidate.kind === 'physical')!
    const evaluation = catalog.evaluations.find((candidate) => candidate.model === model.id)!
    const benchmark = catalog.benchmarks.find((candidate) => candidate.id === evaluation.benchmark)
    const observedMarkup = renderToStaticMarkup(
      <EvaluationCard
        model={model}
        evaluation={{ ...evaluation, measured_at: undefined, observed_at: '2026-09-06' }}
        benchmark={benchmark}
      />,
    )
    const measuredMarkup = renderToStaticMarkup(
      <EvaluationCard
        model={model}
        evaluation={{ ...evaluation, measured_at: '2026-08-31', observed_at: '2026-09-06' }}
        benchmark={benchmark}
      />,
    )

    expect(observedMarkup).toContain('observed 2026-09-06')
    expect(measuredMarkup).toContain('measured 2026-08-31')
    expect(measuredMarkup).not.toContain('observed 2026-09-06')
  })

  it('renders every virtual-model role and recommended backend candidate', () => {
    const row = modelHubRows(catalog, virtualFilters).find(
      (candidate) => candidate.model.id === 'vllm-sr/mom-v1-blend',
    )
    expect(row).toBeDefined()
    const markup = renderToStaticMarkup(<VirtualPool row={row!} catalog={catalog} />)
    for (const role of row!.model.roles ?? []) {
      expect(markup).toContain(role.name.split('_').join(' '))
      for (const model of role.recommended_pool) expect(markup).toContain(model)
    }
    expect(markup).toContain('Custom model slot')
  })

  it('exposes a linked, keyboard-roving tab pattern', () => {
    const row = modelHubRows(catalog, { ...virtualFilters, kind: 'physical' })[0]
    const markup = renderToStaticMarkup(<ModelDetail row={row} catalog={catalog} />)

    expect(markup).toContain('role="tablist"')
    expect(markup).toContain('role="tab"')
    expect(markup).toContain('aria-controls="model-detail-')
    expect(markup).toContain('tabindex="0"')
    expect(markup).toContain('tabindex="-1"')
    expect(markup).toContain('role="tabpanel"')
    expect(markup).toContain('aria-labelledby="model-detail-')
  })
})

describe('model hub provider detail', () => {
  it('labels the creator-to-serving-channel relationship for each access route', () => {
    const provider = catalog.providers.find((candidate) => candidate.id === 'openrouter')
    const binding = provider?.models?.[0]
    const sourceRow = modelHubRows(catalog, { ...virtualFilters, kind: 'physical' }).find(
      (candidate) => candidate.model.id === binding?.catalog,
    )
    expect(provider).toBeDefined()
    expect(binding).toBeDefined()
    expect(sourceRow).toBeDefined()
    const row = {
      ...sourceRow!,
      providers: sourceRow!.providers.map((entry) =>
        entry.provider.id === 'openrouter'
          ? { ...entry, model: { ...entry.model, relationship: 'gateway' as const } }
          : entry,
      ),
    }

    const markup = renderToStaticMarkup(<ModelAccess row={row} catalog={catalog} />)
    expect(markup).toContain('Gateway')
    expect(markup).toContain(provider!.display_name)
  })

  it.each([
    ['bedrock', '/chat/completions'],
    ['minimax', '/v1/chat/completions'],
  ])(
    'renders only supported operations and the effective path for %s',
    (providerID, expectedPath) => {
      const provider = catalog.providers.find((candidate) => candidate.id === providerID)!
      const binding = provider.models?.[0]
      const sourceRow = modelHubRows(catalog, { ...virtualFilters, kind: 'physical' }).find(
        (candidate) => candidate.model.id === binding?.catalog,
      )
      expect(binding).toBeDefined()
      expect(sourceRow).toBeDefined()

      const row = {
        ...sourceRow!,
        providers: [{ provider, model: binding! }],
      }
      const markup = renderToStaticMarkup(<ModelAccess row={row} catalog={catalog} />)

      expect(markup).toContain('POST')
      expect(markup.match(/>create<\/strong>/g)).toHaveLength(1)
      expect(markup).toContain(expectedPath)
      expect(markup).not.toContain('GET')
      expect(markup).not.toContain('List Models')
      if (providerID === 'bedrock') expect(markup).not.toContain('/v1/chat/completions')
    },
  )

  it.each([
    ['openai', 'openai/responses@1', 'create', '/v1/responses'],
    ['openrouter', 'openai/chat-completions@1', 'create', '/api/v1/chat/completions'],
    ['openrouter', 'openai/responses@1', 'create', '/api/v1/responses'],
    ['gemini', 'openai/chat-completions@1', 'create', '/v1beta/openai/chat/completions'],
    ['gemini', 'openai/chat-completions@1', 'list_models', '/v1beta/openai/models'],
    ['baidu-ai-studio', 'openai/chat-completions@1', 'create', '/llm/lmapi/v3/chat/completions'],
    ['bedrock', 'openai/chat-completions@1', 'create', '/chat/completions'],
    ['minimax', 'openai/chat-completions@1', 'create', '/v1/chat/completions'],
  ])(
    'resolves the Go registry effective path for %s %s#%s',
    (providerID, protocolID, operationID, expectedPath) => {
      const provider = catalog.providers.find((candidate) => candidate.id === providerID)!
      const protocol = catalog.protocols.find((candidate) => candidate.id === protocolID)!
      const operation = providerProtocolOperations(provider, protocol).find(
        (candidate) => candidate.id === operationID,
      )

      expect(operation?.path).toBe(expectedPath)
    },
  )

  it('exposes the compact detail as a labelled modal with an explicit close action', () => {
    const row = modelHubRows(catalog, { ...virtualFilters, kind: 'physical' })[0]
    const markup = renderToStaticMarkup(
      <ModelDetail row={row} catalog={catalog} modal onClose={() => undefined} />,
    )

    expect(markup).toContain('role="dialog"')
    expect(markup).toContain('aria-modal="true"')
    expect(markup).toContain('aria-label="Close model details"')
    expect(markup).toContain('aria-labelledby="model-detail-')
  })
})
