import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import generatedCatalog from '../generated/modelCatalog.json'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { EvaluationCard, ModelAccess, ModelDetail, VirtualPool } from './ModelHubDetail'
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
