import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import catalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { TooltipProvider } from '../ui'

vi.mock('../hooks/useBuiltInModelCatalog')

beforeEach(() => {
  vi.mocked(useBuiltInModelCatalog).mockReturnValue({
    catalog: catalog as unknown as BuiltInModelCatalog,
    ready: true,
    loading: false,
    error: null,
    source: 'bundled',
    retry: vi.fn(),
  })
})

import ConfigPageModelsSection from './ConfigPageModelsSection'
import type { ConfigPageModelsSectionProps } from './configPageModelsSectionTypes'

const models = Array.from({ length: 30 }, (_, index) => ({
  name: `custom-model-${String(index).padStart(2, '0')}`,
  endpoints:
    index % 2 === 0
      ? [{ name: 'primary', endpoint: 'vllm:8000', protocol: 'http' as const, weight: 1 }]
      : [],
  backend_refs:
    index % 2 === 0 ? [{ name: 'primary', endpoint: 'vllm:8000', provider: 'vllm' }] : [],
  pricing: index === 0 ? { currency: 'USD', prompt_per_1m: 3.5, completion_per_1m: 7 } : undefined,
}))
const props: ConfigPageModelsSectionProps = {
  config: {
    providers: {
      defaults: { model: models[0].name },
      models: models.map(({ name }) => ({ name })),
    },
  },
  isReadonly: false,
  canVerifyModels: true,
  models,
  defaultModel: models[0].name,
  reasoningFamilies: { 'test-family': { type: 'reasoning_effort', parameter: 'reasoning_effort' } },
  modelsSearch: '',
  onModelsSearchChange: () => {},
  expandedModels: new Set(),
  onExpandedModelsChange: () => {},
  saveConfig: async () => {},
  openEditModal: () => {},
  openViewModal: () => {},
  listInputToArray: (input) => input.split(','),
}

const render = (overrides: Partial<ConfigPageModelsSectionProps> = {}, path = '/models/backends') =>
  renderToStaticMarkup(
    <MemoryRouter initialEntries={[path]}>
      <TooltipProvider>
        <ConfigPageModelsSection {...props} {...overrides} />
      </TooltipProvider>
    </MemoryRouter>,
  )

describe('Backend models page', () => {
  it('pages the inventory at 25 rows with one-line prices, kinds and decision counts', () => {
    const html = render()
    expect(html).toContain('Backend models')
    expect(html).toContain('custom-model-24')
    expect(html).not.toContain('custom-model-25')
    expect(html).toContain('1–25 of 30 models')
    expect(html).toContain('aria-label="Actions for custom-model-00"')
    expect(html).toContain('Price / 1M')
    expect(html).toMatch(/\$3\.50 <span[^>]*>\/<\/span> \$7\.00/)
    expect(html).not.toContain('out $7.00')
    expect(html).toContain('>Chat</span>')
    expect(html).toContain('>Decisions<')
  })

  it('checks health on its own instead of offering a check on every row', () => {
    const html = render()
    expect(html).toContain('>Health<')
    expect(html).toContain('aria-label="Checking health"')
    expect(html).not.toContain('with a real inference query')
    expect(html).not.toContain('>Live<')
    for (const label of ['Connected', 'Healthy', 'Median p95', 'Spend · 24h']) {
      expect(html).toContain(label)
    }
    expect(html).toContain('15 without a backend')
  })

  it('selects models from their logo instead of a checkbox column, protecting the default', () => {
    const html = render()
    const checkbox = (name: string) =>
      html.match(new RegExp(`<button[^>]*aria-label="Select model ${name}"[^>]*>`))?.[0] ?? ''
    expect(checkbox('custom-model-00')).toContain('disabled=""')
    expect(checkbox('custom-model-01')).not.toBe('')
    expect(checkbox('custom-model-01')).not.toContain('disabled=""')
    expect(html).not.toContain('Select all models on this page')
    expect(html).not.toContain('role="tablist"')
  })

  it('hides every write affordance for read-only users', () => {
    const html = render({ isReadonly: true, canVerifyModels: false })
    expect(html).not.toContain('Connect model')
    expect(html).not.toContain('Select model custom-model-01')
    expect(html).toContain('aria-label="Actions for custom-model-00"')
    expect(html).toContain('aria-label="More model actions"')
  })

  it('keeps the model list on screen when an old link asks for the families tab', () => {
    const html = render({}, '/models/backends?tab=families')
    expect(html).toContain('custom-model-00')
    expect(html).not.toContain('role="tablist"')
  })
})
