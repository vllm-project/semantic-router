import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import useBuiltInModelCatalog from '../hooks/useBuiltInModelCatalog'
import catalog from '../modelCatalogDocument'
import type { BuiltInModelCatalog } from '../types/modelCatalog'

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

const models = Array.from({ length: 7 }, (_, index) => ({
  name: `custom-model-${index}`,
  endpoints: [],
}))
const props: ConfigPageModelsSectionProps = {
  config: { providers: { models: models.map(({ name }) => ({ name })) } },
  isPythonCLI: true,
  isReadonly: true,
  canVerifyModels: false,
  models,
  defaultModel: models[0].name,
  reasoningFamilies: Object.fromEntries(
    models.map((_, index) => [
      `test-family-${index}`,
      { type: 'reasoning_effort', parameter: 'reasoning_effort' },
    ]),
  ),
  modelsSearch: '',
  onModelsSearchChange: () => {},
  expandedModels: new Set(),
  onExpandedModelsChange: () => {},
  saveConfig: async () => {},
  openEditModal: () => {},
  openViewModal: () => {},
  listInputToArray: (input) => input.split(','),
}

describe('Models page inventories', () => {
  it('keeps model forms usable while built-in evaluation evidence is still loading', () => {
    vi.mocked(useBuiltInModelCatalog).mockReturnValue({
      catalog: catalog as unknown as BuiltInModelCatalog,
      ready: false,
      loading: true,
      error: null,
      source: 'bundled',
      retry: vi.fn(),
    })
    const html = renderToStaticMarkup(<ConfigPageModelsSection {...props} />)
    expect(html).toContain('custom-model-4')
    expect(html).toContain('Loading model evaluations')
    expect(html).not.toContain('0 built-in · 0 configured')
  })

  it('defaults all three lists to five rows and keeps no-evidence models viewable', () => {
    const html = renderToStaticMarkup(<ConfigPageModelsSection {...props} />)
    expect(html.match(/value="5" selected=""/g)).toHaveLength(3)
    expect(html).toContain('custom-model-4')
    expect(html).not.toContain('custom-model-5')
    expect(html).toContain('test-family-4')
    expect(html).not.toContain('test-family-5')
    expect(html).toContain('0 built-in · 0 configured')
    expect(html).toContain('Evaluation Evidence')
    expect(html).not.toContain('No operator evaluation records configured')
  })
})
