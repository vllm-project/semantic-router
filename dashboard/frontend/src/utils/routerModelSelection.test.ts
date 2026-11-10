import { describe, expect, it } from 'vitest'
import {
  getRouterModelsEndpoint,
  listRouterModels,
  selectRouterAutoModel,
} from './routerModelSelection'

const defaultRoute = {
  api: 'chat',
  resolution: 'virtual',
  selectable: true,
  default_route: true,
  recipe: 'default',
}
const profile = { api: 'chat', resolution: 'virtual', selectable: true, recipe: 'fast' }
const backend = { api: 'chat', resolution: 'passthrough', selectable: true }

describe('advertised Chat model selection', () => {
  it('uses the effective default entrypoint, including ordinary user aliases', () => {
    for (const id of ['vllm-sr/auto', 'MoM', 'auto', 'my-router']) {
      expect(selectRouterAutoModel({ data: [{ id, routing: defaultRoute }] })).toBe(id)
    }
    expect(
      selectRouterAutoModel({
        data: [
          { id: 'first', routing: defaultRoute },
          { id: 'vllm-sr/auto', routing: defaultRoute },
        ],
      }),
    ).toBe('first')
  })

  it('keeps every advertised selectable alias and backend without inventing an alias', () => {
    const data = [
      { id: 'fast', routing: profile, description: 'Fast responses' },
      { id: 'custom', routing: defaultRoute },
      { id: 'MoM', routing: defaultRoute },
      { id: 'provider/model', routing: backend },
      { id: 'custom', routing: defaultRoute },
      { id: 'hidden', routing: { ...backend, selectable: false } },
    ]
    expect(listRouterModels({ data })).toEqual([
      { id: 'fast', recipe: 'fast', description: 'Fast responses' },
      { id: 'custom', recipe: 'default', description: '' },
      { id: 'MoM', recipe: 'default', description: '' },
      { id: 'provider/model', description: '' },
    ])
    expect(listRouterModels({ data: [] })).toEqual([])
  })

  it('isolates SystemOne names from Chat and requires validated routing metadata', () => {
    const data = [
      { id: 'vllm-sr/auto', routing: { ...defaultRoute, api: 'systemone' } },
      { id: 'MoM', owned_by: 'vllm-semantic-router' },
      { id: 'auto', description: 'Automatic model routing' },
      { id: 'bad-default', routing: { ...backend, default_route: true } },
      { id: 'bad-type', routing: { resolution: 'future', selectable: true } },
    ]
    expect(selectRouterAutoModel({ data })).toBeNull()
    expect(listRouterModels({ data })).toEqual([])
    expect(selectRouterAutoModel({ data: 'invalid' })).toBeNull()
    expect(selectRouterAutoModel({ data: [{ id: 'backend/auto', routing: backend }] })).toBeNull()
  })

  it('derives the discovery endpoint from local and absolute Chat endpoints', () => {
    expect(getRouterModelsEndpoint('/api/router/v1/chat/completions')).toBe('/api/router/v1/models')
    expect(getRouterModelsEndpoint('http://localhost:8080/v1/chat/completions')).toBe(
      'http://localhost:8080/v1/models',
    )
    expect(getRouterModelsEndpoint('/custom/chat')).toBe('/api/router/v1/models')
  })
})
