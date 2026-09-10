import { describe, expect, it } from 'vitest'

import {
  configSchemaFields,
  filterConfigSchemaIndex,
  focusedSchemaTitle,
  type ConfigSchemaIndex,
  type ConfigSchemaNode,
} from './configSchemaReferenceSupport'

const index: ConfigSchemaIndex = {
  contract_version: 'contract/v1',
  config_version: 'v0.3',
  schema_id: 'schema',
  default_view: 'index',
  sections: [
    { path: 'global', title: 'Global', description: 'Router services', required: false, href: '' },
    { path: 'routing', title: 'Routing', description: 'Routing policy', required: false, href: '' },
  ],
  surfaces: {
    algorithm: { count: 2, names: ['static', 'weighted'], href_template: '' },
    signal: { count: 1, names: ['keyword'], href_template: '' },
  },
}

describe('config schema reference support', () => {
  it('filters sections and surface names without losing catalog types', () => {
    expect(
      filterConfigSchemaIndex(index, 'policy').sections.map((section) => section.path),
    ).toEqual(['routing'])
    expect(filterConfigSchemaIndex(index, 'weight').surfaces.algorithm.names).toEqual(['weighted'])
    expect(filterConfigSchemaIndex(index, 'algorithm').surfaces.algorithm.names).toEqual([
      'static',
      'weighted',
    ])
  })

  it('resolves focused definition fields and annotations', () => {
    const document: ConfigSchemaNode = {
      $ref: '#/$defs/Selection',
      $defs: {
        Selection: {
          type: 'object',
          required: ['models'],
          properties: {
            models: { type: 'array', items: { type: 'string' }, description: 'Candidates' },
            timeout_ms: { type: 'integer', default: 100 },
          },
        },
      },
      'x-vllm-sr-view': { view: 'surface', kind: 'algorithm', name: 'static' },
      'x-vllm-sr-surface': { display_name: 'Static selection' },
    }

    expect(focusedSchemaTitle(document)).toBe('Static selection')
    expect(configSchemaFields(document)).toEqual([
      {
        name: 'models',
        type: 'array<string>',
        required: true,
        description: 'Candidates',
        details: [],
      },
      {
        name: 'timeout_ms',
        type: 'integer',
        required: false,
        description: undefined,
        details: ['default: 100'],
      },
    ])
  })
})
