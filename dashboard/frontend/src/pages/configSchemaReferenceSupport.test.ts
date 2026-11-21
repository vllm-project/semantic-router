import { describe, expect, it } from 'vitest'

import {
  configSchemaChildFields,
  configSchemaExample,
  configSchemaFields,
  configSchemaStructure,
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

const withoutNodes = (fields: ReturnType<typeof configSchemaFields>) =>
  fields.map((field) => {
    const copy = { ...field }
    delete copy.node
    return copy
  })

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
    expect(withoutNodes(configSchemaFields(document))).toEqual([
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

  it('describes scalar sections and array item fields instead of treating them as empty objects', () => {
    const version: ConfigSchemaNode = {
      type: 'string',
      const: 'v0.3',
      'x-vllm-sr-view': { view: 'section', path: 'version' },
    }
    expect(configSchemaStructure(version)).toEqual({
      type: 'string',
      fieldsLabel: 'Value',
      description: 'This section is a single configuration value.',
    })
    expect(withoutNodes(configSchemaFields(version))).toEqual([
      {
        name: 'value',
        type: 'string',
        required: true,
        description: undefined,
        details: ['fixed: v0.3'],
      },
    ])

    const listeners: ConfigSchemaNode = {
      type: 'array',
      items: { $ref: '#/$defs/Listener' },
      $defs: {
        Listener: {
          type: 'object',
          required: ['name', 'port'],
          properties: {
            name: { type: 'string', description: 'Stable listener name.' },
            port: { type: 'integer', minimum: 1 },
          },
        },
      },
      'x-vllm-sr-view': { view: 'section', path: 'listeners' },
    }
    expect(configSchemaStructure(listeners)).toEqual({
      type: 'array<object>',
      fieldsLabel: 'Item fields',
      description: 'Each list item uses the fields shown below.',
    })
    expect(configSchemaFields(listeners).map(({ name, required }) => ({ name, required }))).toEqual(
      [
        { name: 'name', required: true },
        { name: 'port', required: true },
      ],
    )
  })

  it('walks nested fields and writes a YAML starting point', () => {
    const section: ConfigSchemaNode = {
      $ref: '#/$defs/Global',
      $defs: {
        Global: {
          type: 'object',
          required: ['router'],
          properties: {
            router: { $ref: '#/$defs/Router' },
            stores: { type: 'object', properties: { cache: { type: 'boolean' } } },
          },
        },
        Router: {
          type: 'object',
          properties: {
            strategy: { type: 'string', enum: ['priority', 'confidence'] },
            timeout_ms: { type: 'integer', default: 300 },
          },
        },
      },
      'x-vllm-sr-view': { view: 'section', path: 'global' },
    }
    const [router] = configSchemaFields(section)
    expect(configSchemaChildFields(section, router.node).map((field) => field.name)).toEqual([
      'strategy',
      'timeout_ms',
    ])
    expect(configSchemaExample(section)).toBe(
      ['global:', '  router:', '    strategy: priority', '    timeout_ms: 300'].join('\n'),
    )
  })

  it('reads the compact section summary the bundled contract serves', () => {
    const summary: ConfigSchemaNode = {
      'x-vllm-sr-view': { view: 'section', path: 'global', detail: 'summary' },
      shape: 'object',
      title: 'Global',
      fields: [
        { name: 'router', path: 'global.router', type: 'object', required: false },
        { name: 'plugins', path: 'global.plugins', type: 'array<object>', required: false },
      ],
    }
    expect(configSchemaFields(summary).map((field) => [field.name, field.path])).toEqual([
      ['router', 'global.router'],
      ['plugins', 'global.plugins'],
    ])
    expect(configSchemaStructure(summary).type).toBe('object')
    expect(configSchemaExample(summary)).toBe(
      ['global:', '  router: {}', '  plugins: []'].join('\n'),
    )
  })
})
