import { describe, expect, it } from 'vitest'

import { configCollectionItemYaml, configYaml, diffLines } from './configYaml'

describe('configuration YAML', () => {
  it('orders typed fields like the Router schema and sorts free-form maps', () => {
    const yaml = configCollectionItemYaml(
      ['decisions'],
      {
        plugins: [{ configuration: { ttl_seconds: 900, enabled: true }, type: 'response_cache' }],
        modelRefs: [{ use_reasoning: false, model: 'qwen3-8b' }],
        rules: { conditions: [{ name: 'business', type: 'domain' }], operator: 'AND' },
        priority: 200,
        name: 'static_business_route',
      },
      '#/$defs/Decision',
    )
    expect(yaml).toBe(
      [
        'decisions:',
        '  - name: static_business_route',
        '    priority: 200',
        '    rules:',
        '      operator: AND',
        '      conditions:',
        '        - type: domain',
        '          name: business',
        '    modelRefs:',
        '      - model: qwen3-8b',
        '        use_reasoning: false',
        '    plugins:',
        '      - type: response_cache',
        '        configuration:',
        '          enabled: true',
        '          ttl_seconds: 900',
        '',
      ].join('\n'),
    )
  })

  it('quotes strings YAML would read as something else', () => {
    expect(
      configYaml({
        a: 'true',
        b: '1.0',
        c: '',
        d: 'computer science',
        e: 'x: y',
        f: '-x',
        g: '2026-08',
      }),
    ).toBe(
      [
        'a: "true"',
        'b: "1.0"',
        'c: ""',
        'd: computer science',
        'e: "x: y"',
        'f: "-x"',
        'g: 2026-08',
        '',
      ].join('\n'),
    )
  })

  it('writes multi-line text as a block and empty collections inline', () => {
    expect(configYaml({ prompt: 'line one\nline two', list: [], map: {} })).toBe(
      ['list: []', 'map: {}', 'prompt: |-', '  line one', '  line two', ''].join('\n'),
    )
  })

  it('diffs two documents line by line', () => {
    expect(diffLines('a\nb\nc\n', 'a\nB\nc\n')).toEqual([
      { kind: 'same', text: 'a' },
      { kind: 'remove', text: 'b' },
      { kind: 'add', text: 'B' },
      { kind: 'same', text: 'c' },
    ])
  })
})
