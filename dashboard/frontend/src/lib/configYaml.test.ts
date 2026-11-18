import { readdirSync, readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import {
  configCollectionItemYaml,
  configYaml,
  configYamlDiff,
  configYamlLines,
  diffLines,
} from './configYaml'

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
    expect(diffLines('a\nb\n', 'x\na\ny\nb\nz\n').map((line) => line.kind)).toEqual([
      'add',
      'same',
      'add',
      'same',
      'add',
    ])
  })

  it('tags every line with the path of the value it writes', () => {
    expect(
      configYamlLines({ router: { prompt: 'one\ntwo', strip_blocks: [{ start: '<a>' }] } }),
    ).toEqual([
      { text: 'router:', path: ['router'] },
      { text: '  prompt: |-', path: ['router', 'prompt'] },
      { text: '    one', path: ['router', 'prompt'] },
      { text: '    two', path: ['router', 'prompt'] },
      { text: '  strip_blocks:', path: ['router', 'strip_blocks'] },
      { text: '    - start: <a>', path: ['router', 'strip_blocks', '0', 'start'] },
    ])
  })

  it('cuts a YAML diff into numbered hunks with context', () => {
    const before = Object.fromEntries([...'abcdefghijkl'].map((name, index) => [name, index]))
    const after = { ...before, b: 'one', k: 'ten' }
    const hunks = configYamlDiff({ global: before }, { global: after }, { context: 1 })
    expect(hunks.map((hunk) => [hunk.header, hunk.skippedBefore, hunk.lines.length])).toEqual([
      ['global', 1, 4],
      ['global', 6, 4],
    ])
    expect(hunks[0].lines.map((line) => [line.kind, line.oldNumber, line.newNumber])).toEqual([
      ['same', 2, 2],
      ['remove', 3, undefined],
      ['add', undefined, 3],
      ['same', 4, 4],
    ])
    expect(hunks[0].lines[1].path).toEqual(['global', 'b'])

    const [whole] = configYamlDiff({ global: before }, { global: after }, { context: Infinity })
    expect(whole.lines.filter((line) => line.kind !== 'remove')).toHaveLength(13)
    expect(configYamlDiff({ global: before }, { global: { ...before } })).toEqual([])
  })
})

describe('one configuration YAML renderer', () => {
  const source = (path: string) => readFileSync(new URL(path, import.meta.url), 'utf8')
  const importsShared = /from '\.\.\/lib\/configYaml'/

  it('renders router settings and the setup preview with this module', () => {
    for (const caller of [
      '../pages/ConfigPageRouterConfigSectionReview.tsx',
      '../pages/ConfigPageRouterConfigSectionYaml.tsx',
      '../pages/setupWizardSupport.ts',
    ]) {
      expect(source(caller), caller).toMatch(importsShared)
    }
    const pages = readdirSync(new URL('../pages/', import.meta.url), { withFileTypes: true })
    expect(pages.map((entry) => entry.name)).not.toContain('configPageRouterSettingsYaml.ts')
  })

  it('renders the routing sheets and their diffs with this module', () => {
    for (const caller of [
      '../pages/ConfigPageDecisionsSectionSheet.tsx',
      '../pages/ConfigPageSignalsSectionSheet.tsx',
      '../pages/configPageRoutingScopeViews.tsx',
    ]) {
      expect(source(caller), caller).toMatch(importsShared)
    }
  })
})
