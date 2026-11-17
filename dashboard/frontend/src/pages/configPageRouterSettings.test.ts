import { describe, expect, it } from 'vitest'

import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import { ROUTER_SETTINGS_CARDS } from './configPageRouterSectionCatalog'
import {
  applyMergePatch,
  changedLeafPaths,
  mergePatchDiff,
  rebaseDraft,
  setAt,
  unsetAt,
} from './configPageRouterSettingsDraft'
import {
  contractNodeAt,
  contractOrdered,
  restartReason,
  routerSettingsSections,
  settingLabel,
  settingPresentation,
} from './configPageRouterSettingsModel'
import { validateRouterSettings } from './configPageRouterSettingsValidation'
import { toYaml, yamlDiff, yamlScalar } from './configPageRouterSettingsYaml'

describe('router settings draft', () => {
  it('sets and unsets paths without touching siblings, pruning emptied objects', () => {
    const saved = { router: { strategy: 'priority', fallback: { enabled: false } } }
    const set = setAt(saved, ['router', 'fallback', 'max_attempts'], 4)
    expect(set.router.strategy).toBe('priority')
    expect(set.router).not.toBe(saved.router)
    expect(saved.router.fallback).toEqual({ enabled: false })
    expect(
      unsetAt(unsetAt(set, ['router', 'fallback', 'max_attempts']), [
        'router',
        'fallback',
        'enabled',
      ]),
    ).toEqual({
      router: { strategy: 'priority' },
    })
  })

  it('writes a merge patch with only the changed keys and null for removed ones', () => {
    const saved = {
      router: { strategy: 'priority', drain_timeout_seconds: 25, headers: { debug: 'none' } },
    }
    const draft = {
      router: { strategy: 'priority', drain_timeout_seconds: 40 },
      stores: { memory: { auto_store: true } },
    }
    const patch = mergePatchDiff(saved, draft)
    expect(patch).toEqual({
      router: { drain_timeout_seconds: 40, headers: null },
      stores: { memory: { auto_store: true } },
    })
    expect(applyMergePatch(saved, patch)).toEqual(draft)
    expect(mergePatchDiff(saved, structuredClone(saved))).toBeUndefined()
  })

  it('replaces lists whole and counts changed leaves', () => {
    const saved = { router: { fallback: { retryable_status_codes: [502, 503] } } }
    const draft = { router: { fallback: { retryable_status_codes: [502, 503, 504] } } }
    expect(mergePatchDiff(saved, draft)).toEqual({
      router: { fallback: { retryable_status_codes: [502, 503, 504] } },
    })
    expect(changedLeafPaths(saved, draft)).toEqual([
      ['router', 'fallback', 'retryable_status_codes'],
    ])
  })

  it('keeps unsaved edits when the saved configuration refreshes', () => {
    const before = { router: { strategy: 'priority' } }
    const edited = { router: { strategy: 'confidence' } }
    const refreshed = { router: { strategy: 'priority', enabled: true } }
    expect(rebaseDraft(before, edited, refreshed)).toEqual({
      router: { strategy: 'confidence', enabled: true },
    })
  })
})

describe('router settings YAML', () => {
  it('quotes strings YAML would read as another type', () => {
    expect(yamlScalar('on')).toBe('"on"')
    expect(yamlScalar('25')).toBe('"25"')
    expect(yamlScalar('')).toBe('""')
    expect(yamlScalar('vsr:router-session:v1:')).toBe('"vsr:router-session:v1:"')
    expect(yamlScalar('o200k_base')).toBe('o200k_base')
    expect(yamlScalar('>folded')).toBe('">folded"')
    expect(yamlScalar(0.05)).toBe('0.05')
  })

  it('writes nested mappings and lists of mappings', () => {
    expect(
      toYaml({
        global: {
          router: {
            routing_input: { strip_blocks: [{ start: '<a>', end: '</a>' }] },
            allow_lossy_translation: [],
          },
        },
      }),
    ).toBe(
      [
        'global:',
        '  router:',
        '    routing_input:',
        '      strip_blocks:',
        '        - start: <a>',
        '          end: </a>',
        '    allow_lossy_translation: []',
      ].join('\n'),
    )
  })

  it('diffs into hunks that remove before they add', () => {
    const hunks = yamlDiff(
      { global: { router: { drain_timeout_seconds: 25, strategy: 'priority' } } },
      { global: { router: { drain_timeout_seconds: 40, strategy: 'priority' } } },
    )
    expect(hunks).toHaveLength(1)
    expect(hunks[0].header).toBe('global.router')
    expect(
      hunks[0].lines
        .filter((line) => line.kind !== 'equal')
        .map((line) => `${line.kind} ${line.text.trim()}`),
    ).toEqual(['remove drain_timeout_seconds: 25', 'add drain_timeout_seconds: 40'])
  })

  it('orders keys like the contract', () => {
    expect(Object.keys(contractOrdered({ plugins: [], stores: {}, router: {} }) as object)).toEqual(
      ['router', 'stores', 'plugins'],
    )
  })
})

describe('router settings model', () => {
  const sections = routerSettingsSections()

  it('renders every generated global section, and only contract paths', () => {
    const roots = sections.flatMap((section) => section.roots.map((root) => root.path.join('.')))
    for (const surface of ROUTER_CONFIG_EXTENSION.global_sections) {
      const dotted = surface.path.join('.')
      expect(
        roots.some(
          (root) =>
            root === dotted || root.startsWith(`${dotted}.`) || dotted.startsWith(`${root}.`),
        ),
        dotted,
      ).toBe(true)
    }
    for (const card of ROUTER_SETTINGS_CARDS) {
      for (const root of card.roots)
        expect(contractNodeAt(['global', ...root.path]), root.path.join('.')).toBeDefined()
    }
    expect(sections.map((section) => section.key)).toEqual(
      ROUTER_SETTINGS_CARDS.map((card) => card.key),
    )
  })

  it('splits the router block and lifts section switches into card headers', () => {
    const core = sections.find((section) => section.key === 'router_core')
    expect(core?.fields.map((field) => field.key)).not.toContain('learning')
    expect(core?.fields.map((field) => field.key)).not.toContain('fallback')
    expect(core?.fields.map((field) => field.key)).toContain('drain_timeout_seconds')
    const learning = sections.find((section) => section.key === 'learning')
    expect(learning?.master?.path).toEqual(['router', 'learning', 'enabled'])
    expect(learning?.fields.map((field) => field.key)).not.toContain('enabled')
  })

  it('edits named entries of maps the contract types', () => {
    const field = (key: string, name: string) =>
      sections.find((section) => section.key === key)?.fields.find((entry) => entry.key === name)
    expect(field('admission', 'admission')?.schema.kind).toBe('object-map')
    expect(field('deployments', 'deployments')?.schema.kind).toBe('object-map')
    expect(field('masking_identity', 'namespaces')?.schema.kind).toBe('object-map')
    expect(Object.keys(field('admission', 'admission')?.schema.item?.fields ?? {})).toContain(
      'max_concurrency',
    )
  })

  it('offers the allowed values of lists the contract limits', () => {
    const core = sections.find((section) => section.key === 'router_core')
    const lossy = core?.fields.find((field) => field.key === 'allow_lossy_translation')
    expect(lossy?.schema.kind).toBe('string-list')
    expect(lossy?.schema.options).toContain('tool_result.is_error')
  })

  it('keeps curated labels and options over the contract', () => {
    const core = sections.find((section) => section.key === 'router_core')
    const strategy = core?.fields.find((field) => field.key === 'strategy')
    expect(strategy?.schema.kind).toBe('select')
    expect(strategy?.schema.options).toEqual(['priority', 'confidence'])
  })

  it('reads labels and units from keys', () => {
    expect(settingLabel('Max Entries')).toBe('Max entries')
    expect(settingLabel('Tls Ca Path')).toBe('TLS CA path')
    expect(settingLabel('efConstruction')).toBe('efConstruction')
    expect(settingPresentation('drain_timeout_seconds', 'Drain Timeout Seconds')).toEqual({
      label: 'Drain timeout',
      unit: 's',
    })
    expect(settingPresentation('ttl_seconds', 'TTL (seconds)')).toEqual({ label: 'TTL', unit: 's' })
    expect(settingPresentation('max_bytes', 'Max Bytes')).toEqual({
      label: 'Max bytes',
      unit: 'bytes',
    })
  })

  it('flags the global fields that need a restart', () => {
    expect(restartReason(['services', 'api', 'routing_preview', 'max_concurrency'])).toBeTruthy()
    expect(restartReason(['services', 'authz', 'identity', 'user_id_header'])).toBeTruthy()
    expect(restartReason(['router', 'drain_timeout_seconds'])).toBeUndefined()
  })
})

describe('router settings validation', () => {
  const sections = routerSettingsSections()

  it('checks the edited fields against the contract types and bounds', () => {
    const errors = validateRouterSettings(
      sections,
      { router: { drain_timeout_seconds: '4o' }, services: { management_api: { port: 70000 } } },
      {},
    )
    expect(errors.get('router.drain_timeout_seconds')).toMatch(/number/)
    expect(errors.get('services.management_api.port')).toBeUndefined()
  })

  it('requires the fields list items need', () => {
    const errors = validateRouterSettings(
      sections,
      { services: { budgets: { users: [{ limit: 5, window: 'day' }] } } },
      {},
    )
    expect(errors.get('services.budgets.users')).toMatch(/Name is required/)
  })

  it('runs the shared structured normalizer on curated fields', () => {
    const errors = validateRouterSettings(
      sections,
      { services: { budgets: { run: { idle_timeout_seconds: 60 } } } },
      {},
    )
    expect(errors.get('services.budgets.run')).toMatch(/required/)
  })

  it('rejects list values the contract does not allow', () => {
    const errors = validateRouterSettings(
      sections,
      { router: { allow_lossy_translation: ['store', 'everything'] } },
      {},
    )
    expect(errors.get('router.allow_lossy_translation')).toMatch(/everything/)
  })

  it('leaves unchanged values alone', () => {
    const saved = { router: { drain_timeout_seconds: '4o' } }
    expect(validateRouterSettings(sections, saved, saved).size).toBe(0)
  })
})
