import { describe, expect, it } from 'vitest'

import type { RecipeActivationPlan, RecipePackageSummary } from '../types/recipe'
import {
  filterRecipeRows,
  packageRowId,
  planChanges,
  publicModelCards,
  publicModelResolution,
  recipeRows,
  recipeRowStatus,
  restartRequired,
} from './configPageRecipesPageSupport'
import type { ConfigData } from './configPageSupport'

const config: ConfigData = {
  providers: { models: [{ name: 'model-a' }, { name: 'model-b' }] },
  default_recipe: { name: 'everything', description: 'The default profile' },
  routing: {
    decisions: [
      {
        name: 'low',
        description: '',
        priority: 1,
        rules: { operator: 'AND', conditions: [] },
        modelRefs: [{ model: 'model-a', use_reasoning: false }],
      },
      {
        name: 'high',
        description: '',
        priority: 9,
        rules: { operator: 'AND', conditions: [] },
        modelRefs: [{ model: 'model-b', use_reasoning: false }],
      },
    ],
  },
  entrypoints: [
    { model_names: ['acme/private', 'private'], recipe: 'privacy' },
    { model_names: ['acme/broken'], recipe: 'missing' },
    { api: 'systemone', model_names: ['vllm-sr/native'], recipe: 'privacy' },
  ],
  recipes: [{ name: 'privacy', description: 'Local only', routing: { decisions: [] } }],
}

const summary = (active: boolean): RecipePackageSummary => ({
  id: 'acme/secure',
  name: 'secure',
  version: '1.0.0',
  description: 'Packaged',
  recipe_digest: 'sha256:recipe',
  config_digest: 'sha256:config',
  archive_sha256: 'sha',
  archive_verified: true,
  installed_at: '2026-11-17T00:00:00Z',
  active,
  counts: { unified_models: 1, recipes: 1, decisions: 3, probes: 4 },
  warnings: [],
})

describe('public model cards', () => {
  it('lists chat entrypoints, the derived default and System One names with their resolution', () => {
    const cards = publicModelCards(config)
    expect(cards.map((card) => [card.key, card.api, card.derived])).toEqual([
      ['acme/private', 'chat', false],
      ['acme/broken', 'chat', false],
      ['vllm-sr/auto', 'chat', true],
      ['vllm-sr/native', 'systemone', false],
    ])
    expect(cards.map(publicModelResolution)).toEqual(['empty', 'missing', 'resolving', 'empty'])
    const fallback = cards.find((card) => card.key === 'vllm-sr/auto')!
    expect(fallback.index).toBe(-1)
    expect(fallback.decisions.map((decision) => decision.name)).toEqual(['high', 'low'])
    expect(fallback.targets).toEqual(['model-a', 'model-b'])
    expect(cards[3].index).toBe(2)
  })
})

describe('recipe rows', () => {
  it('combines config recipes, packages and built-in recipes with one status each', () => {
    const rows = recipeRows(
      config,
      {
        items: [summary(false)],
        page: 1,
        page_size: 100,
        total: 1,
        total_pages: 1,
        activation_state: 'none',
      },
      {
        catalog_version: 'latest',
        bundles: [
          {
            name: 'mom-v1',
            sha256: 'sha',
            recipes: [{ name: 'vault', description: 'Vault', decisions: [] }],
          },
        ],
      },
    )
    expect(rows.map((row) => [row.kind, row.id, row.name])).toEqual([
      ['config', 'default', 'everything'],
      ['config', 'privacy', 'privacy'],
      ['package', packageRowId('sha256:recipe'), 'secure'],
      ['builtin', 'builtin:mom-v1/vault', 'vault'],
    ])
    const [defaultRow, privacy] = rows
    expect(defaultRow.kind === 'config' && defaultRow.description).toBe('The default profile')
    expect(privacy.kind === 'config' && privacy.publicModels).toEqual([
      'acme/private',
      'private',
      'vllm-sr/native',
    ])
    const pending = new Set(['sha256:recipe'])
    expect(rows.map((row) => recipeRowStatus(row, new Set()))).toEqual([
      'loaded',
      'loaded',
      'inactive',
      'available',
    ])
    expect(recipeRowStatus(rows[2], pending)).toBe('pending')
    expect(filterRecipeRows(rows, 'acme/private', [], [], pending).map((row) => row.id)).toEqual([
      'privacy',
    ])
    expect(filterRecipeRows(rows, '', ['package'], [], pending)).toHaveLength(1)
    expect(filterRecipeRows(rows, '', [], ['available'], pending)).toHaveLength(1)
  })
})

describe('activation preview', () => {
  const plan: RecipeActivationPlan = {
    recipe_digest: 'sha256:recipe',
    plan_digest: 'sha256:plan',
    mode: 'stack_recreation',
    requires_confirmation: true,
    listeners_before: [{ name: 'public', address: '0.0.0.0', port: 8801, host_port: 8801 }],
    listeners_after: [{ name: 'public', address: '0.0.0.0', port: 8899, host_port: 8899 }],
    storage: { before: [], after: ['redis'], add: ['redis'], remove: [], repair: [] },
    management_auth: { mode: 'disabled', service_permissions: [] },
    effects: ['recreate the managed stack'],
  }

  it('lists listener, storage and stated effects as changes and calls for a restart', () => {
    expect(planChanges(plan)).toEqual([
      { kind: 'add', text: 'Listener public 0.0.0.0:8899' },
      { kind: 'remove', text: 'Listener public 0.0.0.0:8801' },
      { kind: 'add', text: 'Managed storage redis' },
      { kind: 'info', text: 'recreate the managed stack' },
    ])
    expect(restartRequired(plan)).toBe(true)
    expect(restartRequired({ ...plan, mode: 'hot_switch' })).toBe(false)
  })
})
