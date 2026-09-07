import { describe, expect, it } from 'vitest'

import {
  collectRecipeTargetModels,
  getRecipeByName,
  getPublishableRecipeNames,
  getRecipeDeleteBlocker,
  isSyntheticDefaultRecipe,
  listRecipeProfiles,
  normalizeEntrypointModelNames,
  validateEntrypointForm,
  validateRecipeForm,
} from './configPageEntrypointsRecipesSupport'
import type { ConfigData, NormalizedModel, RecipeConfig } from './configPageSupport'

const models: NormalizedModel[] = [
  { name: 'amd/rocm-v1-gemma', endpoints: [] },
  { name: 'amd/rocm-v1-gpt', endpoints: [] },
]

const baseConfig = (): ConfigData => ({
  routing: {
    modelCards: models.map((model) => ({ name: model.name })),
    decisions: [
      {
        name: 'default_route',
        description: 'default',
        priority: 10,
        rules: { operator: 'AND', conditions: [] },
        modelRefs: [{ model: models[0].name, use_reasoning: false }],
      },
    ],
  },
  entrypoints: [{ model_names: ['vllm-sr/mom-v1-blend'], recipe: 'default' }],
  recipes: [
    {
      name: 'frontier',
      description: 'frontier policy',
      routing: {
        signals: {
          keywords: [
            {
              name: 'hard',
              operator: 'OR',
              keywords: ['hard'],
              case_sensitive: false,
            },
          ],
        },
        decisions: [
          {
            name: 'frontier_route',
            description: 'frontier',
            priority: 100,
            rules: { operator: 'AND', conditions: [] },
            modelRefs: [{ model: models[1].name, use_reasoning: true }],
          },
        ],
      },
    },
  ],
  global: { router: { auto_model_names: ['vllm-sr/auto'] } },
})

describe('entrypoints and recipes support', () => {
  it('normalizes public model IDs from newline and comma input', () => {
    expect(
      normalizeEntrypointModelNames(
        ' vllm-sr/mom-v1-flash,\nvllm-sr/mom-v1-flash\nvllm-sr/mom-v1-vault ',
      ),
    ).toEqual(['vllm-sr/mom-v1-flash', 'vllm-sr/mom-v1-vault'])
  })

  it('rejects duplicate and reserved entrypoint model IDs', () => {
    const config = baseConfig()
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'vllm-sr/mom-v1-blend', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).toThrow(/already mapped/)
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'vllm-sr/auto', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).toThrow(/reserved/)
    config.global = { router: { auto_model_name: ' router/custom-auto ' } }
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'router/custom-auto', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).toThrow(/reserved/)
    config.global = {
      integrations: {
        looper: {
          fusion: { model_names: ['router/custom-fusion'] },
        },
      },
    } as ConfigData['global']
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'router/custom-fusion', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).toThrow(/direct router dispatch/)
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'vllm-sr/fusion', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).not.toThrow()
    config.global = undefined
    config.routing = {
      ...config.routing,
      decisions: [
        {
          name: 'remom-route',
          description: 'Direct ReMoM route',
          priority: 1,
          rules: { operator: 'AND', conditions: [] },
          modelRefs: [],
          algorithm: { type: 'remom' },
        },
      ],
    }
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'vllm-sr/remom', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).toThrow(/direct router dispatch/)
  })

  it('rejects entrypoint IDs that collide with physical models', () => {
    expect(() =>
      validateEntrypointForm(
        { modelNames: models[0].name, recipe: 'frontier' },
        baseConfig(),
        models,
        null,
      ),
    ).toThrow(/collides/)
  })

  it('preserves recipe contents while updating its identity and strategy', () => {
    const config = baseConfig()
    const updated = validateRecipeForm(
      {
        name: 'frontier-v2',
        description: 'updated',
        strategy: 'confidence',
      },
      config,
      models,
      'frontier',
    )

    expect(updated.routing.signals).toEqual(config.recipes?.[0].routing.signals)
    expect(updated.routing.strategy).toBe('confidence')
    expect(updated.routing.decisions).toEqual(config.recipes?.[0].routing.decisions)
  })

  it('collects physical targets and blocks deletion of referenced recipes', () => {
    const config = baseConfig()
    config.entrypoints?.push({
      model_names: ['vllm-sr/mom-v1-ultra'],
      recipe: 'frontier',
    })
    const recipe = config.recipes?.[0] as RecipeConfig

    expect(collectRecipeTargetModels(recipe)).toEqual([models[1].name])
    expect(getRecipeDeleteBlocker(config, 'frontier')).toMatch(/before deleting/)
  })

  it('uses and edits an explicit recipes-only default profile', () => {
    const config = baseConfig()
    const explicitDefault: RecipeConfig = {
      name: 'default',
      description: 'Explicit default profile',
      routing: {
        decisions: [
          {
            name: 'explicit_default_route',
            description: 'default',
            priority: 1,
            rules: { operator: 'AND', conditions: [] },
            modelRefs: [{ model: models[0].name, use_reasoning: false }],
          },
        ],
      },
    }
    config.routing = { modelCards: config.routing?.modelCards }
    config.decisions = undefined
    config.recipes = [explicitDefault, ...(config.recipes ?? [])]

    expect(getRecipeByName(config, 'default')).toBe(explicitDefault)
    expect(() =>
      validateRecipeForm(
        {
          name: 'default',
          description: 'Updated default',
          strategy: 'priority',
        },
        config,
        models,
        'default',
      ),
    ).not.toThrow()
    expect(() =>
      validateRecipeForm(
        {
          name: 'renamed-default',
          description: 'Invalid rename',
          strategy: 'priority',
        },
        config,
        models,
        'default',
      ),
    ).toThrow(/cannot be renamed/)
    expect(getRecipeDeleteBlocker(config, 'default')).toMatch(/cannot be deleted/)
  })

  it('lists the top-level default recipe without manufacturing an editable stored recipe', () => {
    const config = baseConfig()
    const recipes = listRecipeProfiles(config)

    expect(recipes.map((recipe) => recipe.name)).toEqual(['default', 'frontier'])
    expect(isSyntheticDefaultRecipe(config, recipes[0])).toBe(true)
    expect(config.recipes?.map((recipe) => recipe.name)).toEqual(['frontier'])
  })

  it('offers only recipes that can be published as a model', () => {
    const config = baseConfig()
    config.routing = { modelCards: config.routing?.modelCards, decisions: [] }

    expect(getPublishableRecipeNames(config)).toEqual(['frontier'])
  })
})
