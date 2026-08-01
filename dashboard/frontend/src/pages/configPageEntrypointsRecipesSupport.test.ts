import { describe, expect, it } from 'vitest'

import {
  collectRecipeTargetModels,
  getRecipeByName,
  getRecipeDeleteBlocker,
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
  entrypoints: [{ model_names: ['vllm-sr/mom-balanced-v1'], recipe: 'default' }],
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
        ' vllm-sr/mom-flash-v1,\nvllm-sr/mom-flash-v1\nvllm-sr/mom-private-v1 ',
      ),
    ).toEqual(['vllm-sr/mom-flash-v1', 'vllm-sr/mom-private-v1'])
  })

  it('rejects duplicate and reserved entrypoint model IDs', () => {
    const config = baseConfig()
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'vllm-sr/mom-balanced-v1', recipe: 'default' },
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
    config.global = { router: { auto_model_name: 'router/custom-auto' } }
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'router/custom-auto', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).toThrow(/reserved/)
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

  it('preserves recipe signals while updating complete model references', () => {
    const config = baseConfig()
    const updated = validateRecipeForm(
      {
        name: 'frontier-v2',
        description: 'updated',
        decisions: [
          {
            name: 'frontier_route',
            description: 'frontier',
            priority: 200,
            rules: { operator: 'AND', conditions: [] },
            modelRefs: [
              {
                model: models[1].name,
                use_reasoning: true,
                reasoning_effort: 'high',
                weight: 0.8,
              },
            ],
          },
        ],
      },
      config,
      models,
      'frontier',
    )

    expect(updated.routing.signals).toEqual(config.recipes?.[0].routing.signals)
    expect(updated.routing.decisions?.[0].modelRefs[0]).toMatchObject({
      model: models[1].name,
      use_reasoning: true,
      reasoning_effort: 'high',
      weight: 0.8,
    })
  })

  it('collects physical targets and blocks deletion of referenced recipes', () => {
    const config = baseConfig()
    config.entrypoints?.push({
      model_names: ['vllm-sr/mom-frontier-v1'],
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
          decisions: explicitDefault.routing.decisions ?? [],
        },
        config,
        models,
        'default',
      ),
    ).not.toThrow()
  })
})
