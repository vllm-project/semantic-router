import { describe, expect, it } from 'vitest'

import {
  collectRecipeTargetModels,
  getRecipeByName,
  getPublishableRecipeNames,
  getRecipeDeleteBlocker,
  isSyntheticDefaultRecipe,
  listRecipeProfiles,
  normalizeEntrypointModelNames,
  requireEntrypointModelNames,
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
})

describe('entrypoints and recipes support', () => {
  it('normalizes public model IDs from newline and comma input', () => {
    expect(
      normalizeEntrypointModelNames(
        ' vllm-sr/mom-v1-flash,\nvllm-sr/mom-v1-flash\nvllm-sr/mom-v1-vault ',
      ),
    ).toEqual(['vllm-sr/mom-v1-flash', 'vllm-sr/mom-v1-vault'])
  })

  it('requires at least one public model name', () => {
    expect(requireEntrypointModelNames([' my/vault ', 'my/vault', 'vault'])).toEqual([
      'my/vault',
      'vault',
    ])
    expect(() => requireEntrypointModelNames([' '])).toThrow(
      'Add at least one request-facing model name.',
    )
  })

  it('rejects duplicate mappings but allows former special names as ordinary aliases', () => {
    const config = baseConfig()
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'vllm-sr/mom-v1-blend', recipe: 'default' },
        config,
        models,
        null,
      ),
    ).toThrow(/already mapped/)
    for (const modelNames of ['vllm-sr/auto', 'MoM', 'auto', 'vllm-sr/fusion', 'vllm-sr/remom']) {
      expect(() =>
        validateEntrypointForm({ modelNames, recipe: 'default' }, config, models, null),
      ).not.toThrow()
    }
  })

  it('reserves the effective builtin default only when no default override exists', () => {
    const config = baseConfig()
    config.entrypoints = []
    expect(() =>
      validateEntrypointForm(
        { modelNames: 'vllm-sr/auto', recipe: 'frontier' },
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
    ).not.toThrow()
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

const delegates = [
  { key: 'review_model', field: 'reviewModel', other: 'memory_model', noun: 'review' },
  { key: 'memory_model', field: 'memoryModel', other: 'review_model', noun: 'memory' },
] as const

it.each(delegates)(
  'saves and clears a $key binding while retaining entrypoint metadata',
  ({ key, field }) => {
    const config = baseConfig()
    config.entrypoints = [
      { model_names: ['main'], recipe: 'default', default: true, [key]: 'review' },
      { model_names: ['review'], recipe: 'frontier' },
    ]
    expect(
      validateEntrypointForm(
        { modelNames: 'main', recipe: 'default', [field]: 'review' },
        config,
        models,
        0,
      ),
    ).toEqual(config.entrypoints[0])
    expect(
      validateEntrypointForm(
        { modelNames: 'main', recipe: 'default', [field]: '' },
        config,
        models,
        0,
      ),
    ).toEqual({ model_names: ['main'], recipe: 'default', default: true })
  },
)

it('sends reviews and memory calls to one entrypoint', () => {
  const config = baseConfig()
  config.entrypoints = [
    { model_names: ['main'], recipe: 'default' },
    { model_names: ['review'], recipe: 'frontier' },
  ]
  expect(
    validateEntrypointForm(
      { modelNames: 'main', recipe: 'default', reviewModel: 'review', memoryModel: 'review' },
      config,
      models,
      0,
    ),
  ).toEqual({
    model_names: ['main'],
    recipe: 'default',
    review_model: 'review',
    memory_model: 'review',
  })
})

it.each(delegates)(
  'rejects missing, self, chained and cyclic $key bindings',
  ({ key, field, other }) => {
    const config = baseConfig()
    config.entrypoints = [
      { model_names: ['main', 'main-alias'], recipe: 'default' },
      { model_names: ['review'], recipe: 'frontier' },
    ]
    const validate = (target: string, index = 0) =>
      validateEntrypointForm(
        {
          modelNames: config.entrypoints![index].model_names.join(','),
          recipe: config.entrypoints![index].recipe,
          [field]: target,
        },
        config,
        models,
        index,
      )
    expect(() => validate('missing')).toThrow('does not exist')
    expect(() => validate('main-alias')).toThrow('another entrypoint')
    for (const sent of [key, other]) {
      config.entrypoints[1][sent] = 'main'
      expect(() => validate('review')).toThrow('own review_model or memory_model')
      delete config.entrypoints[1][sent]
      config.entrypoints[0][sent] = 'review'
      expect(() => validate('main', 1)).toThrow('own review_model or memory_model')
      delete config.entrypoints[0][sent]
    }
  },
)

it.each(delegates)('preserves names used by another entrypoint as its $key', ({ key, noun }) => {
  const config = baseConfig()
  config.entrypoints = [
    { model_names: ['main'], recipe: 'default', [key]: 'review' },
    { model_names: ['review'], recipe: 'frontier' },
  ]
  expect(() =>
    validateEntrypointForm({ modelNames: 'renamed', recipe: 'frontier' }, config, models, 1),
  ).toThrow(`Keep ${noun} model name`)
})
