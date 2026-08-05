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

  it('preserves recipe signals while updating complete model references', () => {
    const config = baseConfig()
    const updated = validateRecipeForm(
      {
        name: 'frontier-v2',
        description: 'updated',
        strategy: 'confidence',
        signals: config.recipes?.[0].routing.signals ?? {},
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
    expect(updated.routing.strategy).toBe('confidence')
    expect(updated.routing.decisions?.[0].modelRefs[0]).toMatchObject({
      model: models[1].name,
      use_reasoning: true,
      reasoning_effort: 'high',
      weight: 0.8,
    })
  })

  it('authors metadata and classifier policy inside one recipe', () => {
    const config = baseConfig()
    const updated = validateRecipeForm(
      {
        name: 'frontier',
        description: 'policy',
        strategy: 'priority',
        signals: {
          metadata: [
            {
              name: 'premium',
              key: 'tier',
              predicate: { in: ['gold', 'platinum'] },
            },
          ],
          classifiers: [
            {
              name: 'risk',
              type: 'local',
              model_path: 'models/risk',
              labels: ['SAFE', 'RISKY'],
              use_cpu: true,
            },
          ],
        },
        decisions: [
          {
            name: 'frontier_route',
            description: 'frontier',
            priority: 100,
            rules: {
              operator: 'AND',
              conditions: [
                { type: 'metadata', name: 'premium' },
                {
                  type: 'classifier',
                  name: 'risk',
                  label: 'RISKY',
                  predicate: { gte: 0.5 },
                  on_error: 'no_match',
                },
              ],
            },
            modelRefs: [{ model: models[1].name, use_reasoning: true }],
          },
        ],
      },
      config,
      models,
      'frontier',
    )

    expect(updated.routing.signals?.metadata?.[0].name).toBe('premium')
    expect(updated.routing.signals?.classifiers?.[0].name).toBe('risk')
    expect(updated.routing.decisions?.[0].rules.conditions).toHaveLength(2)
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
          strategy: 'priority',
          signals: explicitDefault.routing.signals ?? {},
          decisions: explicitDefault.routing.decisions ?? [],
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
          signals: explicitDefault.routing.signals ?? {},
          decisions: explicitDefault.routing.decisions ?? [],
        },
        config,
        models,
        'default',
      ),
    ).toThrow(/cannot be renamed/)
    expect(getRecipeDeleteBlocker(config, 'default')).toMatch(/cannot be deleted/)
  })

  it('allows decision names to repeat across recipes but not within one recipe', () => {
    const config = baseConfig()
    const sharedDecision = {
      name: 'default_route',
      description: 'recipe-local route',
      priority: 1,
      rules: { operator: 'AND' as const, conditions: [] },
      modelRefs: [{ model: models[1].name, use_reasoning: false }],
    }

    expect(() =>
      validateRecipeForm(
        {
          name: 'frontier',
          description: 'updated',
          strategy: 'priority',
          signals: config.recipes?.[0].routing.signals ?? {},
          decisions: [sharedDecision],
        },
        config,
        models,
        'frontier',
      ),
    ).not.toThrow()

    expect(() =>
      validateRecipeForm(
        {
          name: 'frontier',
          description: 'updated',
          strategy: 'priority',
          signals: config.recipes?.[0].routing.signals ?? {},
          decisions: [sharedDecision, sharedDecision],
        },
        config,
        models,
        'frontier',
      ),
    ).toThrow(/within this recipe/)
  })
})
