import { describe, expect, it } from 'vitest'

import { updateSignal } from '@/lib/dslMutations'
import type { ASTProgram } from '@/types/dsl'
import {
  chooseDefaultBuilderRoutingScope,
  listBuilderRoutingScopes,
  mutateBuilderRecipeSource,
  resolveBuilderRoutingScope,
  summarizeBuilderRoutingScopes,
} from './builderPageRoutingScopeSupport'

const emptyProgram = (): ASTProgram => ({
  signals: [],
  routes: [],
  plugins: [],
  models: [],
})

describe('summarizeBuilderRoutingScopes', () => {
  it('includes recipe-owned entities and scoped declarations', () => {
    const ast: ASTProgram = {
      ...emptyProgram(),
      models: [{ name: 'shared-model', fields: {}, pos: { Line: 1, Column: 1 } }],
      entrypoints: [
        {
          modelNames: ['vllm-sr/balanced'],
          recipe: 'balanced',
          pos: { Line: 2, Column: 1 },
        },
      ],
      recipes: [
        {
          name: 'balanced',
          pos: { Line: 3, Column: 1 },
          program: {
            ...emptyProgram(),
            signals: [
              {
                signalType: 'keyword',
                name: 'balanced-keyword',
                fields: {},
                pos: { Line: 4, Column: 1 },
              },
            ],
            routes: [
              {
                name: 'balanced-route',
                priority: 100,
                when: null,
                models: [],
                plugins: [],
                pos: { Line: 5, Column: 1 },
              },
            ],
          },
        },
      ],
    }

    expect(summarizeBuilderRoutingScopes(ast, null)).toMatchObject({
      signalCount: 1,
      routeCount: 1,
      recipeCount: 1,
      entrypointCount: 1,
    })
  })

  it('treats WASM null collections as empty arrays', () => {
    const ast = {
      signals: null,
      routes: null,
      plugins: null,
      models: [],
      entrypoints: null,
      recipes: [
        {
          name: 'balanced',
          program: {
            signals: null,
            routes: null,
            plugins: null,
          },
        },
      ],
    } as unknown as ASTProgram

    const source = `
ENTRYPOINT { model_names: ["vllm-sr/balanced"] recipe: "balanced" }
RECIPE balanced {
  SIGNAL keyword intent { keywords: ["hello"] }
  PROJECTION score effort { method: "weighted_sum" }
  ROUTE balanced_route { MODEL shared }
}`
    expect(() => summarizeBuilderRoutingScopes(ast, null, source)).not.toThrow()
    expect(summarizeBuilderRoutingScopes(ast, null, source)).toMatchObject({
      signalCount: 1,
      routeCount: 1,
      pluginCount: 0,
      recipeCount: 1,
      entrypointCount: 1,
      projectionScoreCount: 1,
    })
  })
})

describe('Builder routing scope navigation', () => {
  const scopedAst: ASTProgram = {
    ...emptyProgram(),
    models: [{ name: 'shared-model', fields: {}, pos: { Line: 1, Column: 1 } }],
    entrypoints: [
      {
        modelNames: ['vllm-sr/mom-balanced-v1'],
        recipe: 'balanced',
        pos: { Line: 2, Column: 1 },
      },
    ],
    recipes: [
      {
        name: 'balanced',
        pos: { Line: 3, Column: 1 },
        program: {
          ...emptyProgram(),
          signals: [
            {
              signalType: 'keyword',
              name: 'balanced-keyword',
              fields: {},
              pos: { Line: 4, Column: 1 },
            },
          ],
        },
      },
    ],
  }

  it('maps recipe scopes to entrypoint model names and shared models', () => {
    expect(listBuilderRoutingScopes(scopedAst)).toEqual([
      {
        id: 'global',
        label: 'Global catalog',
        recipeName: null,
        modelNames: [],
      },
      {
        id: 'recipe:balanced',
        label: 'balanced',
        recipeName: 'balanced',
        modelNames: ['vllm-sr/mom-balanced-v1'],
      },
    ])
    expect(resolveBuilderRoutingScope(scopedAst, 'recipe:balanced')).toMatchObject({
      models: [{ name: 'shared-model' }],
      signals: [{ name: 'balanced-keyword' }],
      recipes: [],
    })
    expect(chooseDefaultBuilderRoutingScope(scopedAst)).toBe('recipe:balanced')
  })

  it('applies mutations only inside the selected recipe', () => {
    const source = `MODEL shared {}
RECIPE balanced {
  SIGNAL keyword balanced-keyword {
    keywords: ["before"]
  }
}
RECIPE speed {
  SIGNAL keyword speed-keyword {
    keywords: ["unchanged"]
  }
}`

    const mutated = mutateBuilderRecipeSource(source, 'balanced', (program) =>
      updateSignal(program, 'keyword', 'balanced-keyword', { keywords: ['after'] }),
    )

    expect(mutated).toContain('keywords: ["after"]')
    expect(mutated).toContain('keywords: ["unchanged"]')
  })
})
