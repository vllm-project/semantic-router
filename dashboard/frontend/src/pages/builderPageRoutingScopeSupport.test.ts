import { describe, expect, it } from 'vitest'

import type { ASTProgram } from '@/types/dsl'
import { summarizeBuilderRoutingScopes } from './builderPageRoutingScopeSupport'

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
})
