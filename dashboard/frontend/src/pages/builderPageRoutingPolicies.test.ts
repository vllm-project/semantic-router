import { describe, expect, it } from 'vitest'
import type { ASTProgram } from '../types/dsl'
import {
  chooseDefaultBuilderRoutingScope,
  resolveBuilderRoutingScope,
} from './builderPageRoutingScopeSupport'

describe('recipe routing policies', () => {
  it('keeps default candidate requirements visible and isolated', () => {
    const secondary: ASTProgram = {
      signals: [],
      routes: [],
      plugins: [],
    }
    const ast: ASTProgram = {
      signals: [],
      routes: [],
      plugins: [],
      candidateRequirements: { capabilities: 'declared' },
      recipes: [{ name: 'secondary', program: secondary, pos: { Line: 1, Column: 1 } }],
    }
    expect(chooseDefaultBuilderRoutingScope(ast)).toBe('global')
    expect(
      resolveBuilderRoutingScope(ast, 'recipe:secondary')?.candidateRequirements,
    ).toBeUndefined()
    expect(resolveBuilderRoutingScope(ast, 'global')?.candidateRequirements).toEqual({
      capabilities: 'declared',
    })
  })

  it('keeps an explicit default strategy visible without assigning it to a named recipe', () => {
    const ast: ASTProgram = {
      signals: [],
      routes: [],
      plugins: [],
      strategy: 'confidence',
      recipes: [
        {
          name: 'secondary',
          program: { signals: [], routes: [], plugins: [] },
          pos: { Line: 1, Column: 1 },
        },
      ],
    }
    expect(chooseDefaultBuilderRoutingScope(ast)).toBe('global')
    expect(resolveBuilderRoutingScope(ast, 'global')?.strategy).toBe('confidence')
    expect(resolveBuilderRoutingScope(ast, 'recipe:secondary')?.strategy).toBeUndefined()
  })
})
