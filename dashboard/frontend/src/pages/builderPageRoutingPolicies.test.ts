import { describe, expect, it } from 'vitest'
import type { ASTProgram } from '../types/dsl'
import {
  chooseDefaultBuilderRoutingScope,
  resolveBuilderRoutingScope,
} from './builderPageRoutingScopeSupport'

describe('recipe standing policies', () => {
  it('keeps a policy-only global scope visible and isolated', () => {
    const secondary: ASTProgram = {
      signals: [],
      routes: [],
      plugins: [],
      dataPolicy: { replay: true },
    }
    const ast: ASTProgram = {
      signals: [],
      routes: [],
      plugins: [],
      candidateRequirements: { capabilities: 'declared' },
      dataPolicy: { replay: false },
      recipes: [{ name: 'secondary', program: secondary, pos: { Line: 1, Column: 1 } }],
    }
    expect(chooseDefaultBuilderRoutingScope(ast)).toBe('global')
    expect(resolveBuilderRoutingScope(ast, 'recipe:secondary')?.dataPolicy).toEqual({
      replay: true,
    })
    expect(
      resolveBuilderRoutingScope(ast, 'recipe:secondary')?.candidateRequirements,
    ).toBeUndefined()
    expect(resolveBuilderRoutingScope(ast, 'global')?.dataPolicy).toEqual({ replay: false })
  })
})
