import { describe, expect, it } from 'vitest'

import { mutateBuilderRecipeSource } from './builderPageRoutingScopeSupport'

describe('mutateBuilderRecipeSource', () => {
  it('changes only the selected Recipe body', () => {
    const source = `MODEL physical {}\n\nRECIPE balanced {\n  ROUTE simple {\n    PRIORITY 1\n  }\n}\n\nENTRYPOINT {\n  MODEL_NAME "auto"\n  USE_RECIPE balanced\n}`
    const result = mutateBuilderRecipeSource(
      source,
      'balanced',
      (body) => `${body}\n\nROUTE complex {\n  PRIORITY 2\n}`,
    )

    expect(result).toContain('ROUTE complex')
    expect(result).toContain('MODEL physical')
    expect(result).toContain('ENTRYPOINT')
  })

  it('fails closed when the Recipe is not present', () => {
    expect(mutateBuilderRecipeSource('ROUTE global {}', 'missing', () => '')).toBe(
      'ROUTE global {}',
    )
  })
})
