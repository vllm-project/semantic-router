import { describe, expect, it } from 'vitest'

import { updateRoute } from './dslMutations'

describe('updateRoute', () => {
  it('keeps route settings that the route form does not edit', () => {
    const source = `ROUTE business_route (description = "Business traffic.", on_unknown = "no_match") {
  PRIORITY 200
  # TIER 5 was too aggressive
  TIER 2
  WHEN domain("business")
  ACTION route "safe-model"
  MODEL "model-a" (reasoning = false)
  FOR candidate IN ["model-b" (weight = 1)] {
    MODEL candidate
  }
  PLUGIN system_prompt {
    system_prompt: "Answer with TIER 9 { details }"
  }
  EMIT retention {
    ttl_turns: 4 # keep short
  }
}
`

    const saved = updateRoute(source, 'business_route', {
      description: 'Business traffic, edited.',
      priority: 150,
      when: 'domain("business")',
      models: [{ model: 'model-a', reasoning: false }],
      plugins: [
        { name: 'system_prompt', fields: { system_prompt: 'Answer with TIER 9 { details }' } },
      ],
    })

    const expected = `ROUTE business_route (description = "Business traffic, edited.", on_unknown = "no_match") {
  PRIORITY 150

  WHEN domain("business")

  MODEL "model-a" (reasoning = false)

  PLUGIN system_prompt {
    system_prompt: "Answer with TIER 9 { details }"
  }
  TIER 2
  ACTION route "safe-model"
  FOR candidate IN ["model-b" (weight = 1)] {
    MODEL candidate
  }
  EMIT retention {
    ttl_turns: 4 # keep short
  }
}
`
    expect(saved).toBe(expected)
  })

  it('keeps header options when the description is cleared', () => {
    const source = `ROUTE guard_route (description = "Guard.", on_unknown = "fail_request") {
  PRIORITY 10
  MODEL "model-a"
}
`

    const saved = updateRoute(source, 'guard_route', {
      priority: 10,
      models: [{ model: 'model-a' }],
      plugins: [],
    })

    expect(saved.split('\n')[0]).toBe('ROUTE guard_route (on_unknown = "fail_request") {')
  })

  it('keeps header options written without commas', () => {
    const source = `ROUTE guard_route (description = "Guard." on_unknown = "fail_request") {
  PRIORITY 10
  MODEL "model-a"
}
`

    const saved = updateRoute(source, 'guard_route', {
      description: 'Guard.',
      priority: 10,
      models: [{ model: 'model-a' }],
      plugins: [],
    })

    expect(saved.split('\n')[0]).toBe(
      'ROUTE guard_route (description = "Guard.", on_unknown = "fail_request") {',
    )
  })

  it('keeps a model named TIER inside its MODEL statement on an unchanged Save', () => {
    const source = `ROUTE tier_route {
  PRIORITY 10
  MODEL TIER
  TIER 2
}
`

    const saved = updateRoute(source, 'tier_route', {
      priority: 10,
      models: [{ model: 'TIER' }],
      plugins: [],
    })

    expect(saved).toBe(`ROUTE tier_route {
  PRIORITY 10

  MODEL "TIER"

  TIER 2
}
`)
  })

  it('writes the model reasoning mode', () => {
    const saved = updateRoute('ROUTE r {\n  PRIORITY 1\n}\n', 'r', {
      priority: 1,
      models: [{ model: 'model-a', reasoning: true, mode: 'enabled' }],
      plugins: [],
    })

    expect(saved).toContain('MODEL "model-a" (mode = "enabled", reasoning = true)')
  })
})
