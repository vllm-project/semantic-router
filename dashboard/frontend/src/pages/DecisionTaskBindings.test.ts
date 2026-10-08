import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import DecisionTaskBindings, { withDecisionTaskBinding } from './DecisionTaskBindings'
import type { RouterConfig } from './dashboardPageTypes'
import type { DecisionTaskBinding } from './useDecisionTasks'

const binding: DecisionTaskBinding = {
  task_id: 'domain',
  consumer: 'domain',
  recipe: 'coding',
  deployment: 'old',
  model: 'model',
  source: 'recipe',
  ready: true,
  editable: true,
  path: ['recipes', '0', 'routing', 'model_bindings', 'domain'],
  binding: { deployment: 'old', contract: 'decision.v1' },
}
describe('inline task binding mutations', () => {
  it.each(['org/actual-model', 'primary', ''])(
    'displays model %s without duplicating its deployment',
    (model) => {
      const html = renderToStaticMarkup(
        createElement(DecisionTaskBindings, {
          data: {
            default_deployment: 'primary',
            tasks: [],
            deployments: [],
            bindings: [{ ...binding, deployment: 'primary', model }],
          },
          error: null,
          observation: {
            data: null,
            error: null,
            updatedAt: 1,
            loading: false,
            refreshing: false,
            stale: false,
          },
          writable: false,
          save: async () => undefined,
        }),
      )
      expect(html.match(/>primary</g)).toHaveLength(1)
      if (model && model !== 'primary') expect(html).toContain(`<span>${model}</span>`)
    },
  )
  it('replaces specialist adapters when choosing a generic decision deployment', () => {
    const specialist: DecisionTaskBinding = {
      ...binding,
      binding: {
        deployment: 'pii-specialist',
        contract: 'token_spans.v1',
        adapter: 'pii',
        head: 'entities',
        mapping_path: 'labels.json',
      },
    }
    const config: RouterConfig = {
      recipes: [
        {
          name: 'coding',
          routing: {
            model_bindings: { domain: specialist.binding, preference: binding.binding },
          },
        },
      ],
    }
    const next = withDecisionTaskBinding(config, specialist, 'primary')
    expect(next.recipes?.[0].routing.model_bindings).toEqual({
      domain: { deployment: 'primary', contract: 'decision.v1' },
      preference: binding.binding,
    })
    expect(config.recipes?.[0].routing.model_bindings?.domain).toEqual(specialist.binding)
  })
  it('resolves a recipe by identity against fresh configuration and preserves other settings', () => {
    const config: RouterConfig = {
      recipes: [
        { name: 'support', routing: {} },
        { name: 'coding', routing: { strategy: 'confidence' } },
      ],
      global: { model_catalog: { system: { decision_model: { deployment: 'primary' } } } },
    }
    const next = withDecisionTaskBinding(config, binding, 'new')
    expect(next.recipes?.[0].routing).toEqual({})
    expect(next.recipes?.[1].routing).toEqual({
      strategy: 'confidence',
      model_bindings: { domain: { deployment: 'new', contract: 'decision.v1' } },
    })
    expect(next.global).toEqual(config.global)
    expect(config.recipes?.[1].routing).toEqual({ strategy: 'confidence' })
    const reset = withDecisionTaskBinding(next, binding, null)
    expect(reset.recipes?.[1].routing.model_bindings).toEqual({})
  })
  it('rejects removed recipes and unsafe paths instead of modifying another target', () => {
    expect(() => withDecisionTaskBinding({ recipes: [] }, binding, 'new')).toThrow('removed')
    expect(() =>
      withDecisionTaskBinding(
        {},
        { ...binding, path: ['global', '__proto__', 'model_bindings'] },
        'new',
      ),
    ).toThrow('editable')
  })
})
