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
  path: ['recipes', '0', 'model_bindings', 'domain'],
  binding: 'old',
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
      expect(html).toContain('coding · domain')
    },
  )

  it('writes the deployment name under the signal type and keeps the other bindings', () => {
    const config: RouterConfig = {
      recipes: [
        {
          name: 'coding',
          routing: { model_bindings: { domain: { deployment: 'old' }, pii: 'kept' } },
        },
      ],
    }
    const next = withDecisionTaskBinding(config, binding, 'primary')
    expect(next.recipes?.[0].routing.model_bindings).toEqual({ domain: 'primary', pii: 'kept' })
    expect(config.recipes?.[0].routing.model_bindings?.domain).toEqual({ deployment: 'old' })
  })

  it.each(['jailbreak', 'pii', 'fact_check', 'user_feedback', 'modality', 'hallucination'])(
    'binds consumer %s by its signal-type key',
    (consumer) => {
      const next = withDecisionTaskBinding(
        { routing: { model_bindings: { [consumer]: 'old' } } },
        {
          ...binding,
          consumer,
          recipe: 'default',
          path: ['recipes', '0', 'model_bindings', consumer],
        },
        'd3-remote',
      )
      expect(next.routing?.model_bindings).toEqual({ [consumer]: 'd3-remote' })
    },
  )

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
      model_bindings: { domain: 'new' },
    })
    expect(next.global).toEqual(config.global)
    expect(config.recipes?.[1].routing).toEqual({ strategy: 'confidence' })
    const reset = withDecisionTaskBinding(next, binding, null)
    expect(reset.recipes?.[1].routing.model_bindings).toEqual({})
  })

  it('edits the global binding for a row the global catalog binds', () => {
    const global: DecisionTaskBinding = {
      ...binding,
      consumer: 'pii',
      source: 'global',
      path: ['recipes', '0', 'model_bindings', 'pii'],
    }
    const config: RouterConfig = {
      recipes: [{ name: 'coding', routing: { model_bindings: { domain: 'kept' } } }],
      global: {
        model_catalog: {
          system: { decision_model: { deployment: 'primary' } },
          bindings: { pii: { deployment: 'old' } },
        },
      },
    }
    const next = withDecisionTaskBinding(config, global, 'local-vela')
    expect(next.global).toEqual({
      model_catalog: {
        system: { decision_model: { deployment: 'primary' } },
        bindings: { pii: 'local-vela' },
      },
    })
    expect(next.recipes).toEqual(config.recipes)
    const inherited = withDecisionTaskBinding(next, global, null)
    expect(
      (inherited.global as { model_catalog: { bindings: object } }).model_catalog.bindings,
    ).toEqual({})
  })

  it('writes a binding of the default recipe into the routing profile', () => {
    const config: RouterConfig = {
      routing: {},
      default_recipe: { name: 'main' },
      recipes: [{ name: 'coding', routing: {} }],
    }
    const next = withDecisionTaskBinding(
      config,
      { ...binding, recipe: 'main', path: ['recipes', '0', 'model_bindings', 'domain'] },
      'primary',
    )
    expect(next.routing?.model_bindings).toEqual({ domain: 'primary' })
    expect(next.recipes?.[0].routing).toEqual({})
  })

  it('binds the complexity task under its signal type and leaves the rules alone', () => {
    const config: RouterConfig = {
      recipes: [
        {
          name: 'coding',
          routing: {
            model_bindings: { complexity: 'old', pii: 'kept' },
            signals: { complexity: [{ name: 'difficulty' }] },
          },
        },
      ],
    }
    const row = {
      ...binding,
      task_id: 'complexity',
      consumer: 'complexity',
      path: ['recipes', '0', 'model_bindings', 'complexity'],
    }
    const next = withDecisionTaskBinding(config, row, 'scorer')
    expect(next.recipes?.[0].routing).toEqual({
      model_bindings: { complexity: 'scorer', pii: 'kept' },
      signals: { complexity: [{ name: 'difficulty' }] },
    })
  })

  it('shows a selector row, which the Router lists without a path', () => {
    const html = renderToStaticMarkup(
      createElement(DecisionTaskBindings, {
        data: {
          default_deployment: 'primary',
          tasks: [],
          deployments: [],
          bindings: [
            {
              ...binding,
              task_id: 'selector',
              consumer: 'selector.pick',
              deployment: 'judge',
              source: 'rule',
              editable: false,
              path: null,
            },
          ],
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
        writable: true,
        save: async () => undefined,
      }),
    )
    expect(html).toContain('coding · selector.pick')
    expect(html).toContain('Decision algorithm')
    expect(html).not.toContain('>Change<')
  })

  it('rejects removed recipes and unsafe paths instead of modifying another target', () => {
    expect(() => withDecisionTaskBinding({ recipes: [] }, binding, 'new')).toThrow('removed')
    expect(() =>
      withDecisionTaskBinding(
        {},
        { ...binding, path: ['global', '__proto__', 'model_bindings', 'domain'] },
        'new',
      ),
    ).toThrow('editable')
    expect(() =>
      withDecisionTaskBinding(
        {},
        { ...binding, consumer: '__proto__', path: ['recipes', '0', 'model_bindings', '__proto__'] },
        'new',
      ),
    ).toThrow('editable')
  })
})
