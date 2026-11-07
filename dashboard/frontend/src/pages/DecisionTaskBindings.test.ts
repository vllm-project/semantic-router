import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'
import DecisionTaskBindings, { withDecisionTaskBinding } from './DecisionTaskBindings'
import type { RouterConfig } from './dashboardPageTypes'
import type { DecisionTaskBinding } from './useDecisionTasks'

const binding: DecisionTaskBinding = {
  task_id: 'domain',
  consumer: 'domain_classifier',
  recipe: 'coding',
  deployment: 'old',
  model: 'model',
  source: 'recipe',
  ready: true,
  editable: true,
  path: ['recipes', '0', 'routing', 'model_bindings', 'domain_classifier'],
  binding: 'old',
}

function containsContract(value: unknown): boolean {
  return JSON.stringify(value).includes('"contract"')
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
      expect(html).not.toContain('domain_classifier')
    },
  )

  it('writes the deployment name under the signal type and drops the old consumer key', () => {
    const config: RouterConfig = {
      recipes: [
        {
          name: 'coding',
          routing: {
            model_bindings: {
              domain_classifier: { deployment: 'old', contract: 'label_distribution.v1' },
              preference: 'tuned',
            },
          },
        },
      ],
    }
    const next = withDecisionTaskBinding(config, binding, 'primary')
    expect(next.recipes?.[0].routing.model_bindings).toEqual({
      domain: 'primary',
      preference: 'tuned',
    })
    expect(containsContract(next)).toBe(false)
    expect(config.recipes?.[0].routing.model_bindings?.domain_classifier).toEqual({
      deployment: 'old',
      contract: 'label_distribution.v1',
    })
  })

  it.each([
    ['prompt_guard', 'jailbreak'],
    ['pii_classifier', 'pii'],
    ['fact_check_classifier', 'fact_check'],
    ['feedback_detector', 'user_feedback'],
    ['modality_detector', 'modality'],
    ['hallucination_detector', 'hallucination'],
    ['preference', 'preference'],
    ['reask', 'reask'],
  ])('binds consumer %s by its signal-type key %s', (consumer, key) => {
    const next = withDecisionTaskBinding(
      { routing: { model_bindings: { [consumer]: 'old' } } },
      {
        ...binding,
        consumer,
        recipe: 'default',
        path: ['routing', 'model_bindings', consumer],
      },
      'd3-remote',
    )
    expect(next.routing?.model_bindings).toEqual({ [key]: 'd3-remote' })
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
      consumer: 'pii_classifier',
      source: 'global',
      path: ['recipes', '0', 'routing', 'model_bindings', 'pii_classifier'],
    }
    const config: RouterConfig = {
      recipes: [{ name: 'coding', routing: { model_bindings: { domain: 'kept' } } }],
      global: {
        model_catalog: {
          system: { decision_model: { deployment: 'primary' } },
          bindings: { pii_classifier: { deployment: 'old', contract: 'token_spans.v1' } },
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

  it('writes an explicit default recipe, not the top-level routing it would conflict with', () => {
    const config: RouterConfig = {
      recipes: [
        { name: 'coding', routing: {} },
        { name: 'default', routing: {} },
      ],
    }
    const next = withDecisionTaskBinding(
      config,
      { ...binding, recipe: 'default', path: ['routing', 'model_bindings', 'domain_classifier'] },
      'primary',
    )
    expect(next.routing).toBeUndefined()
    expect(next.recipes?.[1].routing.model_bindings).toEqual({ domain: 'primary' })
  })

  describe('a rule that asks its own question', () => {
    const recipeRow = (consumer: string): DecisionTaskBinding => ({
      ...binding,
      consumer,
      path: ['recipes', '0', 'routing', 'model_bindings', consumer],
    })
    const config = (): RouterConfig => ({
      recipes: [
        {
          name: 'coding',
          routing: {
            model_bindings: { 'classifier.risk': 'old', complexity: 'old', pii: 'kept' },
            signals: {
              classifiers: [
                { name: 'risk', type: 'llm', model: 'external-risk', labels: ['high', 'low'] },
                { name: 'other', labels: ['a'] },
              ],
              safety: [
                {
                  name: 'violence',
                  deployment: 'guard',
                  hazard: { model: 'external-hazard', categories: ['violence'] },
                },
              ],
              complexity: [{ name: 'difficulty' }, { name: 'reasoning', deployment: 'old' }],
            },
          },
        },
      ],
    })

    it("writes the rule's deployment instead of a binding and drops what it replaces", () => {
      const next = withDecisionTaskBinding(config(), recipeRow('classifier.risk'), 'd3-remote')
      const routing = next.recipes?.[0].routing
      expect(routing?.signals?.classifiers).toEqual([
        { name: 'risk', labels: ['high', 'low'], deployment: 'd3-remote' },
        { name: 'other', labels: ['a'] },
      ])
      expect(routing?.model_bindings).toEqual({ complexity: 'old', pii: 'kept' })
    })

    it('names the deployment of a hazard part on the hazard part', () => {
      const next = withDecisionTaskBinding(config(), recipeRow('safety.violence.hazard'), 'hazard')
      expect(next.recipes?.[0].routing.signals?.safety).toEqual([
        {
          name: 'violence',
          deployment: 'guard',
          hazard: { categories: ['violence'], deployment: 'hazard' },
        },
      ])
    })

    it('names the deployment of every complexity rule for the complexity task', () => {
      const next = withDecisionTaskBinding(config(), recipeRow('complexity'), 'scorer')
      const routing = next.recipes?.[0].routing
      expect(routing?.signals?.complexity).toEqual([
        { name: 'difficulty', deployment: 'scorer' },
        { name: 'reasoning', deployment: 'scorer' },
      ])
      expect(routing?.model_bindings).toEqual({ 'classifier.risk': 'old', pii: 'kept' })
      const inherited = withDecisionTaskBinding(next, recipeRow('complexity'), null)
      expect(inherited.recipes?.[0].routing.signals?.complexity).toEqual([
        { name: 'difficulty' },
        { name: 'reasoning' },
      ])
    })

    it('writes the rule in the row recipe even when a global key bound it', () => {
      const next = withDecisionTaskBinding(
        config(),
        { ...recipeRow('safety.violence'), source: 'global' },
        'judge',
      )
      expect(next.global).toBeUndefined()
      expect(next.recipes?.[0].routing.signals?.safety?.[0]).toMatchObject({ deployment: 'judge' })
    })

    it('refuses a removed rule and a consumer no binding or rule answers', () => {
      expect(() => withDecisionTaskBinding(config(), recipeRow('classifier.gone'), 'x')).toThrow(
        'rule was removed',
      )
      expect(() => withDecisionTaskBinding(config(), recipeRow('embedding'), 'x')).toThrow(
        'editable',
      )
    })

    it('shows the rule field a rule row writes', () => {
      const html = renderToStaticMarkup(
        createElement(DecisionTaskBindings, {
          data: {
            default_deployment: 'primary',
            tasks: [],
            deployments: [],
            bindings: [{ ...recipeRow('safety.violence.hazard'), deployment: 'hazard' }],
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
      expect(html).toContain('coding · signals.safety[violence].hazard.deployment')
    })
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
        { ...binding, consumer: '__proto__', path: ['routing', 'model_bindings', '__proto__'] },
        'new',
      ),
    ).toThrow('editable')
  })
})
