import { describe, expect, it } from 'vitest'

import type { ConfigData } from './configPageSupport'
import {
  canonicalizeConfigForManagerSave,
  removeModelCardDataIfUnreferenced,
  routingModelCardReferenceCount,
  writeRoutingModelCard,
} from './configPageCanonicalization'

const sharedConfig = (): ConfigData => ({
  providers: {
    models: [
      { name: 'public-a', catalog: 'openai/gpt-5.4' },
      { name: 'public-b', catalog: 'openai/gpt-5.4' },
    ],
  },
  routing: {
    modelCards: [{ name: 'openai/gpt-5.4', description: 'Approved profile' }],
  },
})

describe('shared routing model card canonicalization', () => {
  it('preserves an existing catalog override when a second alias adds no metadata', () => {
    const config = sharedConfig()

    writeRoutingModelCard(config, 'openai/gpt-5.4', {})

    expect(config.routing?.modelCards).toEqual([
      { name: 'openai/gpt-5.4', description: 'Approved profile' },
    ])
    expect(routingModelCardReferenceCount(config, 'openai/gpt-5.4')).toBe(2)
  })

  it('removes an old override only after the last alias changes catalog or is deleted', () => {
    const config = sharedConfig()
    config.evaluation = {
      records: [
        {
          model: 'openai/gpt-5.4',
          benchmark: 'acme/quality@1.0.0',
          metrics: { score: 0.9 },
        },
      ],
    }
    config.providers!.models[0].catalog = 'anthropic/claude-sonnet-5'

    removeModelCardDataIfUnreferenced(config, 'openai/gpt-5.4')
    expect(config.routing?.modelCards).toHaveLength(1)
    expect(config.evaluation?.records).toHaveLength(1)

    config.providers!.models = config.providers!.models.filter((model) => model.name !== 'public-b')
    removeModelCardDataIfUnreferenced(config, 'openai/gpt-5.4')
    expect(config.routing?.modelCards).toEqual([])
    expect(config.evaluation).toBeUndefined()
  })

  it('allows an explicit edit to clear the shared override', () => {
    const config = sharedConfig()

    writeRoutingModelCard(config, 'openai/gpt-5.4', {}, { removeWhenEmpty: true })

    expect(config.routing?.modelCards).toEqual([])
  })
})

describe('legacy reasoning family canonicalization', () => {
  it('inlines custom definitions and retains unresolved built-in family references', () => {
    const customReasoning = {
      type: 'reasoning_effort',
      parameter: 'reasoning_effort',
      activation_parameter: 'enable_thinking',
      levels: ['disabled', 'low', 'high'],
      default: 'low',
      disabled: 'disabled',
    }
    const config: ConfigData = {
      providers: {
        defaults: {
          reasoning_families: {
            nested: {
              type: 'chat_template_kwargs',
              parameter: 'thinking_budget',
              levels: ['low', 'high'],
              default: 'high',
            },
          },
        },
        models: [{ name: 'existing-custom' }],
      },
      reasoning_families: { custom: customReasoning },
      model_config: {
        'existing-custom': { model_id: 'custom-v1', reasoning_family: 'custom' },
        'new-custom': { model_id: 'custom-v2', reasoning_family: 'custom' },
        'nested-custom': { model_id: 'nested-v1', reasoning_family: 'nested' },
        built_in: { model_id: 'built-in-v1', reasoning_family: 'qwen3' },
      },
    }

    const canonical = canonicalizeConfigForManagerSave(config)

    expect(canonical.providers?.models).toEqual([
      {
        name: 'existing-custom',
        provider_model_id: 'custom-v1',
        reasoning: customReasoning,
      },
      {
        name: 'new-custom',
        provider_model_id: 'custom-v2',
        backend_refs: [],
        reasoning: customReasoning,
      },
      {
        name: 'nested-custom',
        provider_model_id: 'nested-v1',
        backend_refs: [],
        reasoning: {
          type: 'chat_template_kwargs',
          parameter: 'thinking_budget',
          levels: ['low', 'high'],
          default: 'high',
        },
      },
      {
        name: 'built_in',
        provider_model_id: 'built-in-v1',
        backend_refs: [],
        reasoning: { family: 'qwen3' },
      },
    ])
    expect(canonical.reasoning_families).toBeUndefined()
    expect(canonical.providers?.defaults).toBeUndefined()
    expect(config.reasoning_families).toEqual({ custom: customReasoning })
    expect(config.providers?.defaults?.reasoning_families?.nested.default).toBe('high')
  })

  it('does not synthesize an empty provider defaults block', () => {
    const canonical = canonicalizeConfigForManagerSave({
      providers: { models: [{ name: 'custom', provider_model_id: 'custom-v1' }] },
    })

    expect(canonical.providers?.defaults).toBeUndefined()
  })

  it('canonicalizes a reasoning scalar directly on an existing provider model', () => {
    const definition = {
      type: 'chat_template_kwargs',
      parameter: 'thinking_budget',
      levels: ['low', 'high'],
      default: 'high',
    }
    const canonical = canonicalizeConfigForManagerSave({
      reasoning_families: { custom: definition },
      providers: { models: [{ name: 'direct-custom', reasoning_family: 'custom' }] },
    })

    expect(canonical.providers?.models[0]).toEqual({
      name: 'direct-custom',
      reasoning: definition,
    })
    expect(canonical.reasoning_families).toBeUndefined()
  })
})
