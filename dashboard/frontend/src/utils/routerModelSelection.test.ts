import { describe, expect, it } from 'vitest'

import {
  CANONICAL_AUTO_MODEL,
  getRouterModelsEndpoint,
  listConfiguredBackendModels,
  listRouterModels,
  selectRouterAutoModel,
} from './routerModelSelection'

const routingMetadata = {
  defaultRoute: { resolution: 'virtual', selectable: true, default_route: true },
  profile: { resolution: 'virtual', selectable: true },
  passthrough: { resolution: 'passthrough', selectable: false },
} as const

describe('router model selection', () => {
  it('prefers the canonical automatic-routing alias', () => {
    expect(
      selectRouterAutoModel({
        data: [
          { id: 'MoM', routing: routingMetadata.defaultRoute },
          { id: CANONICAL_AUTO_MODEL, routing: routingMetadata.defaultRoute },
          { id: 'qwen/qwen3.5-rocm', routing: routingMetadata.passthrough },
        ],
      }),
    ).toBe(CANONICAL_AUTO_MODEL)
  })

  it('uses a live custom auto alias when the canonical alias is not exposed', () => {
    expect(
      selectRouterAutoModel({
        data: [
          {
            id: 'router/production',
            routing: routingMetadata.defaultRoute,
            description: 'Any display copy can be used here',
          },
          { id: 'qwen/qwen3.5-rocm', routing: routingMetadata.passthrough },
        ],
      }),
    ).toBe('router/production')
  })

  it('accepts a case-insensitive bare auto alias advertised by the router', () => {
    expect(
      selectRouterAutoModel({
        data: [{ id: 'AUTO', owned_by: 'vllm-semantic-router' }],
      }),
    ).toBe('AUTO')
  })

  it('does not mistake a backend model for the automatic router', () => {
    expect(
      selectRouterAutoModel({
        data: [{ id: 'qwen/qwen3.5-rocm', routing: routingMetadata.passthrough }],
      }),
    ).toBeNull()
    expect(
      selectRouterAutoModel({
        data: [
          {
            id: 'backend/auto',
            routing: routingMetadata.passthrough,
            description: 'Automatic model routing',
          },
        ],
      }),
    ).toBeNull()
    expect(
      selectRouterAutoModel({
        data: [{ id: 'backend/auto', owned_by: 'upstream-endpoint' }],
      }),
    ).toBeNull()
    expect(selectRouterAutoModel({ data: 'invalid' })).toBeNull()
  })

  it('rejects the retired bare MoM alias even when stale metadata marks it selectable', () => {
    expect(
      selectRouterAutoModel({
        data: [
          {
            id: 'MoM',
            routing: routingMetadata.defaultRoute,
            description: 'Intelligent Router for Mixture-of-Models',
          },
        ],
      }),
    ).toBeNull()
    expect(
      listRouterModels({
        data: [
          {
            id: 'MoM',
            routing: routingMetadata.defaultRoute,
            description: 'Intelligent Router for Mixture-of-Models',
          },
        ],
      }),
    ).toEqual([])
  })

  it('requires the canonical alias to be advertised as a selectable default route', () => {
    expect(
      selectRouterAutoModel({
        data: [
          {
            id: CANONICAL_AUTO_MODEL,
            routing: routingMetadata.passthrough,
            description: 'Automatic model routing',
          },
        ],
      }),
    ).toBeNull()
  })

  it('lists explicit routing profiles without interpreting their descriptions', () => {
    expect(
      listRouterModels({
        data: [
          {
            id: 'vllm-sr/mom-v1-blend',
            routing: { ...routingMetadata.profile, recipe: 'balanced' },
            description: 'Intelligent Router for Mixture-of-Models',
          },
          {
            id: 'vllm-sr/mom-v1-flash',
            routing: { ...routingMetadata.profile, mode: 'future-orchestrator' },
            description: 'Latency-first Mixture-of-Models profile',
          },
          {
            id: 'vllm-sr/auto',
            routing: routingMetadata.defaultRoute,
            description: 'Intelligent Router for Mixture-of-Models',
          },
          {
            id: 'router/production',
            routing: routingMetadata.defaultRoute,
            description: 'Automatic model routing',
          },
          { id: 'partner/backend', routing: routingMetadata.passthrough },
        ],
      }),
    ).toEqual([
      {
        id: 'vllm-sr/mom-v1-blend',
        description: 'Intelligent Router for Mixture-of-Models',
        recipe: 'balanced',
      },
      {
        id: 'vllm-sr/mom-v1-flash',
        description: 'Latency-first Mixture-of-Models profile',
      },
    ])
  })

  it('keeps the canonical auto model usable when no entrypoints are advertised', () => {
    expect(
      listRouterModels({
        data: [
          { id: 'auto', routing: routingMetadata.defaultRoute },
          { id: CANONICAL_AUTO_MODEL, routing: routingMetadata.defaultRoute },
          { id: 'MoM', routing: routingMetadata.defaultRoute },
        ],
      }),
    ).toEqual([{ id: CANONICAL_AUTO_MODEL, description: '' }])
  })

  it('lists configured backend models for admin playground testing', () => {
    expect(
      listConfiguredBackendModels({
        providers: {
          models: [
            { name: 'local/qwen', provider_model_id: 'upstream/qwen' },
            { name: ' local/qwen ', provider_model_id: 'duplicate' },
            { name: 'hosted/claude' },
            { name: '' },
            null,
          ],
        },
      }),
    ).toEqual([
      {
        id: 'local/qwen',
        description: 'upstream/qwen',
        kind: 'individual',
      },
      {
        id: 'hosted/claude',
        description: 'Configured backend model',
        kind: 'individual',
      },
    ])
    expect(listConfiguredBackendModels({ providers: { models: 'invalid' } })).toEqual([])
  })

  it('rejects model records without valid routing metadata', () => {
    const payload = {
      data: [
        {
          id: 'router/profile',
          owned_by: 'vllm-semantic-router',
          description: 'Intelligent Router for Mixture-of-Models',
        },
        { id: 'router/unknown', routing: { resolution: 'future', selectable: true } },
        {
          id: 'router/invalid-default',
          routing: { resolution: 'passthrough', selectable: false, default_route: true },
        },
      ],
    }
    expect(selectRouterAutoModel(payload)).toBeNull()
    expect(listRouterModels(payload)).toEqual([])
  })

  it('derives the models endpoint from local and absolute chat endpoints', () => {
    expect(getRouterModelsEndpoint('/api/router/v1/chat/completions')).toBe('/api/router/v1/models')
    expect(getRouterModelsEndpoint('http://localhost:8080/v1/chat/completions')).toBe(
      'http://localhost:8080/v1/models',
    )
    expect(getRouterModelsEndpoint('/custom/chat')).toBe('/api/router/v1/models')
  })
})
