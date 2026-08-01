import { describe, expect, it } from 'vitest'

import {
  CANONICAL_AUTO_MODEL,
  getRouterModelsEndpoint,
  listRouterModels,
  selectRouterAutoModel,
} from './routerModelSelection'

describe('router model selection', () => {
  it('prefers the canonical automatic-routing alias', () => {
    expect(
      selectRouterAutoModel({
        data: [
          { id: 'MoM', routing_type: 'auto_alias' },
          { id: CANONICAL_AUTO_MODEL, routing_type: 'auto_alias' },
          { id: 'qwen/qwen3.5-rocm', routing_type: 'backend' },
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
            routing_type: 'auto_alias',
            description: 'Any display copy can be used here',
          },
          { id: 'qwen/qwen3.5-rocm', routing_type: 'backend' },
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
      selectRouterAutoModel({ data: [{ id: 'qwen/qwen3.5-rocm', routing_type: 'backend' }] }),
    ).toBeNull()
    expect(
      selectRouterAutoModel({
        data: [
          {
            id: 'backend/auto',
            routing_type: 'backend',
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

  it('rejects the retired MoM compatibility alias instead of sending it from Playground', () => {
    expect(
      selectRouterAutoModel({
        data: [
          {
            id: 'MoM',
            routing_type: 'auto_alias',
            description: 'Intelligent Router for Mixture-of-Models',
          },
          {
            id: 'vllm-sr/MoM',
            routing_type: 'auto_alias',
            description: 'Intelligent Router for Mixture-of-Models',
          },
        ],
      }),
    ).toBeNull()
  })

  it('requires the canonical alias to be advertised as an auto alias', () => {
    expect(
      selectRouterAutoModel({
        data: [
          {
            id: CANONICAL_AUTO_MODEL,
            routing_type: 'backend',
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
            id: 'vllm-sr/mom-balanced-v1',
            routing_type: 'entrypoint',
            description: 'Intelligent Router for Mixture-of-Models',
          },
          {
            id: 'vllm-sr/mom-flash-v1',
            routing_type: 'entrypoint',
            description: 'Latency-first Mixture-of-Models profile',
          },
          {
            id: 'vllm-sr/auto',
            routing_type: 'auto_alias',
            description: 'Intelligent Router for Mixture-of-Models',
          },
          {
            id: 'router/production',
            routing_type: 'auto_alias',
            description: 'Automatic model routing',
          },
          { id: 'partner/backend', routing_type: 'backend' },
        ],
      }),
    ).toEqual([
      {
        id: 'vllm-sr/mom-balanced-v1',
        description: 'Intelligent Router for Mixture-of-Models',
      },
      {
        id: 'vllm-sr/mom-flash-v1',
        description: 'Latency-first Mixture-of-Models profile',
      },
    ])
  })

  it('keeps the canonical auto model usable when no entrypoints are advertised', () => {
    expect(
      listRouterModels({
        data: [
          { id: 'auto', routing_type: 'auto_alias' },
          { id: CANONICAL_AUTO_MODEL, routing_type: 'auto_alias' },
          { id: 'MoM', routing_type: 'auto_alias' },
        ],
      }),
    ).toEqual([{ id: CANONICAL_AUTO_MODEL, description: '' }])
  })

  it('rejects model records without a recognized routing type', () => {
    const payload = {
      data: [
        {
          id: 'router/profile',
          owned_by: 'vllm-semantic-router',
          description: 'Intelligent Router for Mixture-of-Models',
        },
        { id: 'router/unknown', routing_type: 'unknown' },
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
