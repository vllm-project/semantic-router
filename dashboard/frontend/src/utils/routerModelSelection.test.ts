import { describe, expect, it } from 'vitest'

import {
  CANONICAL_AUTO_MODEL,
  getRouterModelsEndpoint,
  selectRouterAutoModel,
} from './routerModelSelection'

describe('router model selection', () => {
  it('prefers the canonical automatic-routing alias', () => {
    expect(selectRouterAutoModel({
      data: [
        { id: 'MoM', owned_by: 'vllm-semantic-router' },
        { id: CANONICAL_AUTO_MODEL, owned_by: 'vllm-semantic-router' },
        { id: 'qwen/qwen3.5-rocm', owned_by: 'vllm' },
      ],
    })).toBe(CANONICAL_AUTO_MODEL)
  })

  it('uses a live custom auto alias when the canonical alias is not exposed', () => {
    expect(selectRouterAutoModel({
      data: [
        {
          id: 'router/production',
          owned_by: 'vllm-semantic-router',
          description: 'Automatic model routing',
        },
        { id: 'qwen/qwen3.5-rocm', owned_by: 'vllm' },
      ],
    })).toBe('router/production')
  })

  it('does not mistake a backend model for the automatic router', () => {
    expect(selectRouterAutoModel({ data: [{ id: 'qwen/qwen3.5-rocm', owned_by: 'vllm' }] }))
      .toBeNull()
    expect(selectRouterAutoModel({
      data: [
        {
          id: 'backend/auto',
          owned_by: 'upstream-endpoint',
          description: 'Automatic model routing',
        },
      ],
    })).toBeNull()
    expect(selectRouterAutoModel({ data: 'invalid' })).toBeNull()
  })

  it('rejects the retired MoM compatibility alias instead of sending it from Playground', () => {
    expect(selectRouterAutoModel({
      data: [
        {
          id: 'MoM',
          owned_by: 'vllm-semantic-router',
          description: 'Intelligent Router for Mixture-of-Models',
        },
        {
          id: 'vllm-sr/MoM',
          owned_by: 'vllm-semantic-router',
          description: 'Intelligent Router for Mixture-of-Models',
        },
      ],
    })).toBeNull()
  })

  it('requires the canonical alias to be advertised by the semantic router', () => {
    expect(selectRouterAutoModel({
      data: [
        {
          id: CANONICAL_AUTO_MODEL,
          owned_by: 'upstream-endpoint',
          description: 'Automatic model routing',
        },
      ],
    })).toBeNull()
  })

  it('derives the models endpoint from local and absolute chat endpoints', () => {
    expect(getRouterModelsEndpoint('/api/router/v1/chat/completions')).toBe('/api/router/v1/models')
    expect(getRouterModelsEndpoint('http://localhost:8080/v1/chat/completions')).toBe('http://localhost:8080/v1/models')
    expect(getRouterModelsEndpoint('/custom/chat')).toBe('/api/router/v1/models')
  })
})
