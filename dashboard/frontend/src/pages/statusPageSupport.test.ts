import { describe, expect, it } from 'vitest'

import type { EmbeddingProviderRuntimeStatus, ServiceStatus } from '../utils/routerRuntime'
import {
  clampPage,
  embeddingProviderHealthLabel,
  embeddingProviderTone,
  filterServices,
  formatEmbeddingProviderBackend,
  formatProviderCheckedAt,
} from './statusPageSupport'

const services: ServiceStatus[] = [
  { name: 'router', component: 'semantic-router', status: 'ready', healthy: true },
  { name: 'envoy', status: 'degraded', message: 'upstream unavailable', healthy: false },
]

describe('status page support', () => {
  it('filters services by health and searchable metadata', () => {
    expect(filterServices(services, '', 'unhealthy')).toEqual([services[1]])
    expect(filterServices(services, 'semantic', 'all')).toEqual([services[0]])
    expect(filterServices(services, 'upstream', 'all')).toEqual([services[1]])
  })

  it('clamps pagination after result counts change', () => {
    expect(clampPage(4, 13, 6)).toBe(3)
    expect(clampPage(0, 0, 6)).toBe(1)
  })

  it('maps embedding provider health without treating an unprobed provider as healthy', () => {
    const healthy: EmbeddingProviderRuntimeStatus = { healthy: true }
    const failed: EmbeddingProviderRuntimeStatus = { healthy: false }
    const pending: EmbeddingProviderRuntimeStatus = {}

    expect(embeddingProviderTone(healthy)).toBe('healthy')
    expect(embeddingProviderHealthLabel(healthy)).toBe('Healthy')
    expect(embeddingProviderTone(failed)).toBe('unhealthy')
    expect(embeddingProviderHealthLabel(failed)).toBe('Needs attention')
    expect(embeddingProviderTone(pending)).toBe('pending')
    expect(embeddingProviderHealthLabel(pending)).toBe('Not checked')
  })

  it('formats provider metadata for operational display', () => {
    expect(formatEmbeddingProviderBackend('openai_compatible')).toBe('OpenAI compatible')
    expect(formatEmbeddingProviderBackend('future_backend')).toBe('Future Backend')
    expect(formatProviderCheckedAt()).toBe('Not checked')
    expect(formatProviderCheckedAt('not-a-date')).toBe('not-a-date')
  })
})
