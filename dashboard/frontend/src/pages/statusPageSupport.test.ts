import { describe, expect, it } from 'vitest'

import type {
  EmbeddingProviderRuntimeStatus,
  ServiceStatusHistory,
  SystemStatus,
} from '../utils/routerRuntime'
import {
  deploymentUptime,
  describeOverall,
  describeService,
  embeddingProviderHealthLabel,
  embeddingProviderTone,
  formatDeploymentType,
  formatEmbeddingProviderBackend,
  formatProviderCheckedAt,
  formatUptime,
  historyUptime,
  orderedServiceNames,
} from './statusPageSupport'

const status = (overrides: Partial<SystemStatus> = {}): SystemStatus => ({
  overall: 'healthy',
  deployment_type: 'local',
  services: [
    { name: 'Router', status: 'running', healthy: true },
    { name: 'Dashboard', status: 'running', healthy: true },
  ],
  ...overrides,
})

const history = (name: string, states: ServiceStatusHistory['hours'][number]['status'][]) => ({
  name,
  hours: states.map((state, index) => ({
    observedAt: new Date(Date.UTC(2026, 7, 24, index)).toISOString(),
    status: state,
  })),
})

describe('status page support', () => {
  it('only calls a deployment operational when every reported service is healthy', () => {
    expect(describeOverall(status())).toMatchObject({
      title: 'All systems operational',
      tone: 'success',
      fullyOperational: true,
    })
    const degraded = status({
      overall: 'degraded',
      services: [
        ...status().services,
        { name: 'Telemetry', status: 'unavailable', healthy: false },
      ],
    })
    expect(describeOverall(degraded)).toMatchObject({ title: 'Degraded', tone: 'warning' })
    expect(describeOverall(status({ services: [] })).title).toBe('No running services detected')
    expect(describeOverall(null).title).toBe('Status unavailable')
  })

  it('reports an offline Router as a danger state instead of a generic label', () => {
    const offline = status({
      overall: 'not_running',
      services: [{ name: 'Router', status: 'not running', healthy: false }],
    })
    expect(describeOverall(offline)).toMatchObject({ title: 'Router offline', tone: 'danger' })
  })

  it('describes each service state with its condition', () => {
    expect(describeService({ name: 'Router', status: 'running', healthy: true })).toEqual({
      label: 'Operational',
      condition: 'Normal',
      tone: 'success',
    })
    expect(describeService({ name: 'Router', status: 'starting', healthy: false }).tone).toBe(
      'info',
    )
    expect(describeService({ name: 'Envoy', status: 'unavailable', healthy: false })).toEqual({
      label: 'Unavailable',
      condition: 'Unavailable',
      tone: 'danger',
    })
    expect(describeService({ name: 'Router', status: 'degraded', healthy: false })).toEqual({
      label: 'Degraded',
      condition: 'Needs attention',
      tone: 'warning',
    })
    expect(describeService(undefined).label).toBe('Not reported')
  })

  it('keeps services that only appear in history after the live ones', () => {
    const value = status({
      history: {
        windowHours: 2,
        through: '2026-08-24T01:00:00Z',
        services: [history('Router', ['operational']), history('Gateway', ['unknown'])],
      },
    })
    expect(orderedServiceNames(value)).toEqual(['Router', 'Dashboard', 'Gateway'])
  })

  it('computes uptime from observed hours only', () => {
    expect(historyUptime(history('Router', ['operational', 'unknown', 'unavailable']))).toBe(50)
    expect(historyUptime(history('Router', ['unknown', 'unknown']))).toBeNull()
    expect(formatUptime(null)).toBe('No data')
    expect(formatUptime(100)).toBe('100%')
    expect(formatUptime(97.5)).toBe('97.50%')
    const value = status({
      history: {
        windowHours: 2,
        through: '2026-08-24T01:00:00Z',
        services: [
          history('Router', ['operational', 'unavailable']),
          history('Dashboard', ['operational', 'operational']),
          history('Gateway', ['unknown', 'unknown']),
        ],
      },
    })
    expect(deploymentUptime(value)).toBe(75)
    expect(deploymentUptime(status())).toBeNull()
  })

  it('formats deployment facts for display', () => {
    expect(formatDeploymentType('none')).toBe('Not detected')
    expect(formatDeploymentType('local-spool')).toBe('Local Spool')
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
