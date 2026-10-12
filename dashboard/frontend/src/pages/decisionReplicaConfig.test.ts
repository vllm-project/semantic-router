import { describe, expect, it } from 'vitest'
import type { RouterConfig } from './dashboardPageTypes'
import { deploymentPlacements, withDecisionReplicas } from './decisionReplicaConfig'
import { decisionRuntimeDeclarations } from './decisionRuntimeDeployment'

const snapshot = (): RouterConfig => ({
  version: 'v0.3',
  listeners: [{ name: 'public', address: '127.0.0.1', port: 8899, models: ['vllm-sr/auto'] }],
  global: {
    router: { enabled: false },
    model_catalog: {
      deployments: {
        primary: {
          provider: 'model_runtime',
          artifact: 'vllm-sr/Vela-2.0-4B',
          revision: 'a'.repeat(40),
          device: 'rocm:0',
          profile: 'exact',
          public_name: 'decider',
        },
        specialist: { provider: 'model_runtime', artifact: 'other/model', device: 'cpu' },
      },
      system: { decision_model: { deployment: 'primary' } },
      bindings: { prompt_guard: { deployment: 'specialist', contract: 'decision.v1' } },
    },
  },
})

describe('replica placement updates', () => {
  it('scales one logical resource without changing model identity, task bindings, mode or public scope', () => {
    const original = snapshot()
    const next = withDecisionReplicas(original, 'primary', [
      { device: 'rocm:0' },
      { device: 'rocm:1' },
    ])
    expect(next.listeners).toEqual(original.listeners)
    expect(next.global?.router).toEqual({ enabled: false })
    expect(next.global?.model_catalog).toMatchObject({
      system: { decision_model: { deployment: 'primary' } },
      bindings: { prompt_guard: { deployment: 'specialist', contract: 'decision.v1' } },
    })
    const deployment = decisionRuntimeDeclarations(next).primary
    expect(deployment).toEqual({
      provider: 'model_runtime',
      artifact: 'vllm-sr/Vela-2.0-4B',
      revision: 'a'.repeat(40),
      profile: 'exact',
      public_name: 'decider',
      replicas: [{ device: 'rocm:0' }, { device: 'rocm:1' }],
    })
    expect(decisionRuntimeDeclarations(original).primary.device).toBe('rocm:0')
  })

  it('preserves attached served names and removes ambiguous placement shorthand', () => {
    const current = snapshot()
    const deployment = decisionRuntimeDeclarations(current).primary
    delete deployment.device
    deployment.endpoint = 'http://worker:8100'
    deployment.served_name = 'upstream-id'
    const replicas = deploymentPlacements(deployment)
    const next = withDecisionReplicas(current, 'primary', [
      ...replicas,
      { endpoint: 'http://worker2:8100', served_name: 'another-id' },
    ])
    expect(decisionRuntimeDeclarations(next).primary).not.toHaveProperty('endpoint')
    expect(decisionRuntimeDeclarations(next).primary).not.toHaveProperty('served_name')
    expect(decisionRuntimeDeclarations(next).primary.replicas).toEqual([
      ...replicas,
      { endpoint: 'http://worker2:8100', served_name: 'another-id' },
    ])
  })

  it('refuses a deleted resource, empty pool and invalid placement before persisting', () => {
    expect(() => withDecisionReplicas(snapshot(), 'missing', [{ device: 'auto' }])).toThrow(
      'no longer configured',
    )
    expect(() => withDecisionReplicas(snapshot(), 'primary', [])).toThrow('at least one')
    expect(() =>
      withDecisionReplicas(snapshot(), 'primary', [{ endpoint: 'http://user:password@worker' }]),
    ).toThrow('without credentials')
    expect(() => withDecisionReplicas(snapshot(), 'primary', [{ device: 'rocm:-1' }])).toThrow(
      'Enter a device',
    )
  })
})
