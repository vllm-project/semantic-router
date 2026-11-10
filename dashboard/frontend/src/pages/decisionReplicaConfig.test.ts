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
          max_num_seqs: 8,
          public_name: 'decider',
        },
        specialist: { provider: 'model_runtime', artifact: 'other/model', device: 'cpu' },
      },
      system: { decision_model: { deployment: 'primary' } },
      bindings: { jailbreak: 'specialist' },
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
      bindings: { jailbreak: 'specialist' },
    })
    const deployment = decisionRuntimeDeclarations(next).primary
    expect(deployment).toEqual({
      provider: 'model_runtime',
      artifact: 'vllm-sr/Vela-2.0-4B',
      revision: 'a'.repeat(40),
      max_num_seqs: 8,
      public_name: 'decider',
      replicas: [{ device: 'rocm:0' }, { device: 'rocm:1' }],
    })
    expect(decisionRuntimeDeclarations(original).primary.device).toBe('rocm:0')
  })

  it('moves an attached served model onto each placement as model', () => {
    const current = snapshot()
    const deployment = decisionRuntimeDeclarations(current).primary
    delete deployment.device
    delete deployment.max_num_seqs
    deployment.endpoint = 'https://worker:8100'
    deployment.model = 'upstream-id'
    deployment.auth = { api_key_env: 'WORKER_KEY' }
    const replicas = deploymentPlacements(deployment)
    expect(replicas).toEqual([{ endpoint: 'https://worker:8100', model: 'upstream-id' }])
    const next = withDecisionReplicas(current, 'primary', [
      ...replicas,
      { endpoint: 'https://worker2:8100', model: 'upstream-id' },
    ])
    expect(decisionRuntimeDeclarations(next).primary).not.toHaveProperty('model')
    expect(decisionRuntimeDeclarations(next).primary.auth).toEqual({ api_key_env: 'WORKER_KEY' })
    expect(decisionRuntimeDeclarations(next).primary.replicas).toEqual([
      { endpoint: 'https://worker:8100', model: 'upstream-id' },
      { endpoint: 'https://worker2:8100', model: 'upstream-id' },
    ])
  })

  it('reads an attached model and writes it on each replica', () => {
    const current = snapshot()
    const deployment = decisionRuntimeDeclarations(current).primary
    delete deployment.device
    deployment.endpoint = 'http://worker:8100'
    deployment.model = 'upstream-id'
    const replicas = deploymentPlacements(deployment)
    expect(replicas).toEqual([{ endpoint: 'http://worker:8100', model: 'upstream-id' }])
    const next = withDecisionReplicas(current, 'primary', [
      ...replicas,
      { endpoint: 'http://worker2:8100', model: 'another-id' },
    ])
    expect(decisionRuntimeDeclarations(next).primary).not.toHaveProperty('endpoint')
    expect(decisionRuntimeDeclarations(next).primary).not.toHaveProperty('model')
    expect(decisionRuntimeDeclarations(next).primary.replicas).toEqual([
      { endpoint: 'http://worker:8100', model: 'upstream-id' },
      { endpoint: 'http://worker2:8100', model: 'another-id' },
    ])
    expect(
      deploymentPlacements({ replicas: [{ endpoint: 'unix:///run/worker.sock', model: 'm' }] }),
    ).toEqual([{ endpoint: 'unix:///run/worker.sock', model: 'm' }])
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
