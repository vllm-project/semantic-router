import { describe, expect, it } from 'vitest'
import {
  decisionActivationLabel,
  decisionModelRuntimeState,
  readDecisionModelApplyResult,
  type ModelRuntimeDeployment,
} from './decisionModelManagement'

const deployment: ModelRuntimeDeployment = {
  name: 'primary',
  managed: true,
  process: 'model-runtime-1',
  served_name: 'vela',
  ready: true,
  state: 'ready',
  restarts: 0,
}

describe('decision model deployment management', () => {
  it.each(['restart_required', 'persisted'] as const)(
    'keeps a 202 %s result distinct from an applied configuration',
    async (status) => {
      const result = await readDecisionModelApplyResult(
        new Response(
          JSON.stringify({ status, message: 'Activation requires an operator action.' }),
          { status: 202 },
        ),
      )
      expect(result).toEqual({ status, message: 'Activation requires an operator action.' })
    },
  )

  it('does not claim model readiness from a successful config write', async () => {
    const result = await readDecisionModelApplyResult(
      new Response('{"status":"success"}', { status: 200 }),
    )
    expect(result.status).toBe('success')
    expect(result.message).toContain('Check the observed runtime')
    await expect(
      readDecisionModelApplyResult(new Response('{"status":"success"}', { status: 202 })),
    ).rejects.toThrow('did not confirm activation')
    await expect(
      readDecisionModelApplyResult(new Response('{"error":"GPU unavailable"}', { status: 500 })),
    ).rejects.toThrow('GPU unavailable')
  })

  it('uses the exact live resource, even when another deployment has the same artifact', () => {
    expect(decisionModelRuntimeState('primary', { deployments: [deployment] })).toBe('Ready')
    expect(decisionModelRuntimeState('another', { deployments: [deployment] })).toBe('Not reported')
    expect(
      decisionModelRuntimeState('primary', {
        deployments: [
          { ...deployment, ready: false, state: 'failed' },
          { ...deployment, name: 'another' },
        ],
      }),
    ).toBe('Needs attention')
    expect(decisionModelRuntimeState('primary', null)).toBe('Not reported')
  })

  it('distinguishes pending, rejected, restart-required and observed active configuration', () => {
    expect(decisionActivationLabel(null)).toBe('Not reported')
    expect(decisionActivationLabel({ activation_status: 'pending' })).toBe('Activation pending')
    expect(
      decisionActivationLabel({
        activation_status: 'active',
        generated_runtime_hash: 'a',
        active_runtime_hash: 'different-observation',
      }),
    ).toBe('Active')
    expect(
      decisionActivationLabel({
        activation: {
          status: 'rejected',
          reasons: [{ code: 'restart_required', message: 'Model must restart.' }],
        },
      }),
    ).toBe('Restart required')
    expect(
      decisionActivationLabel({
        activation: {
          status: 'failed',
          reasons: [{ code: 'prepare_failed', message: 'GPU unavailable.' }],
        },
      }),
    ).toBe('Activation failed')
  })
})
