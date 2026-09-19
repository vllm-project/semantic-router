import { describe, expect, it } from 'vitest'
import { decodeDashboardSettings } from './dashboardSettings'

const readySettings = {
  readonlyMode: false,
  serverReadonly: false,
  runtimeConfigWritable: true,
  recipeStoreWritable: true,
  setupMode: false,
  platform: 'amd',
  envoyUrl: 'http://envoy',
  routerEvalEndpoint: 'http://router/api/v1/routing/preview',
  srBenchAvailable: true,
  srBenchUnavailableReason: '',
}

describe('decodeDashboardSettings', () => {
  it('accepts the exact current settings contract', () => {
    expect(decodeDashboardSettings(readySettings)).toEqual(readySettings)
  })

  it.each([
    ['legacy readonly-only response', { readonlyMode: false }],
    ['missing split capability', { ...readySettings, runtimeConfigWritable: undefined }],
    ['missing Evaluation availability', { ...readySettings, srBenchAvailable: undefined }],
    ['wrong Evaluation reason type', { ...readySettings, srBenchUnavailableReason: null }],
    ['array payload', []],
  ])('rejects %s instead of inferring authority', (_label, payload) => {
    expect(() => decodeDashboardSettings(payload)).toThrow()
  })
})
