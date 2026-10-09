import { describe, expect, it } from 'vitest'
import {
  isSystemOneRoutes,
  systemOneTargetOptions,
  systemOneTargetRequest,
  systemOneTargets,
  systemOneTargetTimeout,
} from './systemOneTargets'

const routes = {
  available: true,
  routes: [
    {
      model: 'vllm-sr/auto',
      recipe: 'typed',
      algorithms: ['cascade'],
      question_types: ['choice', 'score', 'noul'],
      timeout_ms: 120000,
    },
  ],
}
const deployment = {
  id: 'vllm-sr/auto',
  model: 'Decision 2.0',
  ready: true,
  question_types: ['choice'],
  surfaces: ['decisions'],
}

describe('System One target contract', () => {
  it('separates API entrypoints from similarly named deployments and groups labels once', () => {
    const targets = systemOneTargets([deployment], routes)
    expect(new Set(targets.map((target) => target.key)).size).toBe(2)
    const options = systemOneTargetOptions(targets)
    expect(options.map((option) => option.group)).toEqual(['Automatic routes', 'Direct models'])
    expect(options[0].label).toBe('vllm-sr/auto')
    expect(options[0].description).not.toContain('vllm-sr/auto')
    expect(systemOneTargetRequest(targets[0], { state: 'hello' })).toEqual({
      path: '/api/decision-model/routes',
      body: { model: 'vllm-sr/auto', request: { state: 'hello' } },
    })
    expect(systemOneTargetRequest(targets[1], {})).toEqual({
      path: '/api/decision-model/test',
      body: { deployment: 'vllm-sr/auto', request: {} },
    })
  })
  it('uses the recipe budget with transport grace, without timer overflow', () => {
    const targets = systemOneTargets([deployment], routes)
    expect(systemOneTargetTimeout(targets[0])).toBe(130000)
    expect(systemOneTargetTimeout(targets[1])).toBe(40000)
    expect(systemOneTargetTimeout({ ...targets[0], timeout_ms: 1e15 })).toBe(2147483647)
  })
  it('keeps direct models when route discovery is unavailable and rejects malformed deadlines', () => {
    expect(systemOneTargets([deployment], null)).toHaveLength(1)
    expect(systemOneTargets([deployment], { ...routes, available: false })).toHaveLength(1)
    expect(isSystemOneRoutes(routes)).toBe(true)
    expect(
      isSystemOneRoutes({ ...routes, routes: [{ ...routes.routes[0], timeout_ms: '120000' }] }),
    ).toBe(false)
    expect(
      isSystemOneRoutes({ ...routes, routes: [{ ...routes.routes[0], timeout_ms: Infinity }] }),
    ).toBe(false)
  })
})
