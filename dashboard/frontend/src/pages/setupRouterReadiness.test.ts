import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'
import type { ServiceStatus, SystemStatus } from '../utils/routerRuntime'
import { SetupSuccessView } from './SetupWizardReviewPanel'
import { describeSetupRouterReadiness, type SetupRouterReadiness } from './setupRouterReadiness'

const service = (
  name: string,
  status: string,
  healthy: boolean,
  message: string,
  component = 'container',
): ServiceStatus => ({ name, status, healthy, message, component })

const systemStatus = (
  services: ServiceStatus[],
  extra: Partial<SystemStatus> = {},
): SystemStatus => ({
  overall: services.every((entry) => entry.healthy) ? 'healthy' : 'degraded',
  deployment_type: 'docker',
  services,
  ...extra,
})

describe('describeSetupRouterReadiness', () => {
  it('is checking until the status answers', () => {
    expect(describeSetupRouterReadiness(null)).toEqual({ state: 'checking' })
  })

  it('is ready once routing access is healthy', () => {
    const status = systemStatus([
      service('Routing access', 'running', true, 'Ready', 'gateway'),
      service('Router', 'running', true, 'Running'),
      service('Dashboard', 'running', true, 'Running'),
    ])
    expect(describeSetupRouterReadiness(status).state).toBe('ready')
  })

  it('reports a Router that never started as stopped, with its status message', () => {
    // `vllm-sr serve` exited: no heartbeat, the activation is still pending.
    const status = systemStatus([
      service(
        'Routing access',
        'unavailable',
        false,
        'Router or gateway is unavailable',
        'gateway',
      ),
      service(
        'Router',
        'not running',
        false,
        'Not running: run `vllm-sr serve` to apply the saved configuration',
      ),
      service('Dashboard', 'running', true, 'Running'),
    ])
    expect(describeSetupRouterReadiness(status)).toEqual({
      state: 'stopped',
      detail: 'Not running: run `vllm-sr serve` to apply the saved configuration',
    })
  })

  it('is starting while serve brings the Router up, with the runtime progress', () => {
    const status = systemStatus(
      [
        service(
          'Routing access',
          'unavailable',
          false,
          'Router or gateway is unavailable',
          'gateway',
        ),
        service('Router', 'running', true, 'Running'),
        service('Dashboard', 'running', true, 'Running'),
      ],
      {
        router_runtime: {
          phase: 'downloading_models',
          ready: false,
          message: 'Downloading models',
        },
      },
    )
    expect(describeSetupRouterReadiness(status)).toEqual({
      state: 'starting',
      detail: 'Downloading models',
    })
  })

  it('is starting while the attached serve has not created the Router yet', () => {
    const status = systemStatus([
      service(
        'Routing access',
        'unavailable',
        false,
        'Router or gateway is unavailable',
        'gateway',
      ),
      service('Router', 'starting', false, 'Starting: `vllm-sr serve` is starting it'),
    ])
    expect(describeSetupRouterReadiness(status)).toEqual({
      state: 'starting',
      detail: 'Starting: `vllm-sr serve` is starting it',
    })
  })

  it('treats a stopped Envoy as not ready even when the Router answers', () => {
    const status = systemStatus([
      service(
        'Routing access',
        'unavailable',
        false,
        'Router or gateway is unavailable',
        'gateway',
      ),
      service('Router', 'running', true, 'Running'),
      service('Envoy', 'not running', false, 'Not running: run `vllm-sr serve`'),
    ])
    expect(describeSetupRouterReadiness(status).state).toBe('stopped')
  })
})

const renderSuccess = (readiness: SetupRouterReadiness, activationMessage?: string) =>
  renderToStaticMarkup(
    createElement(
      TooltipProvider,
      null,
      createElement(SetupSuccessView, {
        baseUrl: 'http://localhost:8899/v1',
        defaultModel: 'auto',
        summary: '1 model · From scratch',
        leaving: false,
        readiness,
        activationMessage,
        onTest: () => undefined,
        onDashboard: () => undefined,
      }),
    ),
  )

describe('SetupSuccessView', () => {
  it('claims the router is live only when it is ready', () => {
    const markup = renderSuccess({ state: 'ready' })
    expect(markup).toContain('Your router is live')
    expect(markup).toContain('Send a test request')
    expect(markup).toContain('http://localhost:8899/v1')
  })

  it('shows a starting state, not live, while the Router comes up', () => {
    const markup = renderSuccess({ state: 'checking' }, 'Setup saved. The Router is starting.')
    expect(markup).toContain('Starting your router')
    expect(markup).toContain('Setup saved. The Router is starting.')
    expect(markup).not.toContain('Your router is live')
    expect(markup).not.toContain('Send a test request')
  })

  it('shows why the Router is not running instead of claiming live', () => {
    const markup = renderSuccess({
      state: 'stopped',
      detail: 'Not running: run `vllm-sr serve` to apply the saved configuration',
    })
    expect(markup).toContain('Your router is not running')
    expect(markup).toContain('Not running: run `vllm-sr serve` to apply the saved configuration')
    expect(markup).not.toContain('Your router is live')
    expect(markup).not.toContain('Apps can call it now')
  })
})
