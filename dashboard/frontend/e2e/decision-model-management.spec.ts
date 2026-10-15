import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

async function mockDecisionModelManager(
  page: Page,
  options: {
    readonly?: boolean
    defaultModel?: boolean
    metrics?: 'reported' | 'empty' | 'unavailable' | 'partial'
    engine?: boolean
    lowTraffic?: boolean
    applyStatus?: 'success' | 'restart_required' | 'persisted' | 'failed'
  } = {},
) {
  await mockAuthenticatedAppShell(page, {
    settings: { platform: 'amd', readonlyMode: Boolean(options.readonly) },
    ...(options.readonly
      ? {
          user: {
            id: 'reader',
            email: 'reader@example.com',
            name: 'Reader',
            role: 'read',
            permissions: ['config.read'],
          },
        }
      : {}),
  })
  const global = {
    model_catalog: {
      system: (options.defaultModel
        ? {}
        : { decision_model: { deployment: 'primary' }, pii_classifier: 'models/custom-pii' }) as {
        decision_model?: { deployment: string }
        pii_classifier?: string
      },
      deployments: (options.defaultModel
        ? {}
        : {
            primary: {
              provider: 'model_runtime',
              artifact: 'vllm-sr/Vela-2.0-4B',
              device: 'rocm:0',
            },
          }) as Record<string, { provider: string; artifact: string; device?: string }>,
    },
  }
  let observed = options.defaultModel ? 'Vela-2.0-0.3B' : 'Vela-2.0-4B'
  let observedDeployment = 'primary'
  const config = {
    version: 'v0.3',
    global,
    providers: { models: [] },
    routing: {
      signals: options.defaultModel
        ? {}
        : { decision: [{ name: 'task', question: { type: 'noul' } }] },
      decisions: [],
    },
  }
  let pending = false
  let revision = 'active-config'
  const requests: unknown[] = []
  const reply = (data: unknown) => ({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(data),
  })
  const metricsRequests: string[] = []
  const metricsRanges: URLSearchParams[] = []
  let metricsMode = options.metrics
  await page.route('**/embedded/prometheus/api/v1/query_range?*', (route) => {
    const params = new URL(route.request().url()).searchParams
    const query = params.get('query') ?? ''
    metricsRequests.push(query)
    metricsRanges.push(params)
    if (
      metricsMode === 'unavailable' ||
      (metricsMode === 'partial' && query.includes('outcome!="ok"'))
    )
      return route.fulfill({ status: 503, body: 'Prometheus unavailable' })
    const value = query.includes('histogram_quantile')
      ? '0.125'
      : query.includes('phase="forward"')
        ? '0.010'
        : query.includes('result="hit"')
          ? '75'
          : query.includes('outcome!="ok"')
            ? '0'
            : query.includes('duration_seconds_sum')
              ? '0.025'
              : options.lowTraffic
                ? '0.02'
                : '2.5'
    const start = Number(params.get('start'))
    const end = Number(params.get('end'))
    const step = Number(params.get('step'))
    const values = []
    for (let timestamp = start; timestamp <= end; timestamp += step) {
      const variation = timestamp === end ? 1 : 1 + Math.sin((timestamp - start) / step / 8) * 0.15
      values.push([timestamp, String(Number(value) * variation)])
    }
    return route.fulfill(
      reply({
        status: 'success',
        data: {
          resultType: 'matrix',
          result:
            metricsMode === 'empty'
              ? []
              : [
                  { metric: { deployment: observedDeployment }, values },
                  { metric: { deployment: 'backend-llm' }, values: [[end, '999']] },
                ],
        },
      }),
    )
  })
  await page.route('**/api/router/config/global', (route) => route.fulfill(reply(global)))
  await page.route('**/api/router/config/all', (route) => route.fulfill(reply(config)))
  await page.route('**/api/instance', (route) =>
    route.fulfill(
      reply({
        ownership: 'managed',
        controller_available: false,
        observed_mode: options.engine ? 'engine' : 'router',
        active_deployment: observedDeployment,
        model: observed,
      }),
    ),
  )
  await page.route('**/api/decision-model/tasks', (route) =>
    route.fulfill(
      reply({
        default_deployment: global.model_catalog.system.decision_model?.deployment ?? 'primary',
        tasks: [{ id: 'pii_spans', title: 'PII extraction' }],
        deployments: [],
        bindings: options.defaultModel
          ? []
          : [
              {
                task_id: 'pii_spans',
                consumer: 'pii',
                recipe: '',
                deployment: 'pii-specialist',
                model: 'models/custom-pii',
                source: 'module',
                ready: true,
                editable: false,
                path: [],
                binding: { deployment: 'pii-specialist', contract: 'token_spans.v1' },
              },
            ],
      }),
    ),
  )
  await page.route('**/api/status', (route) =>
    route.fulfill(
      reply({
        overall: 'healthy',
        deployment_type: 'docker',
        serving_mode: options.engine ? 'engine' : 'router',
        services: [],
        models: {
          models: [
            {
              name: 'task',
              recipe: 'default',
              type: 'decision',
              loaded: true,
              state: 'ready',
              metadata: {
                provider: 'model_runtime',
                deployment: observedDeployment,
                resource_id: 'shared-vela',
                device: 'rocm:0',
              },
            },
          ],
        },
      }),
    ),
  )
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill(
      reply({
        deployments: [
          {
            name: observedDeployment,
            managed: true,
            process: 'model-runtime-1',
            served_name: 'vela',
            ready: true,
            state: 'ready',
            restarts: 2,
            repo: `vllm-sr/${observed}`,
            revision: 'revision-123',
            device: 'rocm:0',
            engine: 'torch',
            profile: 'exact',
            family: 'vela2',
            surfaces: ['question'],
            heads: [{ name: 'safety', kind: 'classification' }],
          },
        ],
      }),
    ),
  )
  await page.route('**/api/router/api/v1/config/hash', (route) =>
    route.fulfill(
      reply({
        generated_runtime_hash: pending ? 'new-config' : revision,
        active_runtime_hash: revision,
        activation_status: pending ? 'pending' : 'active',
        ...(pending
          ? {
              activation: {
                status: 'rejected',
                reasons: [
                  {
                    code: 'restart_required',
                    path: 'global.model_catalog.system.decision_model',
                    message: 'The model runtime needs a restart.',
                  },
                ],
              },
            }
          : {}),
      }),
    ),
  )
  await page.route('**/api/router/config/update', async (route) => {
    const patch = route.request().postDataJSON()
    requests.push(patch)
    global.model_catalog = patch.global.model_catalog
    const status = options.applyStatus ?? 'restart_required'
    pending = status !== 'success'
    if (status === 'failed') {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: '{"error":"Model preparation failed: GPU memory unavailable."}',
      })
    } else {
      await route.fulfill({
        status: status === 'success' ? 200 : 202,
        contentType: 'application/json',
        body: JSON.stringify({
          status,
          message:
            status === 'persisted'
              ? 'Saved to ConfigMap; roll out Router and Envoy.'
              : status === 'restart_required'
                ? 'Run vllm-sr serve to activate the saved model.'
                : undefined,
        }),
      })
    }
  })
  return {
    requests,
    config,
    changeExternally: (model: string, nextRevision: string) => {
      const deployment = model.toLowerCase().replace(/[^a-z0-9]+/g, '-')
      global.model_catalog.deployments[deployment] = {
        provider: 'model_runtime',
        artifact: `vllm-sr/${model}`,
      }
      global.model_catalog.system.decision_model = { deployment }
      observedDeployment = deployment
      observed = model
      revision = nextRevision
    },
    metricsRequests,
    metricsRanges,
    setMetrics: (mode: typeof metricsMode) => {
      metricsMode = mode
    },
    failMetrics: () => {
      metricsMode = 'unavailable'
    },
    activate: () => {
      observedDeployment = global.model_catalog.system.decision_model?.deployment ?? 'primary'
      observed =
        global.model_catalog.deployments[observedDeployment]?.artifact.split('/').at(-1) ??
        'Vela-2.0-0.3B'
      pending = false
    },
  }
}

test.describe('System One model management and monitoring', () => {
  test('shows measured runtime statistics and clears them when observation fails', async ({
    page,
  }) => {
    const fixture = await mockDecisionModelManager(page)
    await page.goto('/decision-model/monitoring')
    const deployment = page.getByRole('article', { name: 'primary', exact: true })
    const metric = (name: string) =>
      deployment
        .locator('dl[aria-label="Model statistics"] > div')
        .filter({ has: page.getByText(name, { exact: true }) })
    await expect(metric('Runtime calls / sec')).toContainText('2.50')
    await expect(metric('Unsuccessful calls')).toContainText('0.0%')
    await expect(metric('Mean call latency')).toContainText('25.0 ms')
    await expect(metric('P95 call latency')).toContainText('125.0 ms')
    await expect(metric('Result cache hit rate')).toContainText('75.0%')
    await expect(metric('Mean model forward')).toContainText('10.0 ms')
    await expect(deployment).not.toContainText('999')
    fixture.failMetrics()
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect(metric('Runtime calls / sec')).toContainText('Not reported')
    await expect(page.getByText(/Some model statistics are unavailable/)).toBeVisible()
    await expect(page.getByRole('link', { name: 'Decision Models', exact: true })).toHaveAttribute(
      'href',
      '/decision-model',
    )
  })

  test('keeps the default 0.3B model and shows absent samples as unknown without empty configuration clutter', async ({
    page,
  }) => {
    await mockDecisionModelManager(page, { defaultModel: true, metrics: 'empty' })
    await page.goto('/decision-model')
    await expect(page.getByRole('radio', { name: /Vela 2.0 0.3B/ })).toBeChecked()
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'No active task bindings in this configuration.',
    )
    await expect(page.getByRole('heading', { name: 'Custom model assignments' })).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Custom questions' })).toHaveCount(0)
    await expect(page.getByText('Default routing', { exact: true })).toHaveCount(0)
    await expect(page.getByText(/No custom questions configured/)).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Explicit signal overrides' })).toHaveCount(0)
    await page.getByRole('link', { name: 'Decision Monitoring', exact: true }).click()
    const stats = page
      .getByRole('article', { name: 'primary', exact: true })
      .locator('dl[aria-label="Model statistics"]')
    await expect(stats.getByText('Not reported', { exact: true })).toHaveCount(6)
  })

  test('changes real monitoring windows, handles partial failure, and recovers from missing observations', async ({
    page,
  }) => {
    const fixture = await mockDecisionModelManager(page, { lowTraffic: true })
    await page.goto('/decision-model/monitoring')
    const stats = page.locator('dl[aria-label="Model statistics"]')
    await expect(stats).toContainText('0.02')
    await expect(
      page.getByRole('region', { name: 'Traffic & reliability' }).locator('svg.recharts-surface'),
    ).toHaveCount(1)
    await expect(
      page.getByRole('region', { name: 'Latency breakdown' }).locator('svg.recharts-surface'),
    ).toHaveCount(1)
    await expect(
      page.getByRole('region', { name: 'Result cache efficiency' }).locator('svg.recharts-surface'),
    ).toHaveCount(1)
    const rateTicks = page
      .getByRole('region', { name: 'Traffic & reliability' })
      .locator('.recharts-yAxis')
      .first()
      .locator('.recharts-cartesian-axis-tick-value')
    await expect
      .poll(async () => new Set(await rateTicks.allTextContents()).size)
      .toBeGreaterThan(1)
    const windows = page.getByRole('group', { name: 'Monitoring time range' })
    await expect(windows.getByRole('button', { name: '1h' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    const checkRange = (seconds: number) => {
      const params = fixture.metricsRanges.at(-1)!
      expect(Number(params.get('end')) - Number(params.get('start'))).toBe(seconds)
      expect(seconds / Number(params.get('step')) + 1).toBeLessThanOrEqual(181)
    }
    checkRange(3600)
    fixture.setMetrics('partial')
    await windows.getByRole('button', { name: '6h' }).click()
    await expect(page.getByText(/Unavailable: Unsuccessful calls/)).toBeVisible()
    await expect(stats).toContainText('0.02')
    await expect(stats.getByText('Not reported', { exact: true })).toHaveCount(1)
    checkRange(21600)
    fixture.setMetrics('empty')
    await windows.getByRole('button', { name: '15m' }).click()
    await expect(stats.getByText('Not reported', { exact: true })).toHaveCount(6)
    await expect(
      page
        .getByRole('region', { name: 'Traffic & reliability' })
        .getByText('No samples in this window'),
    ).toBeVisible()
    checkRange(900)
    fixture.setMetrics('reported')
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect(stats).toContainText('0.02')
    await expect(stats.getByText('Not reported', { exact: true })).toHaveCount(0)
    await expect(
      page.getByRole('link', { name: 'Decision Playground', exact: true }),
    ).toHaveAttribute('href', '/decision-model/playground')
  })

  test('keeps charts and model controls usable on a narrow viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockDecisionModelManager(page)
    await page.goto('/decision-model/monitoring')
    await expect(page.locator('dl[aria-label="Model statistics"]')).toContainText('2.50')
    await expect(page.getByRole('button', { name: '6h', exact: true })).toBeVisible()
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await page.getByRole('region', { name: 'Latency breakdown' }).scrollIntoViewIfNeeded()
    await expect(page.getByRole('region', { name: 'Latency breakdown' })).toBeInViewport()
    await page.getByRole('link', { name: 'Decision Models', exact: true }).click()
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
  })

  test('isolates stalled monitoring requests from model management', async ({ page }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    let queries = 0
    await page.route('**/embedded/prometheus/api/v1/query_range?*', () => {
      queries += 1
    })
    await page.goto('/decision-model')
    await expect(page.getByRole('radio', { name: /Vela 2.0 9B/ })).toBeEnabled()
    expect(queries).toBe(0)
    await page.getByRole('link', { name: 'Decision Monitoring', exact: true }).click()
    await expect.poll(() => queries).toBeGreaterThanOrEqual(6)
    await page.clock.fastForward(8_100)
    await expect(page.getByText(/Some model statistics are unavailable/)).toBeVisible()
    await page.getByRole('link', { name: 'Decision Models', exact: true }).click()
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect.poll(() => fixture.requests.length).toBe(1)
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
  })

  test('renders saved controls before slow ancillary reads and avoids polling static configuration', async ({
    page,
  }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    let globalReads = 0
    let configReads = 0
    await page.route('**/api/router/config/global', async (route) => {
      globalReads += 1
      await route.fallback()
    })
    await page.route('**/api/router/config/all', () => {
      configReads += 1
    })
    await page.route('**/api/router/api/v1/inventory/model-runtime', () => {})
    await page.goto('/decision-model')
    await expect(page.getByRole('heading', { name: 'Decision Models', exact: true })).toBeVisible()
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
    expect(globalReads).toBe(1)
    expect(configReads).toBe(1)
    expect(fixture.metricsRequests).toHaveLength(0)
    await page.clock.fastForward(10_100)
    await expect(page.getByText('Runtime unavailable.', { exact: false })).toContainText(
      'The request timed out. Please retry.',
    )
  })

  test('polls runtime state without repeatedly loading saved configuration', async ({ page }) => {
    await page.clock.install()
    await mockDecisionModelManager(page)
    let globalReads = 0
    let configReads = 0
    let inventoryReads = 0
    await page.route('**/api/router/api/v1/inventory/model-runtime', async (route) => {
      inventoryReads += 1
      if (inventoryReads === 1) {
        await route.fulfill({ status: 503, body: 'Runtime temporarily unavailable' })
      } else {
        await route.fallback()
      }
    })
    await page.route('**/api/router/config/global', async (route) => {
      globalReads += 1
      await route.fallback()
    })
    await page.route('**/api/router/config/all', async (route) => {
      configReads += 1
      await route.fallback()
    })
    await page.goto('/decision-model')
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await expect(page.getByText('Runtime unavailable.', { exact: false })).toContainText(
      'Observation unavailable (HTTP 503).',
    )
    await page.clock.fastForward(30_100)
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await expect(
      page.getByText('Observation unavailable (HTTP 503).', { exact: false }),
    ).toHaveCount(0)
    expect(inventoryReads).toBeGreaterThan(1)
    expect(globalReads).toBe(1)
    expect(configReads).toBe(1)
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect.poll(() => globalReads).toBe(2)
    expect(configReads).toBe(2)
  })

  test('monitoring loads runtime observations without configuration requests', async ({ page }) => {
    await mockDecisionModelManager(page)
    const configRequests: string[] = []
    page.on('request', (request) => {
      const path = new URL(request.url()).pathname
      if (path === '/api/router/config/all' || path === '/api/router/config/global')
        configRequests.push(path)
    })
    await page.goto('/decision-model/monitoring')
    await expect(
      page.getByRole('heading', { name: 'Decision Monitoring', exact: true }),
    ).toBeVisible()
    await expect(page.locator('dl[aria-label="Model statistics"]')).toContainText('2.50')
    expect(configRequests).toEqual([])
  })

  test('refreshes externally changed saved configuration while preserving an unsaved selection', async ({
    page,
  }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    let globalReads = 0
    let configReads = 0
    await page.route('**/api/router/config/global', async (route) => {
      globalReads += 1
      await route.fallback()
    })
    await page.route('**/api/router/config/all', async (route) => {
      configReads += 1
      await route.fallback()
    })
    await page.goto('/decision-model')
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    expect([globalReads, configReads]).toEqual([1, 1])
    fixture.changeExternally('Vela-2.0-9B', 'external-change-1')
    await page.clock.fastForward(10_100)
    const status = page
      .getByRole('region')
      .filter({ has: page.getByText('Current deployment', { exact: true }) })
    await expect(status).toContainText('Vela-2.0-9B')
    await expect(page.getByRole('radio', { name: /Vela 2.0 9B/ })).toBeChecked()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    expect([globalReads, configReads]).toEqual([2, 2])

    await page.getByRole('radio', { name: /Vela 2.0 0.3B/ }).check()
    fixture.changeExternally('Vela-2.0-4B', 'external-change-2')
    await page.clock.fastForward(10_100)
    await expect(status).toContainText('Vela-2.0-4B')
    await expect(page.getByRole('radio', { name: /Vela 2.0 0.3B/ })).toBeChecked()
    await expect(page.getByText('Selected · not deployed', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    expect([globalReads, configReads]).toEqual([3, 3])
    await page.clock.fastForward(30_100)
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    expect([globalReads, configReads]).toEqual([3, 3])
    expect(fixture.requests).toEqual([])
  })

  test('bounds stalled status reads before and after a deployment request', async ({ page }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    let stallInventory = false
    let stalledReads = 0
    await page.route('**/api/router/api/v1/inventory/model-runtime', async (route) => {
      if (stallInventory) {
        stalledReads += 1
        return
      }
      await route.fallback()
    })
    await page.goto('/decision-model')
    await expect(page.getByRole('radio', { name: /Vela 2.0 9B/ })).toBeEnabled()
    stallInventory = true
    await page.clock.fastForward(10_100)
    await expect.poll(() => stalledReads).toBe(1)

    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect.poll(() => fixture.requests.length).toBe(1)
    await page.clock.fastForward(10_100)
    await expect.poll(() => stalledReads).toBe(2)
    await page.clock.fastForward(10_100)
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
    await expect(
      page.getByRole('status').filter({ hasText: /^Runtime last observed/ }),
    ).toContainText('The request timed out. Please retry.')
    await expect(page.getByText('Restart required', { exact: true })).toBeVisible()
  })

  test('opens from System One and keeps management and runtime details on separate pages', async ({
    page,
  }) => {
    await mockDecisionModelManager(page)
    await page.goto('/status')
    await page.getByRole('button', { name: 'Build', exact: true }).click()
    const menu = page.getByRole('navigation', { name: 'Build' })
    await menu.getByRole('tab', { name: /System One/ }).click()
    await menu.getByRole('link', { name: 'Decision Models', exact: true }).click()
    await expect(page).toHaveURL('/decision-model')
    await expect(page.getByRole('heading', { name: 'Decision Models', exact: true })).toBeVisible()
    await expect(page.getByRole('radio')).toHaveCount(6)
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    const status = page
      .getByRole('region')
      .filter({ has: page.getByText('Current deployment', { exact: true }) })
    await expect(status.getByText('Ready', { exact: true })).toBeVisible()
    await expect(status).toContainText('Router mode')
    const bindings = page.getByRole('region', { name: 'Task bindings' })
    await expect(bindings).toContainText('PII extraction')
    await expect(bindings).toContainText('models/custom-pii')
    await expect(bindings).toContainText('Specialist binding')
    await page.getByRole('link', { name: 'Decision Monitoring', exact: true }).click()
    const deployments = page.getByRole('region', { name: 'Model runtime deployments' })
    await deployments.getByText('Deployment details', { exact: true }).click()
    for (const value of ['revision-123', 'rocm:0', 'torch', 'exact', 'model-runtime-1', 'safety']) {
      await expect(deployments).toContainText(value)
    }
  })

  test('keeps saved and observed models separate until a restart is observed', async ({ page }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    await page.goto('/decision-model')
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect(page.getByText('Restart required', { exact: true })).toBeVisible()
    expect(fixture.requests).toEqual([
      {
        ...fixture.config,
        global: {
          model_catalog: {
            system: {
              decision_model: { deployment: 'vela-2-0-9b' },
              pii_classifier: 'models/custom-pii',
            },
            deployments: {
              primary: {
                provider: 'model_runtime',
                artifact: 'vllm-sr/Vela-2.0-4B',
                device: 'rocm:0',
              },
              'vela-2-0-9b': {
                provider: 'model_runtime',
                artifact: 'vllm-sr/Vela-2.0-9B',
                device: 'auto',
              },
            },
          },
        },
      },
    ])
    const status = page
      .getByRole('region')
      .filter({ has: page.getByText('Current deployment', { exact: true }) })
    await expect(status.getByRole('heading', { name: 'Vela-2.0-4B' })).toBeVisible()
    await status.getByText('Deployment diagnostics', { exact: true }).click()
    await expect(status).toContainText('Saved model: Vela 2.0 9B')
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'models/custom-pii',
    )
    fixture.activate()
    await page.clock.fastForward(10_100)
    await expect(status.getByText('Ready', { exact: true })).toBeVisible()
    await expect(status.getByRole('heading', { name: 'Vela-2.0-9B' })).toBeVisible()
    await expect(status).toContainText('Active')
  })

  test('shows a ConfigMap save as requiring rollout', async ({ page }) => {
    await mockDecisionModelManager(page, { applyStatus: 'persisted' })
    await page.goto('/decision-model')
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect(page.getByRole('status').filter({ hasText: 'Saved to ConfigMap' })).toContainText(
      'roll out Router and Envoy',
    )
    await expect(page.getByRole('heading', { name: 'Vela-2.0-4B' })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Vela-2.0-9B' })).toHaveCount(0)
  })

  test('retains the selected model and refreshes the saved configuration after apply fails', async ({
    page,
  }) => {
    await mockDecisionModelManager(page, { applyStatus: 'failed' })
    await page.goto('/decision-model')
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect(page.getByRole('alert')).toContainText('GPU memory unavailable')
    await expect(page.getByRole('radio', { name: /Vela 2.0 9B/ })).toBeChecked()
    await page.getByText('Deployment diagnostics', { exact: true }).click()
    await expect(page.getByText('Saved model: Vela 2.0 9B', { exact: true })).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Vela-2.0-4B' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
  })

  test('keeps management observable but disables writes for config readers', async ({ page }) => {
    const fixture = await mockDecisionModelManager(page, { readonly: true })
    await page.goto('/decision-model')
    await expect(page.getByRole('heading', { name: 'Decision Models', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeDisabled()
    await expect(page.getByRole('radio').first()).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await page.getByRole('link', { name: 'Decision Monitoring', exact: true }).click()
    await expect(
      page.getByText(/Viewing model statistics requires observability read access/),
    ).toBeVisible()
    expect(fixture.metricsRequests).toEqual([])
  })

  test('updates the decision model without changing standalone Engine mode', async ({ page }) => {
    const fixture = await mockDecisionModelManager(page, { engine: true })
    const modeWrites: string[] = []
    page.on('request', (request) => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/instance/deploy')
        modeWrites.push(request.url())
    })
    await page.goto('/decision-model')
    await expect(page.getByText('Engine mode', { exact: true })).toBeVisible()
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect.poll(() => fixture.requests.length).toBe(1)
    await expect(page.getByText('Engine mode', { exact: true })).toBeVisible()
    await expect(page.getByRole('group', { name: 'Instance mode', exact: true })).toHaveCount(0)
    expect(modeWrites).toEqual([])
    expect(fixture.metricsRequests).toEqual([])
  })
})

async function mockDecisionRuntimeCatalog(
  page: Page,
  options: {
    readonly?: boolean
    noConsumers?: boolean
    applyStatus?: 'success' | 'restart_required' | 'persisted' | 'failed'
  } = {},
) {
  const fixture = await mockDecisionModelManager(page, options)
  const config = {
    ...fixture.config,
    global: {
      model_catalog: {
        system: { decision_model: { deployment: 'primary' }, pii_classifier: 'models/custom-pii' },
        deployments: {
          primary: { provider: 'model_runtime', artifact: 'vllm-sr/Vela-2.0-4B', device: 'rocm:0' },
          existing: {
            provider: 'model_runtime',
            artifact: 'team/custom-decision-model',
            device: 'cpu',
          },
          lex: {
            provider: 'model_runtime',
            artifact: 'vllm-sr/Decision-1.0-Lex-0.6B',
            device: 'cpu',
          },
        } as Record<string, Record<string, unknown>>,
        admission: { shared: { max_inflight: 4 } },
      },
      services: { opaque_secret: 'test-preserved-secret' },
    },
    providers: { models: [{ name: 'answer', api_key: 'test-provider-secret' }] },
    routing: {
      signals: {
        decision: options.noConsumers
          ? []
          : [{ name: 'task', question: { type: 'noul', instructions: 'Is this coding?' } }],
      },
      decisions: [],
      replay: { opaque_future_flag: true },
    },
    recipes: [
      {
        name: 'research',
        routing: {
          model_bindings: {
            complexity: { deployment: 'existing', contract: 'decision.v1' },
            model_selection: { deployment: 'existing', contract: 'decision.v1' },
          } as Record<string, { deployment: string; contract: string }>,
          signals: {
            complexity: options.noConsumers ? [] : [{ name: 'difficulty', threshold: 0.5 }],
          },
          decisions: options.noConsumers
            ? []
            : [
                {
                  name: 'pick',
                  algorithm: {
                    type: 'decision',
                    decision: {
                      instructions: 'Which model?',
                      candidates: { answer: 'General model' },
                    },
                  },
                  modelRefs: [{ model: 'answer' }],
                },
              ],
        },
      },
    ],
    entrypoints: [{ model_names: ['research'], recipe: 'research' }],
  }
  const requests: (typeof config)[] = []
  let ready = false
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: config.global }))
  await page.route('**/api/router/config/all', (route) => route.fulfill({ json: config }))
  await page.route('**/api/router/config/update', async (route) => {
    const body = route.request().postDataJSON() as typeof config
    requests.push(body)
    if (options.applyStatus === 'failed')
      return route.fulfill({
        status: 500,
        body: 'Runtime preparation failed: unavailable GPU memory.',
      })
    Object.assign(config, body)
    return route.fulfill({
      status: options.applyStatus && options.applyStatus !== 'success' ? 202 : 200,
      json: {
        status: options.applyStatus ?? 'success',
        ...(options.applyStatus === 'restart_required'
          ? { message: 'Configuration saved. Restart the service to activate.' }
          : {}),
      },
    })
  })
  await page.route('**/api/decision-model/tasks', (route) => {
    const deployments = Object.entries(config.global.model_catalog.deployments).map(
      ([deployment, resource]) => ({
        deployment,
        model: resource.artifact,
        ready: deployment === 'primary' || ready,
        native_question_types: ['choice', 'score', 'noul'],
        tasks: ['complexity', 'model_selection'].map((task_id) => ({
          task_id,
          supported: !['embeddings', 'reranker'].includes(deployment),
          quality: 'unevaluated',
        })),
      }),
    )
    const bindings = options.noConsumers
      ? []
      : ['complexity', 'model_selection'].map((task_id) => {
          const binding = config.recipes[0]?.routing.model_bindings[task_id] ?? {
            deployment: 'primary',
            contract: 'decision.v1',
          }
          return {
            task_id,
            consumer: task_id === 'complexity' ? 'difficulty' : 'pick',
            recipe: 'research',
            deployment: binding.deployment,
            model: config.global.model_catalog.deployments[binding.deployment]?.artifact,
            source: 'recipe',
            ready: binding.deployment === 'primary' || ready,
            editable: true,
            path: ['recipes', '0', 'routing', 'model_bindings', task_id],
            binding,
          }
        })
    return route.fulfill({
      json: {
        default_deployment: config.global.model_catalog.system.decision_model.deployment,
        tasks: [
          { id: 'complexity', title: 'Task difficulty' },
          { id: 'model_selection', title: 'Choose a candidate model' },
        ],
        deployments,
        bindings,
      },
    })
  })
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill({
      json: {
        deployments: Object.entries(config.global.model_catalog.deployments)
          .filter(([name]) => name === 'primary' || ready)
          .map(([name, resource]) => ({
            name,
            repo: resource.artifact,
            ready: true,
            state: 'ready',
            managed: true,
            restarts: 0,
          })),
      },
    }),
  )
  return {
    ...fixture,
    config,
    runtimeRequests: requests,
    activateRuntime: () => {
      ready = true
    },
  }
}

test.describe('Decision model catalog and deployment bindings', () => {
  test('browses all released families with logos and filters capabilities without loading Model Hub', async ({
    page,
  }, testInfo) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    const catalogRequests: string[] = []
    page.on('request', (request) => {
      if (request.url().includes('/api/models/catalog')) catalogRequests.push(request.url())
    })
    await page.goto('/decision-model')
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    const pagination = page.getByRole('navigation', { name: 'Decision model pagination' })
    await expect(pagination).toContainText('of 17')
    const names = new Set<string>()
    for (let index = 0; index < 3; index++) {
      for (const value of await page
        .getByRole('radio')
        .evaluateAll((inputs) => inputs.map((input) => (input as HTMLInputElement).value)))
        names.add(value)
      if (index < 2) await pagination.getByRole('button', { name: 'Next', exact: true }).click()
    }
    expect(names.size).toBe(17)
    expect(names.has('Decision-2.0-Kai-0.6B')).toBe(true)
    expect(names.has('Decision-1.0-Lex-0.6B')).toBe(true)
    await page.getByRole('button', { name: 'Decision 1.0', exact: true }).click()
    await expect(pagination).toContainText('of 7')
    await page.getByLabel('Search models').fill('Lex')
    await expect(page.getByRole('radio')).toHaveCount(1)
    await expect(page.getByRole('radio', { name: /Decision 1.0 Lex 0.6B/ })).toBeVisible()
    await expect(page.getByRole('img', { name: 'vLLM Semantic Router', exact: true })).toBeVisible()
    await page.getByLabel('Search models').clear()
    await page.getByRole('button', { name: 'All families', exact: true }).click()
    await page.getByRole('combobox', { name: 'Question capability' }).click()
    await page.getByRole('option', { name: 'span', exact: true }).click()
    await expect(page.getByRole('radio')).toHaveCount(4)
    for (const radio of await page.getByRole('radio').all())
      await expect(radio).toHaveValue(/^Vela-/)
    await page.screenshot({
      path: testInfo.outputPath('decision-catalog-desktop.png'),
      fullPage: true,
    })
    expect(fixture.metricsRequests).toHaveLength(0)
    expect(catalogRequests).toHaveLength(0)
  })

  test('keeps unrelated embedding deployments out of the compatible task binding choices', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    Object.assign(fixture.config.global.model_catalog.deployments, {
      embeddings: { provider: 'model_runtime', artifact: 'team/embedding-model' },
      reranker: { provider: 'model_runtime', artifact: 'team/reranker-model' },
    })
    await page.goto('/decision-model')
    const row = page.getByRole('row').filter({ hasText: 'Task difficulty' })
    await expect(row).toContainText('team/custom-decision-model')
    await expect(row).toContainText('research · difficulty')
    await row.getByRole('button', { name: 'Change' }).click()
    await row.getByRole('combobox', { name: 'Model for Task difficulty' }).click()
    await expect(page.getByRole('option', { name: /team\/custom-decision-model/ })).toBeVisible()
    await expect(
      page.getByRole('option', { name: /team\/(embedding|reranker)-model/ }),
    ).toHaveCount(0)
    expect(fixture.runtimeRequests).toHaveLength(0)
  })

  test('atomically deploys a pinned Decision 2.0 default while preserving existing resources, recipe bindings and credentials', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    await page.goto('/decision-model')
    await page.getByLabel('Search models').fill('Decision 2.0 Kai')
    await page.getByRole('radio', { name: /Decision 2.0 Kai 0.6B/ }).check()
    fixture.config.global.services.opaque_secret = 'test-updated-secret'
    const previous = structuredClone(fixture.config)
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect(
      page.getByRole('status').filter({ hasText: 'Configuration applied.' }),
    ).toBeVisible()
    expect(fixture.runtimeRequests).toHaveLength(1)
    const payload = fixture.runtimeRequests[0]
    expect(payload).toEqual({
      ...previous,
      global: {
        ...previous.global,
        model_catalog: {
          ...previous.global.model_catalog,
          system: {
            decision_model: { deployment: 'decision-2-0-kai-0-6b' },
            pii_classifier: 'models/custom-pii',
          },
          deployments: {
            ...previous.global.model_catalog.deployments,
            'decision-2-0-kai-0-6b': {
              provider: 'model_runtime',
              artifact: 'vllm-sr/Decision-2.0-Kai-0.6B',
              revision: 'cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764',
              device: 'auto',
            },
          },
        },
      },
    })
    await expect(page.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
    const replicas = page.getByRole('region', { name: 'Model replicas' })
    await expect(replicas).toContainText('Not observed')
    fixture.activateRuntime()
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect(replicas).toContainText('ready')
  })

  test('saves a Decision 1.0 default without claiming a new runtime is ready when no tasks are bound', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, {
      noConsumers: true,
      applyStatus: 'restart_required',
    })
    await page.goto('/decision-model')
    await page.getByLabel('Search models').fill('Decision 1.0 Lex')
    await page.getByRole('radio', { name: /Decision 1.0 Lex 0.6B/ }).check()
    await page.getByRole('button', { name: 'Deploy selection' }).click()
    await expect(
      page.getByRole('status').filter({ hasText: 'Configuration saved. Restart' }),
    ).toBeVisible()
    expect(fixture.runtimeRequests[0].routing.signals.decision).toEqual([])
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'No active task bindings',
    )
    await expect(page.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Model replicas' })).toContainText('Not observed')
    await expect(
      page.getByRole('link', { name: 'Decision Playground', exact: true }),
    ).toHaveAttribute('href', '/decision-model/playground')
  })

  test('binds a decision selector task while preserving the default and exposes deferred activation honestly', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, { applyStatus: 'restart_required' })
    await page.goto('/decision-model')
    const row = page.getByRole('row').filter({ hasText: 'Choose a candidate model' })
    await row.getByRole('button', { name: 'Change' }).click()
    await row.getByRole('combobox', { name: 'Model for Choose a candidate model' }).click()
    await page.getByRole('option', { name: /Decision-1.0-Lex-0.6B/ }).click()
    const previous = structuredClone(fixture.config)
    await row.getByRole('button', { name: 'Apply', exact: true }).click()
    await expect(
      page.getByRole('status').filter({ hasText: 'Configuration saved. Restart' }),
    ).toBeVisible()
    expect(fixture.runtimeRequests).toHaveLength(1)
    previous.recipes[0].routing.model_bindings.model_selection = {
      deployment: 'lex',
      contract: 'decision.v1',
    }
    expect(fixture.runtimeRequests[0]).toEqual(previous)
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect(row).toContainText('vllm-sr/Decision-1.0-Lex-0.6B')
    await expect(row.getByText('Not ready', { exact: true })).toBeVisible()
    fixture.activateRuntime()
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect(row.getByText('Ready', { exact: true })).toBeVisible()
  })

  test('revalidates the recipe identity against a fresh snapshot before any task binding write', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    await page.goto('/decision-model')
    const row = page.getByRole('row').filter({ hasText: 'Task difficulty' })
    await row.getByRole('button', { name: 'Change' }).click()
    await row.getByRole('combobox', { name: 'Model for Task difficulty' }).click()
    await page.getByRole('option', { name: /Vela-2.0-4B/ }).click()
    fixture.config.recipes = []
    await row.getByRole('button', { name: 'Apply', exact: true }).click()
    await expect(row.getByRole('alert')).toContainText('This recipe was removed')
    expect(fixture.runtimeRequests).toHaveLength(0)
  })

  test('allows readonly model and task inspection without configuration mutation', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, { readonly: true })
    await page.goto('/decision-model')
    await page.getByLabel('Search models').fill('Decision 2.0 Kai')
    await expect(page.getByRole('radio', { name: /Decision 2.0 Kai 0.6B/ })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeDisabled()
    const bindings = page.getByRole('region', { name: 'Task bindings' })
    await expect(bindings).toContainText('Task difficulty')
    await expect(bindings.getByRole('button', { name: 'Change' })).toHaveCount(0)
    const replicas = page.getByRole('region', { name: 'Model replicas' })
    await replicas.getByText('Configure replicas', { exact: true }).click()
    await expect(replicas.getByLabel('Device', { exact: true })).toBeDisabled()
    await expect(replicas.getByRole('button', { name: 'Add replica' })).toBeDisabled()
    expect(fixture.runtimeRequests).toHaveLength(0)
  })

  test('keeps catalog and task binding controls usable at mobile width with keyboard dismissal', async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 })
    const fixture = await mockDecisionRuntimeCatalog(page)
    await page.goto('/decision-model')
    await page.getByRole('button', { name: 'Decision 2.0', exact: true }).click()
    await page.getByRole('radio', { name: /Decision 2.0 Kai 0.6B/ }).check()
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    const row = page.getByRole('row').filter({ hasText: 'Task difficulty' })
    await row.getByRole('button', { name: 'Change' }).click()
    const consumer = row.getByRole('combobox', { name: 'Model for Task difficulty' })
    await consumer.click()
    await page.keyboard.press('Escape')
    await expect(consumer).toBeFocused()
    await expect(consumer).toHaveAttribute('aria-expanded', 'false')
    await consumer.click()
    await page.mouse.wheel(0, -120)
    await expect(consumer).toHaveAttribute('aria-expanded', 'true')
    await page.getByRole('heading', { name: 'Task bindings', exact: true }).click()
    await expect(consumer).toHaveAttribute('aria-expanded', 'false')
    await consumer.focus()
    await page.keyboard.press('Home')
    await page.keyboard.press('Enter')
    await expect(consumer).toContainText('vllm-sr/Vela-2.0-4B')
    await consumer.click()
    await page.getByRole('option', { name: /Decision-1.0-Lex-0.6B/ }).click()
    await row.getByRole('button', { name: 'Apply', exact: true }).scrollIntoViewIfNeeded()
    await expect(row.getByRole('button', { name: 'Apply', exact: true })).toBeInViewport()
    await page.screenshot({
      path: testInfo.outputPath('decision-catalog-mobile.png'),
      fullPage: true,
    })
    await row.getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(consumer).toHaveCount(0)
    expect(fixture.runtimeRequests).toEqual([])
  })
})
