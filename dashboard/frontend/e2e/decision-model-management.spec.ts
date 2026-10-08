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
      system: options.defaultModel
        ? ({} as { decision_model?: string; pii_classifier?: string })
        : { decision_model: 'Vela-2.0-4B', pii_classifier: 'models/custom-pii' },
    },
  }
  let observed = options.defaultModel ? 'Vela-2.0-0.3B' : 'Vela-2.0-4B'
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
                  { metric: { deployment: `@${observed}/auto` }, values },
                  { metric: { deployment: 'backend-llm' }, values: [[end, '999']] },
                ],
        },
      }),
    )
  })
  await page.route('**/api/router/config/global', (route) => route.fulfill(reply(global)))
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill(
      reply({
        version: 'v0.3',
        global,
        providers: { models: [] },
        routing: {
          signals: options.defaultModel
            ? {}
            : { decision: [{ name: 'task', question: { type: 'noul' } }] },
          decisions: [],
        },
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
                deployment: `@${observed}/auto`,
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
            name: `@${observed}/auto`,
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
  await page.route('**/api/router/config/global/update', async (route) => {
    const patch = route.request().postDataJSON()
    requests.push(patch)
    global.model_catalog.system.decision_model = patch.model_catalog.system.decision_model
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
    changeExternally: (model: string, nextRevision: string) => {
      global.model_catalog.system.decision_model = model
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
      observed = global.model_catalog.system.decision_model ?? 'Vela-2.0-0.3B'
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
    const deployment = page.getByRole('article', { name: '@Vela-2.0-4B/auto', exact: true })
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
    await page.getByText('Advanced bindings', { exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Custom model assignments' })).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Custom questions' })).toHaveCount(0)
    await expect(page.getByText('Default routing', { exact: true })).toHaveCount(0)
    await expect(page.getByText(/No custom questions configured/)).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Explicit signal overrides' })).toHaveCount(0)
    await page.getByRole('link', { name: 'Decision Monitoring', exact: true }).click()
    const stats = page
      .getByRole('article', { name: '@Vela-2.0-0.3B/auto', exact: true })
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
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
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
    await page.getByRole('button', { name: 'Deploy selected model' }).click()
    await expect.poll(() => fixture.requests.length).toBe(1)
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
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
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
    expect(globalReads).toBe(1)
    expect(configReads).toBe(1)
    expect(fixture.metricsRequests).toHaveLength(0)
    await page.clock.fastForward(10_100)
    await expect(page.getByRole('alert')).toContainText('Status request timed out after 10 seconds')
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
    await expect(page.getByRole('alert')).toContainText('Runtime deployments')
    await page.clock.fastForward(30_100)
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await expect(page.getByRole('alert')).toHaveCount(0)
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
    const status = page.getByRole('region', { name: 'Deployment status' })
    await expect(status).toContainText('Vela-2.0-9B')
    await expect(page.getByRole('radio', { name: /Vela 2.0 9B/ })).toBeChecked()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    expect([globalReads, configReads]).toEqual([2, 2])

    await page.getByRole('radio', { name: /Vela 2.0 0.3B/ }).check()
    fixture.changeExternally('Vela-2.0-4B', 'external-change-2')
    await page.clock.fastForward(10_100)
    await expect(status).toContainText('Vela-2.0-4B')
    await expect(page.getByRole('radio', { name: /Vela 2.0 0.3B/ })).toBeChecked()
    await expect(page.getByText('Selected · not saved', { exact: true })).toBeVisible()
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
    await page.getByRole('button', { name: 'Deploy selected model' }).click()
    expect(fixture.requests).toHaveLength(0)
    await page.clock.fastForward(10_100)
    await expect.poll(() => fixture.requests.length).toBe(1)
    await expect.poll(() => stalledReads).toBe(2)
    await page.clock.fastForward(10_100)
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
    await expect(page.getByRole('alert')).toContainText('Status request timed out after 10 seconds')
    await expect(page.getByRole('status')).toContainText('Restart required')
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
    await expect(page.getByRole('radio')).toHaveCount(5)
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    const status = page.getByRole('region', { name: 'Deployment status' })
    await expect(status).toContainText('Matching runtime ready')
    await expect(status).not.toContainText('Serving mode')
    await expect(page.getByText('Custom model assignments', { exact: true })).not.toBeVisible()
    await page.getByText('Advanced bindings', { exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Custom questions' })).toBeVisible()
    await expect(page.locator('details[open]').last()).toContainText('task')
    await expect(page.getByText('Default routing', { exact: true })).toHaveCount(0)
    await expect(page.getByText(/\(noul\)/)).toHaveCount(0)
    await expect(
      page.getByRole('link', { name: /Manage advanced model bindings/ }),
    ).toHaveAttribute('href', '/config/global-config#global-section-system_models')
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
    await page.getByRole('button', { name: 'Deploy selected model' }).click()
    await expect(page.getByRole('status')).toContainText('Restart required')
    expect(fixture.requests).toEqual([
      { model_catalog: { system: { decision_model: 'Vela-2.0-9B' } } },
    ])
    const status = page.getByRole('region', { name: 'Deployment status' })
    await expect(status).toContainText('Vela-2.0-9B')
    await expect(status).not.toContainText('Matching runtime ready')
    await page.getByText('Advanced bindings', { exact: true }).click()
    await expect(page.getByText('models/custom-pii', { exact: true })).toBeVisible()
    fixture.activate()
    await page.clock.fastForward(10_100)
    await expect(status).toContainText('Matching runtime ready')
    await expect(status).toContainText('Active')
  })

  test('shows a ConfigMap save as requiring rollout', async ({ page }) => {
    await mockDecisionModelManager(page, { applyStatus: 'persisted' })
    await page.goto('/decision-model')
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selected model' }).click()
    await expect(page.getByRole('status')).toContainText('Saved; rollout required')
    await expect(page.getByRole('status')).toContainText('roll out Router and Envoy')
    await expect(page.getByRole('region', { name: 'Deployment status' })).not.toContainText(
      'Matching runtime ready',
    )
  })

  test('retains the selected model and refreshes the saved configuration after apply fails', async ({
    page,
  }) => {
    await mockDecisionModelManager(page, { applyStatus: 'failed' })
    await page.goto('/decision-model')
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selected model' }).click()
    await expect(page.getByRole('alert')).toContainText('GPU memory unavailable')
    await expect(page.getByRole('radio', { name: /Vela 2.0 9B/ })).toBeChecked()
    await expect(page.getByRole('region', { name: 'Deployment status' })).toContainText(
      'Vela-2.0-9B',
    )
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
  })

  test('keeps management observable but disables writes for config readers', async ({ page }) => {
    const fixture = await mockDecisionModelManager(page, { readonly: true })
    await page.goto('/decision-model')
    await expect(page.getByRole('heading', { name: 'Decision Models', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeDisabled()
    await expect(page.getByRole('radio').first()).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await page.getByRole('link', { name: 'Decision Monitoring', exact: true }).click()
    await expect(
      page.getByText(/Viewing model statistics requires observability read access/),
    ).toBeVisible()
    expect(fixture.metricsRequests).toEqual([])
  })

  test('does not apply router model settings to a standalone engine', async ({ page }) => {
    const fixture = await mockDecisionModelManager(page, { engine: true })
    await page.goto('/decision-model')
    await expect(page.getByText(/This deployment is a standalone engine/)).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeDisabled()
    await expect(page.getByRole('radio').first()).toBeDisabled()
    expect(fixture.requests).toEqual([])
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
    version: 'v0.3',
    global: {
      model_catalog: {
        system: { decision_model: 'Vela-2.0-4B', pii_classifier: 'models/custom-pii' },
        deployments: {} as Record<string, unknown>,
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
          signals: {
            decision: options.noConsumers
              ? []
              : [
                  {
                    name: 'difficulty',
                    deployment: 'existing',
                    question: {
                      type: 'score',
                      instructions: 'Difficulty?',
                      levels: ['easy', 'hard'],
                    },
                  },
                  {
                    name: 'entities',
                    question: {
                      type: 'span',
                      instructions: 'Extract',
                      labels: [{ key: 'person' }],
                    },
                  },
                ],
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
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill({
      json: {
        deployments: Object.keys(config.global.model_catalog.deployments)
          .filter(() => ready)
          .map((name) => ({
            name,
            ready: true,
            state: 'ready',
            managed: true,
            family: 'decision2',
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
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    const catalogRequests: string[] = []
    page.on('request', (request) => {
      if (request.url().includes('/api/models/catalog')) catalogRequests.push(request.url())
    })
    await page.goto('/decision-model')
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    await expect(page.getByRole('article')).toHaveCount(13)
    await page.screenshot({
      path: '../../.agent-harness/decision-model-dashboard/decision-catalog-desktop.png',
      fullPage: true,
    })
    await expect(
      page
        .getByRole('article', { name: 'Decision-2.0-Kai-0.6B', exact: true })
        .getByRole('img', { name: 'vLLM Semantic Router' }),
    ).toBeVisible()
    await page.getByRole('button', { name: /^Decision 1.0/ }).click()
    await expect(page.getByRole('article')).toHaveCount(7)
    await page.getByRole('textbox', { name: 'Search decision models' }).fill('Lex')
    await expect(page.getByRole('article')).toHaveCount(1)
    await expect(page.getByRole('article', { name: 'Decision-1.0-Lex-0.6B' })).toContainText(
      'ModernBERT',
    )
    await page.getByRole('textbox', { name: 'Search decision models' }).clear()
    await page.getByRole('button', { name: /^All families/ }).click()
    await page.getByRole('combobox', { name: 'Question capability' }).click()
    await page.getByRole('option', { name: 'span', exact: true }).click()
    await expect(page.getByRole('article')).toHaveCount(0)
    await expect(page.getByRole('radio')).toHaveCount(4)
    expect(fixture.metricsRequests).toHaveLength(0)
    expect(catalogRequests).toHaveLength(0)
  })

  test('keeps unrelated embedding deployments out of the decision runtime binding view', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    fixture.config.global.model_catalog.deployments = {
      embeddings: { provider: 'model_runtime', artifact: 'team/embedding-model' },
      reranker: { provider: 'model_runtime', artifact: 'team/reranker-model' },
      existing: { provider: 'model_runtime', artifact: 'team/custom-decision-model' },
      unbound: { provider: 'model_runtime', artifact: 'vllm-sr/Decision-2.0-Kai-0.6B' },
    }
    await page.goto('/decision-model')
    const configured = page.getByRole('region', { name: 'Configured runtimes' })
    await expect(configured).toContainText('team/custom-decision-model')
    await expect(configured).toContainText('research / difficulty')
    await expect(configured).toContainText('Saved · no decision binding')
    await expect(configured).not.toContainText('team/embedding-model')
    await expect(configured).not.toContainText('team/reranker-model')
    await expect(configured).not.toContainText('Bind a consumer to activate')
  })

  test('atomically deploys a pinned Decision 2.0 runtime for one recipe while preserving the Vela default and credentials', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    await page.goto('/decision-model')
    await page
      .getByRole('article', { name: 'Decision-2.0-Kai-0.6B', exact: true })
      .getByRole('button', { name: 'Configure' })
      .click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('textbox', { name: 'Deployment name' }).fill('research-decider')
    await dialog.getByRole('combobox', { name: 'Deployment consumer' }).click()
    await expect(dialog.getByRole('option', { name: /entities/ })).toHaveCount(0)
    await dialog.getByRole('option', { name: /research \/ difficulty/ }).click()
    await expect(dialog).toContainText('existing → research-decider')
    fixture.config.global.services.opaque_secret = 'test-updated-secret'
    await dialog.getByRole('button', { name: 'Deploy and bind model' }).click()
    await expect(dialog.getByRole('status')).toContainText('Configuration applied')
    expect(fixture.runtimeRequests).toHaveLength(1)
    const payload = fixture.runtimeRequests[0]
    expect(payload.global.model_catalog.system).toEqual({
      decision_model: 'Vela-2.0-4B',
      pii_classifier: 'models/custom-pii',
    })
    expect(payload.global.model_catalog.deployments['research-decider']).toEqual({
      provider: 'model_runtime',
      artifact: 'vllm-sr/Decision-2.0-Kai-0.6B',
      revision: 'cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764',
      device: 'auto',
    })
    expect(payload.recipes[0].routing.signals.decision[0].deployment).toBe('research-decider')
    expect(payload.routing.signals.decision[0]).not.toHaveProperty('deployment')
    expect(payload.recipes[0].routing.signals.decision[1].question.type).toBe('span')
    expect(payload.global.services.opaque_secret).toBe('test-updated-secret')
    expect(payload.providers.models[0].api_key).toBe('test-provider-secret')
    expect(payload.routing.replay).toEqual({ opaque_future_flag: true })
    await dialog.getByRole('button', { name: 'Close deployment dialog' }).click()
    const configured = page.getByRole('region', { name: 'Configured runtimes' })
    await expect(configured).toContainText('Not reported')
    fixture.activateRuntime()
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect(configured).toContainText('Ready')
  })

  test('saves an unbound Decision 1.0 honestly and keeps a question-creation path open', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, { noConsumers: true })
    await page.goto('/decision-model')
    await page
      .getByRole('article', { name: 'Decision-1.0-Lex-0.6B', exact: true })
      .getByRole('button', { name: 'Configure' })
      .click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toContainText('No compatible consumers yet')
    await expect(dialog.getByRole('link', { name: /Open Signals/ })).toHaveAttribute(
      'target',
      '_blank',
    )
    await dialog.getByRole('button', { name: 'Save configuration', exact: true }).click()
    await expect(dialog.getByRole('status')).toContainText('Saved without activation')
    await expect(dialog.getByRole('status')).not.toContainText('Ready')
    expect(fixture.runtimeRequests[0].routing.signals.decision).toEqual([])
    await dialog.getByRole('button', { name: 'Close deployment dialog' }).click()
    await expect(page.getByRole('region', { name: 'Configured runtimes' })).toContainText(
      'Saved · no decision binding',
    )
  })

  test('binds a decision selector and exposes deferred activation without claiming readiness', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, { applyStatus: 'restart_required' })
    await page.goto('/decision-model')
    await page
      .getByRole('article', { name: 'Decision-1.0-Kai-0.6B', exact: true })
      .getByRole('button', { name: 'Configure' })
      .click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('combobox', { name: 'Deployment consumer' }).click()
    await dialog.getByRole('option', { name: /research \/ pick/ }).click()
    await dialog.getByRole('button', { name: 'Deploy and bind model' }).click()
    await expect(dialog.getByRole('status')).toContainText('Restart required')
    expect(fixture.runtimeRequests[0].recipes[0].routing.decisions[0].algorithm.decision).toEqual({
      instructions: 'Which model?',
      candidates: { answer: 'General model' },
      deployment: 'decision-1-0-kai-0-6b',
    })
  })

  test('revalidates consumer compatibility against a fresh snapshot before any write', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    await page.goto('/decision-model')
    await page
      .getByRole('article', { name: 'Decision-2.0-Kai-0.6B', exact: true })
      .getByRole('button', { name: 'Configure' })
      .click()
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('combobox', { name: 'Deployment consumer' }).click()
    await dialog.getByRole('option', { name: /Default routing \/ task/ }).click()
    fixture.config.routing.signals.decision[0].question.type = 'span'
    await dialog.getByRole('button', { name: 'Deploy and bind model' }).click()
    await expect(dialog.getByRole('alert')).toContainText('no longer compatible')
    expect(fixture.runtimeRequests).toHaveLength(0)
  })

  test('allows readonly model inspection but no configuration mutation', async ({ page }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, { readonly: true })
    await page.goto('/decision-model')
    await page
      .getByRole('article', { name: 'Decision-2.0-Kai-0.6B', exact: true })
      .getByRole('button', { name: 'View model' })
      .click()
    const dialog = page.getByRole('dialog')
    await expect(dialog.getByRole('textbox', { name: 'Deployment name' })).toBeDisabled()
    await expect(dialog.getByRole('combobox', { name: 'Deployment consumer' })).toBeDisabled()
    await expect(
      dialog.getByRole('button', { name: 'Save configuration', exact: true }),
    ).toBeDisabled()
    expect(fixture.runtimeRequests).toHaveLength(0)
  })

  test('keeps the catalog and deployment dialog usable at mobile width', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockDecisionRuntimeCatalog(page)
    await page.goto('/decision-model')
    await page.getByRole('button', { name: /^Decision 2.0/ }).click()
    await page
      .getByRole('article', { name: 'Decision-2.0-Kai-0.6B', exact: true })
      .getByRole('button', { name: 'Configure' })
      .click()
    const dialog = page.getByRole('dialog')
    await expect(dialog).toBeVisible()
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    const consumer = dialog.getByRole('combobox', { name: 'Deployment consumer' })
    await consumer.click()
    await page.keyboard.press('Escape')
    await expect(dialog).toBeVisible()
    await expect(consumer).toBeFocused()
    await expect(consumer).toHaveAttribute('aria-expanded', 'false')
    await consumer.click()
    await dialog.getByRole('option', { name: /Default routing \/ task/ }).click()
    await expect(dialog.getByRole('button', { name: 'Deploy and bind model' })).toBeInViewport()
    await page.screenshot({
      path: '../../.agent-harness/decision-model-dashboard/decision-catalog-mobile.png',
      fullPage: true,
    })
    await page.keyboard.press('Escape')
    await expect(dialog).toHaveCount(0)
  })
})
