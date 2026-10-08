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
        generated_runtime_hash: pending ? 'new-config' : 'active-config',
        active_runtime_hash: 'active-config',
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

test.describe('Decision model management', () => {
  test('shows measured runtime statistics and clears them when observation fails', async ({
    page,
  }) => {
    const fixture = await mockDecisionModelManager(page)
    await page.goto('/decision-model')
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
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
  })

  test('keeps the default 0.3B model and shows absent samples as unknown without empty configuration clutter', async ({
    page,
  }) => {
    await mockDecisionModelManager(page, { defaultModel: true, metrics: 'empty' })
    await page.goto('/decision-model')
    await expect(page.getByRole('radio', { name: /Vela 2.0 0.3B/ })).toBeChecked()
    const stats = page
      .getByRole('article', { name: '@Vela-2.0-0.3B/auto', exact: true })
      .locator('dl[aria-label="Model statistics"]')
    await expect(stats.getByText('Not reported', { exact: true })).toHaveCount(6)
    await page.getByText('Advanced bindings', { exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Custom model assignments' })).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Custom questions' })).toHaveCount(0)
    await expect(page.getByText('Default routing', { exact: true })).toHaveCount(0)
    await expect(page.getByText(/No custom questions configured/)).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Explicit signal overrides' })).toHaveCount(0)
  })

  test('changes real monitoring windows, handles partial failure, and recovers from missing observations', async ({
    page,
  }) => {
    const fixture = await mockDecisionModelManager(page, { lowTraffic: true })
    await page.goto('/decision-model')
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
      page.getByRole('link', { name: 'Test decision model', exact: true }),
    ).toHaveAttribute('href', '/decision-model/playground')
  })

  test('keeps charts and model controls usable on a narrow viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockDecisionModelManager(page)
    await page.goto('/decision-model')
    await expect(page.locator('dl[aria-label="Model statistics"]')).toContainText('2.50')
    await expect(page.getByRole('button', { name: '6h', exact: true })).toBeVisible()
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await page.getByRole('region', { name: 'Latency breakdown' }).scrollIntoViewIfNeeded()
    await expect(page.getByRole('region', { name: 'Latency breakdown' })).toBeInViewport()
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
  })

  test('bounds stalled metrics independently of configuration deployment', async ({ page }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    let queries = 0
    await page.route('**/embedded/prometheus/api/v1/query_range?*', () => {
      queries += 1
    })
    await page.goto('/decision-model')
    await expect.poll(() => queries).toBeGreaterThanOrEqual(6)
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await page.getByRole('radio', { name: /Vela 2.0 9B/ }).check()
    await page.getByRole('button', { name: 'Deploy selected model' }).click()
    await expect.poll(() => fixture.requests.length).toBe(1)
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeEnabled()
    await page.clock.fastForward(8_100)
    await expect(page.getByText(/Some model statistics are unavailable/)).toBeVisible()
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

  test('opens from Routing Models and shows model, hardware, runtime and binding details', async ({
    page,
  }) => {
    await mockDecisionModelManager(page)
    await page.goto('/status')
    await page.getByRole('button', { name: 'Build', exact: true }).click()
    const menu = page.getByRole('navigation', { name: 'Build' })
    await menu.getByRole('tab', { name: /Routing/ }).click()
    await menu.getByRole('link', { name: 'Decision Model', exact: true }).click()
    await expect(page).toHaveURL('/decision-model')
    await expect(page.getByRole('heading', { name: 'Decision Model', exact: true })).toBeVisible()
    await expect(page.getByRole('radio')).toHaveCount(5)
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    const status = page.getByRole('region', { name: 'Deployment status' })
    await expect(status).toContainText('Matching runtime ready')
    await expect(status).toContainText('Router')
    const deployments = page.getByRole('region', { name: 'Model runtime deployments' })
    await deployments.getByText('Deployment details', { exact: true }).click()
    for (const value of ['revision-123', 'rocm:0', 'torch', 'exact', 'model-runtime-1', 'safety']) {
      await expect(deployments).toContainText(value)
    }
    await expect(page.getByText('Custom model assignments', { exact: true })).not.toBeVisible()
    await page.getByText('Advanced bindings', { exact: true }).click()
    await expect(page.getByRole('heading', { name: 'Custom questions' })).toBeVisible()
    await expect(page.locator('details[open]').last()).toContainText('task')
    await expect(page.getByText('Default routing', { exact: true })).toHaveCount(0)
    await expect(page.getByText(/\(noul\)/)).toHaveCount(0)
    await expect(
      page.getByRole('link', { name: /Manage advanced model bindings/ }),
    ).toHaveAttribute('href', '/config/global-config#global-section-system_models')
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
    await expect(page.getByRole('region', { name: 'Model runtime deployments' })).toContainText(
      '@Vela-2.0-4B/auto',
    )
    await page.getByText('Advanced bindings', { exact: true }).click()
    await expect(page.getByText('models/custom-pii', { exact: true })).toBeVisible()
    fixture.activate()
    await page.clock.fastForward(10_100)
    await expect(status).toContainText('Matching runtime ready')
    await expect(status).toContainText('Active')
    await expect(page.getByRole('region', { name: 'Model runtime deployments' })).toContainText(
      '@Vela-2.0-9B/auto',
    )
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
    await expect(page.getByRole('heading', { name: 'Decision Model', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeDisabled()
    await expect(page.getByRole('radio').first()).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await expect(
      page.getByText(/Viewing model statistics requires observability read access/),
    ).toBeVisible()
    expect(fixture.metricsRequests).toEqual([])
  })

  test('does not apply router model settings to a standalone engine', async ({ page }) => {
    const fixture = await mockDecisionModelManager(page, { engine: true })
    await page.goto('/decision-model')
    await expect(page.getByRole('region', { name: 'Deployment status' })).toContainText('Engine')
    await expect(page.getByText(/This deployment is a standalone engine/)).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeDisabled()
    await expect(page.getByRole('radio').first()).toBeDisabled()
    expect(fixture.requests).toEqual([])
    expect(fixture.metricsRequests).toEqual([])
  })
})
