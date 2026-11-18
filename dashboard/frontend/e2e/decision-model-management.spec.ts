import { readFileSync } from 'node:fs'
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
      system: (options.defaultModel ? {} : { decision_model: { deployment: 'primary' } }) as {
        decision_model?: { deployment: string }
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
  let observed = options.defaultModel ? 'vela3-fast' : 'Vela-2.0-4B'
  let observedDeployment = 'primary'
  const config = {
    version: 'v0.5',
    global,
    providers: { models: [] },
    recipes: [
      {
        name: 'default',
        signals: options.defaultModel
          ? {}
          : { systemone: [{ name: 'task', question: { type: 'noul' } }] },
        decisions: [],
      },
    ],
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
                recipe: 'default',
                deployment: 'pii-specialist',
                model: 'models/custom-pii',
                source: 'global',
                ready: true,
                editable: true,
                path: ['recipes', '0', 'model_bindings', 'pii'],
                binding: 'pii-specialist',
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
        'vela3-fast'
      pending = false
    },
  }
}

const decisionCatalog = JSON.parse(
  readFileSync(
    new URL('../src/pages/decisionRuntimeCatalog.generated.json', import.meta.url),
    'utf8',
  ),
) as {
  models: Array<{
    id: string
    name: string
    revision: string
    questionTypes: string[]
    pending: string
  }>
}
const revisionOf = (id: string) => decisionCatalog.models.find((model) => model.id === id)!.revision
const defaultCard = (page: Page) => page.getByRole('region', { name: 'Default decision model' })

// The first visit compiles the page's modules in the dev server; wait for the tab, not a fixed delay.
async function gotoDecisionModels(page: Page, path = '/models/decision') {
  await page.goto(path)
  const ready = path.includes('tab=monitoring')
    ? page.getByRole('heading', { name: 'Model runtime deployments', exact: true })
    : page.getByRole('heading', { name: 'Catalog', exact: true })
  await expect(ready).toBeVisible({ timeout: 30_000 })
}

async function openDeployDialog(page: Page, label: string) {
  await page.getByRole('button', { name: `Deploy ${label}`, exact: true }).click()
  return page.getByRole('dialog')
}

async function deployAsDefault(page: Page, label: string) {
  const dialog = await openDeployDialog(page, label)
  await dialog.getByRole('radio', { name: /Default decision model/ }).click()
  await dialog.getByRole('button', { name: 'Deploy and activate', exact: true }).click()
  return dialog
}

async function deployAsDefaultAndClose(page: Page, label: string) {
  const dialog = await deployAsDefault(page, label)
  await expect(dialog).toBeHidden({ timeout: 15_000 })
}

test.describe('System One model management and monitoring', () => {
  test('shows measured runtime statistics and clears them when observation fails', async ({
    page,
  }) => {
    const fixture = await mockDecisionModelManager(page)
    await gotoDecisionModels(page, '/models/decision?tab=monitoring')
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
    await expect(page.getByRole('tab', { name: 'Deploy' })).toBeVisible()
  })

  test('keeps the default vela3-fast model and shows absent samples as unknown without empty configuration clutter', async ({
    page,
  }) => {
    await mockDecisionModelManager(page, { defaultModel: true, metrics: 'empty' })
    await gotoDecisionModels(page, '/models/decision')
    await expect(
      defaultCard(page).getByRole('heading', { name: 'vela3-fast', exact: true }),
    ).toBeVisible()
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'No active task bindings in this configuration.',
    )
    await expect(page.getByRole('heading', { name: 'Custom model assignments' })).toHaveCount(0)
    await expect(page.getByRole('heading', { name: 'Custom questions' })).toHaveCount(0)
    await expect(page.getByText(/No custom questions configured/)).toHaveCount(0)
    await page.getByRole('tab', { name: 'Monitoring' }).click()
    const stats = page
      .getByRole('article', { name: 'primary', exact: true })
      .locator('dl[aria-label="Model statistics"]')
    await expect(stats.getByText('Not reported', { exact: true })).toHaveCount(6)
  })

  test('changes real monitoring windows, handles partial failure, and recovers from missing observations', async ({
    page,
  }) => {
    const fixture = await mockDecisionModelManager(page, { lowTraffic: true })
    await gotoDecisionModels(page, '/models/decision?tab=monitoring')
    const stats = page.locator('dl[aria-label="Model statistics"]')
    await expect(stats).toContainText('0.02')
    for (const region of ['Traffic & reliability', 'Latency breakdown', 'Result cache efficiency'])
      await expect(
        page.getByRole('region', { name: region }).locator('svg.recharts-surface'),
      ).toHaveCount(1)
    const rateTicks = page
      .getByRole('region', { name: 'Traffic & reliability' })
      .locator('.recharts-yAxis')
      .first()
      .locator('.recharts-cartesian-axis-tick-value')
    await expect
      .poll(async () => new Set(await rateTicks.allTextContents()).size)
      .toBeGreaterThan(1)
    const windows = page.getByRole('radiogroup', { name: 'Monitoring time range' })
    await expect(windows.getByRole('radio', { name: '1h' })).toHaveAttribute('aria-checked', 'true')
    const checkRange = (seconds: number) => {
      const params = fixture.metricsRanges.at(-1)!
      expect(Number(params.get('end')) - Number(params.get('start'))).toBe(seconds)
      expect(seconds / Number(params.get('step')) + 1).toBeLessThanOrEqual(181)
    }
    checkRange(3600)
    fixture.setMetrics('partial')
    await windows.getByRole('radio', { name: '6h' }).click()
    await expect(page.getByText(/Unavailable: Unsuccessful calls/)).toBeVisible()
    await expect(stats).toContainText('0.02')
    await expect(stats.getByText('Not reported', { exact: true })).toHaveCount(1)
    checkRange(21600)
    fixture.setMetrics('empty')
    await windows.getByRole('radio', { name: '15m' }).click()
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
    await expect(page.getByRole('tab', { name: 'Playground' })).toBeVisible()
  })

  test('keeps charts and model controls usable on a narrow viewport', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await mockDecisionModelManager(page)
    await gotoDecisionModels(page, '/models/decision?tab=monitoring')
    await expect(page.locator('dl[aria-label="Model statistics"]')).toContainText('2.50')
    await expect(
      page
        .getByRole('radiogroup', { name: 'Monitoring time range' })
        .getByRole('radio', { name: '6h' }),
    ).toBeVisible()
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await page.getByRole('region', { name: 'Latency breakdown' }).scrollIntoViewIfNeeded()
    await expect(page.getByRole('region', { name: 'Latency breakdown' })).toBeInViewport()
    await page.getByRole('tab', { name: 'Deploy' }).click()
    await expect(
      page.getByRole('button', { name: 'Deploy Vela 2.0 9B', exact: true }),
    ).toBeEnabled()
  })

  test('isolates stalled monitoring requests from model management', async ({ page }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    let queries = 0
    await page.route('**/embedded/prometheus/api/v1/query_range?*', () => {
      queries += 1
    })
    await gotoDecisionModels(page, '/models/decision')
    await expect(
      page.getByRole('button', { name: 'Deploy Vela 2.0 9B', exact: true }),
    ).toBeEnabled()
    expect(queries).toBe(0)
    await page.getByRole('tab', { name: 'Monitoring' }).click()
    await expect.poll(() => queries).toBeGreaterThanOrEqual(6)
    await page.clock.fastForward(8_100)
    await expect(page.getByText(/Some model statistics are unavailable/)).toBeVisible()
    await page.getByRole('tab', { name: 'Deploy' }).click()
    await deployAsDefaultAndClose(page, 'Vela 2.0 9B')
    await expect.poll(() => fixture.requests.length).toBe(1)
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
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
    await gotoDecisionModels(page, '/models/decision')
    await expect(page.getByRole('heading', { name: 'Decision models', exact: true })).toBeVisible()
    await expect(
      page.getByRole('button', { name: 'Deploy Vela 2.0 9B', exact: true }),
    ).toBeVisible()
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
    await gotoDecisionModels(page, '/models/decision')
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
    await gotoDecisionModels(page, '/models/decision?tab=monitoring')
    await expect(page.getByRole('heading', { name: 'Decision models', exact: true })).toBeVisible()
    await expect(page.locator('dl[aria-label="Model statistics"]')).toContainText('2.50')
    expect(configRequests).toEqual([])
  })

  test('refreshes externally changed saved configuration while preserving an open deploy draft', async ({
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
    await gotoDecisionModels(page, '/models/decision')
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    expect([globalReads, configReads]).toEqual([1, 1])
    fixture.changeExternally('Vela-2.0-9B', 'external-change-1')
    await page.clock.fastForward(10_100)
    await expect(
      defaultCard(page).getByRole('heading', { name: 'Vela-2.0-9B', exact: true }),
    ).toBeVisible()
    expect([globalReads, configReads]).toEqual([2, 2])

    const dialog = await openDeployDialog(page, 'Vela 2.0 0.3B')
    await dialog.getByRole('radio', { name: /Default decision model/ }).click()
    fixture.changeExternally('Vela-2.0-4B', 'external-change-2')
    await page.clock.fastForward(10_100)
    await expect(dialog.getByRole('radio', { name: /Default decision model/ })).toBeChecked()
    await expect(dialog.getByLabel('Deployment', { exact: true })).toHaveValue('vela-2-0-0-3b')
    expect([globalReads, configReads]).toEqual([3, 3])
    await page.clock.fastForward(30_100)
    expect([globalReads, configReads]).toEqual([3, 3])
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
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
    await gotoDecisionModels(page, '/models/decision')
    await expect(
      page.getByRole('button', { name: 'Deploy Vela 2.0 9B', exact: true }),
    ).toBeEnabled()
    stallInventory = true
    await page.clock.fastForward(10_100)
    await expect.poll(() => stalledReads).toBe(1)

    await deployAsDefaultAndClose(page, 'Vela 2.0 9B')
    await expect.poll(() => fixture.requests.length).toBe(1)
    await page.clock.fastForward(10_100)
    await expect.poll(() => stalledReads).toBe(2)
    await page.clock.fastForward(10_100)
    await expect(
      page.getByRole('button', { name: 'Deploy Vela 2.0 0.3B', exact: true }),
    ).toBeEnabled()
    await expect(
      page.getByRole('status').filter({ hasText: /^Runtime last observed/ }),
    ).toContainText('The request timed out. Please retry.')
    await expect(
      defaultCard(page).getByText('Restart required', { exact: true }).first(),
    ).toBeVisible()
  })

  test('keeps management and runtime details on separate tabs', async ({ page }) => {
    await mockDecisionModelManager(page)
    await gotoDecisionModels(page, '/models/decision')
    await expect(page.getByRole('heading', { name: 'Decision models', exact: true })).toBeVisible()
    const card = defaultCard(page)
    await expect(card.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
    await expect(card.getByText('Ready', { exact: true })).toBeVisible()
    await expect(card).toContainText('Router mode')
    const bindings = page.getByRole('region', { name: 'Task bindings' })
    await expect(bindings).toContainText('PII extraction')
    await expect(bindings).toContainText('models/custom-pii')
    await expect(bindings).toContainText('Global binding')
    await page.getByRole('tab', { name: 'Monitoring' }).click()
    await expect(page).toHaveURL(/\/models\/decision\?tab=monitoring$/)
    const deployments = page.getByRole('region', { name: 'Model runtime deployments' })
    await deployments.getByText('Deployment details', { exact: true }).click()
    for (const value of ['revision-123', 'rocm:0', 'torch', 'exact', 'model-runtime-1', 'safety']) {
      await expect(deployments).toContainText(value)
    }
  })

  test('keeps saved and observed models separate until a restart is observed', async ({ page }) => {
    await page.clock.install()
    const fixture = await mockDecisionModelManager(page)
    await gotoDecisionModels(page, '/models/decision')
    await deployAsDefaultAndClose(page, 'Vela 2.0 9B')
    await expect(page.getByRole('status').filter({ hasText: 'Run vllm-sr serve' })).toBeVisible()
    expect(fixture.requests).toEqual([
      {
        ...fixture.config,
        global: {
          model_catalog: {
            system: { decision_model: { deployment: 'vela-2-0-9b' } },
            deployments: {
              primary: {
                provider: 'model_runtime',
                artifact: 'vllm-sr/Vela-2.0-4B',
                device: 'rocm:0',
              },
              'vela-2-0-9b': {
                provider: 'model_runtime',
                artifact: 'vllm-sr/Vela-2.0-9B',
                revision: revisionOf('vllm-sr/Vela-2.0-9B'),
                device: 'auto',
              },
            },
          },
        },
      },
    ])
    const card = defaultCard(page)
    await expect(card.getByRole('heading', { name: 'Vela-2.0-9B', exact: true })).toBeVisible()
    await expect(card).toContainText('Still serving Vela-2.0-4B on primary')
    await expect(card.getByText('Restart required', { exact: true }).first()).toBeVisible()
    await card.getByText('Deployment diagnostics', { exact: true }).click()
    await expect(card).toContainText('Saved model: Vela 2.0 9B')
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'models/custom-pii',
    )
    fixture.activate()
    await page.clock.fastForward(10_100)
    await expect(card.getByText('Ready', { exact: true })).toBeVisible()
    await expect(card).not.toContainText('Still serving')
    await expect(card).toContainText('Router configuration active')
  })

  test('shows a ConfigMap save as requiring rollout', async ({ page }) => {
    await mockDecisionModelManager(page, { applyStatus: 'persisted' })
    await gotoDecisionModels(page, '/models/decision')
    await deployAsDefaultAndClose(page, 'Vela 2.0 9B')
    await expect(page.getByRole('status').filter({ hasText: 'Saved to ConfigMap' })).toContainText(
      'roll out Router and Envoy',
    )
    await expect(defaultCard(page)).toContainText('Still serving Vela-2.0-4B')
  })

  test('keeps the deploy draft and reports the server error after apply fails', async ({
    page,
  }) => {
    await mockDecisionModelManager(page, { applyStatus: 'failed' })
    await gotoDecisionModels(page, '/models/decision')
    const dialog = await deployAsDefault(page, 'Vela 2.0 9B')
    await expect(dialog.getByRole('alert')).toContainText('GPU memory unavailable')
    await expect(dialog.getByRole('radio', { name: /Default decision model/ })).toBeChecked()
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    await defaultCard(page).getByText('Deployment diagnostics', { exact: true }).click()
    await expect(page.getByText('Saved model: Vela 2.0 9B', { exact: true })).toBeVisible()
  })

  test('keeps management observable but disables writes for config readers', async ({ page }) => {
    const fixture = await mockDecisionModelManager(page, { readonly: true })
    await gotoDecisionModels(page, '/models/decision')
    await expect(page.getByRole('heading', { name: 'Decision models', exact: true })).toBeVisible()
    await expect(
      page.getByRole('button', { name: 'Deploy Vela 2.0 9B', exact: true }),
    ).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
    await page.getByRole('tab', { name: 'Monitoring' }).click()
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
    await gotoDecisionModels(page, '/models/decision')
    await expect(page.getByText('Engine mode', { exact: true })).toBeVisible()
    await deployAsDefaultAndClose(page, 'Vela 2.0 9B')
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
    version: 'v0.5',
    global: {
      model_catalog: {
        system: { decision_model: { deployment: 'primary' } },
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
    providers: {
      models: [
        {
          name: 'answer',
          backend_refs: [
            { name: 'primary', endpoint: 'answer.example.com', api_key: 'test-provider-secret' },
          ],
        },
      ],
    },
    recipes: [
      {
        name: 'default',
        signals: {
          systemone: options.noConsumers
            ? []
            : [{ name: 'task', question: { type: 'noul', instructions: 'Is this coding?' } }],
        },
        decisions: [],
      },
      {
        name: 'research',
        model_bindings: { complexity: 'existing' } as Record<string, string>,
        signals: {
          complexity: options.noConsumers ? [] : [{ name: 'difficulty' }],
        },
        decisions: options.noConsumers
          ? []
          : [
              {
                name: 'pick',
                algorithm: {
                  type: 'systemone',
                  systemone: {
                    instructions: 'Which model?',
                    candidates: { answer: 'General model' },
                  },
                },
                modelRefs: [{ model: 'answer' }],
              },
            ],
      },
    ],
    entrypoints: [
      { model_names: ['vllm-sr/auto'], recipe: 'default', default: true },
      { model_names: ['research'], recipe: 'research' },
    ],
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
    // The Router lists a recipe's complexity rules once, under their signal
    // type, and a selector with no path, since its decision writes it.
    const index = config.recipes.findIndex((recipe) => recipe.name === 'research')
    const bound = (config.recipes[index] as { model_bindings?: Record<string, string> } | undefined)
      ?.model_bindings?.complexity
    const deployment = bound ?? 'primary'
    const bindings =
      options.noConsumers || index < 0
        ? []
        : [
            {
              task_id: 'complexity',
              consumer: 'complexity',
              recipe: 'research',
              deployment,
              model: config.global.model_catalog.deployments[deployment]?.artifact,
              source: bound ? 'recipe' : 'default',
              ready: deployment === 'primary' || ready,
              editable: true,
              path: ['recipes', String(index), 'model_bindings', 'complexity'],
              binding: deployment,
            },
            {
              task_id: 'selector',
              consumer: 'selector.pick',
              recipe: 'research',
              deployment: 'primary',
              model: config.global.model_catalog.deployments.primary.artifact,
              source: 'default',
              ready: true,
              editable: false,
              path: null,
              binding: 'primary',
            },
          ]
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
  test('lists every family from the registry projection, filters capabilities and serves the Vela 3.1 releases, without loading Model Hub', async ({
    page,
  }, testInfo) => {
    const fixture = await mockDecisionRuntimeCatalog(page)
    const catalogRequests: string[] = []
    page.on('request', (request) => {
      if (request.url().includes('/api/models/catalog')) catalogRequests.push(request.url())
    })
    await gotoDecisionModels(page, '/models/decision')
    const catalog = page.locator('#decision-catalog')
    await expect(catalog).toContainText(
      `${decisionCatalog.models.length} of ${decisionCatalog.models.length} models`,
    )
    for (const name of [
      'vela3-fast',
      'vela3-lite',
      'vela3-mini',
      'vela3-flash',
      'vela3-turbo',
      'vela3-fast-vision',
      'd3-ultra',
      'd3-turbo',
      'Vela-2.0-0.3B',
      'Decision-2.0-Kai-0.6B',
    ])
      await expect(catalog.getByText(name, { exact: true })).toBeVisible()
    const vela31 = catalog.getByRole('region', { name: 'Vela 3.1 models' })
    await expect(vela31.getByRole('article')).toHaveCount(3)
    for (const label of ['Vela 3.1 fast', 'Vela 3.1 pico 150M', 'Vela 3.1 pico 380M'])
      await expect(
        vela31.getByRole('button', { name: `Deploy ${label}`, exact: true }),
      ).toBeVisible()
    await expect(vela31.getByText('Not servable yet')).toHaveCount(0)
    await catalog.getByRole('radio', { name: 'Decision 1.0', exact: true }).click()
    await expect(catalog.getByRole('article')).toHaveCount(7)
    await page.getByPlaceholder('Search decision models…').fill('Lex')
    await expect(catalog.getByRole('article')).toHaveCount(1)
    await expect(catalog.getByRole('article', { name: 'Decision 1.0 Lex 0.6B' })).toBeVisible()
    await page.getByPlaceholder('Search decision models…').clear()
    await catalog.getByRole('radio', { name: 'All', exact: true }).click()
    await catalog.getByRole('button', { name: 'Answers' }).click()
    await page.getByRole('menuitemcheckbox', { name: /^Span/ }).click()
    await page.keyboard.press('Escape')
    const spanModels = decisionCatalog.models.filter((model) =>
      model.questionTypes.includes('span'),
    )
    await expect(catalog.getByRole('article')).toHaveCount(spanModels.length)
    await expect(catalog.getByText('Recommended for CPU')).toHaveCount(0)
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
    await gotoDecisionModels(page, '/models/decision')
    const row = page.getByRole('row').filter({ hasText: 'Task difficulty' })
    await expect(row).toContainText('team/custom-decision-model')
    await expect(row).toContainText('research · complexity')
    await row.click()
    await page.getByRole('combobox', { name: 'Model for Task difficulty' }).click()
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
    await gotoDecisionModels(page, '/models/decision')
    await page.getByPlaceholder('Search decision models…').fill('Decision 2.0 Kai')
    const dialog = await openDeployDialog(page, 'Decision 2.0 Kai 0.6B')
    await expect(dialog.getByLabel('Deployment', { exact: true })).toHaveValue(
      'decision-2-0-kai-0-6b',
    )
    await dialog.getByRole('radio', { name: /Default decision model/ }).click()
    fixture.config.global.services.opaque_secret = 'test-updated-secret'
    const previous = structuredClone(fixture.config)
    await dialog.getByRole('button', { name: 'Deploy and activate', exact: true }).click()
    await expect(
      page.getByRole('status').filter({ hasText: 'Configuration applied.' }),
    ).toBeVisible()
    expect(fixture.runtimeRequests).toHaveLength(1)
    expect(fixture.runtimeRequests[0]).toEqual({
      ...previous,
      global: {
        ...previous.global,
        model_catalog: {
          ...previous.global.model_catalog,
          system: { decision_model: { deployment: 'decision-2-0-kai-0-6b' } },
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
    await expect(
      defaultCard(page).getByRole('heading', { name: 'Decision-2.0-Kai-0.6B', exact: true }),
    ).toBeVisible()
    await page.goto('/models/decision?deployment=decision-2-0-kai-0-6b')
    const replicas = page.getByRole('region', { name: 'Replicas' })
    await expect(replicas).toContainText('Not observed', { timeout: 30_000 })
    fixture.activateRuntime()
    await page.goto('/models/decision?deployment=decision-2-0-kai-0-6b')
    await expect(page.getByRole('region', { name: 'Replicas' })).toContainText('ready', {
      timeout: 30_000,
    })
  })

  test('makes an existing Decision 1.0 deployment the default without claiming a new runtime is ready', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, {
      noConsumers: true,
      applyStatus: 'restart_required',
    })
    await gotoDecisionModels(page, '/models/decision')
    await page.getByPlaceholder('Search decision models…').fill('Decision 1.0 Lex')
    await page.getByRole('button', { name: 'Open deployment lex' }).click()
    const sheet = page.getByRole('dialog')
    await sheet.getByRole('button', { name: 'Make default' }).click()
    const dialog = page.getByRole('dialog', { name: 'Bind consumers to lex' })
    await expect(dialog.getByRole('radio', { name: /Default decision model/ })).toBeChecked()
    await dialog.getByRole('button', { name: 'Save and activate', exact: true }).click()
    await expect(
      page.getByRole('status').filter({ hasText: 'Configuration saved. Restart' }),
    ).toBeVisible()
    expect(fixture.runtimeRequests[0].recipes[0].signals.systemone).toEqual([])
    expect(fixture.runtimeRequests[0].global.model_catalog.system).toEqual({
      decision_model: { deployment: 'lex' },
    })
    expect(fixture.runtimeRequests[0].global.model_catalog.deployments).toEqual(
      fixture.config.global.model_catalog.deployments,
    )
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'No active task bindings',
    )
    await expect(defaultCard(page)).toContainText('Still serving Vela-2.0-4B')
  })

  test('binds the complexity task in its recipe while preserving the default and exposes deferred activation honestly', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, { applyStatus: 'restart_required' })
    await gotoDecisionModels(page, '/models/decision')
    const row = page.getByRole('row').filter({ hasText: 'Task difficulty' })
    await row.click()
    await page.getByRole('combobox', { name: 'Model for Task difficulty' }).click()
    await page.getByRole('option', { name: /Decision-1.0-Lex-0.6B/ }).click()
    const previous = structuredClone(fixture.config)
    await page.getByRole('dialog').getByRole('button', { name: 'Apply', exact: true }).click()
    await expect(
      page.getByRole('status').filter({ hasText: 'Configuration saved. Restart' }),
    ).toBeVisible()
    expect(fixture.runtimeRequests).toHaveLength(1)
    Object.assign(previous.recipes[1], { model_bindings: { complexity: 'lex' } })
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
    await gotoDecisionModels(page, '/models/decision')
    await page.getByRole('row').filter({ hasText: 'Task difficulty' }).click()
    await page.getByRole('combobox', { name: 'Model for Task difficulty' }).click()
    await page.getByRole('option', { name: /Vela-2.0-4B/ }).click()
    fixture.config.recipes = []
    const dialog = page.getByRole('dialog')
    await dialog.getByRole('button', { name: 'Apply', exact: true }).click()
    await expect(dialog.getByRole('alert')).toContainText('This recipe was removed')
    expect(fixture.runtimeRequests).toHaveLength(0)
  })

  test('allows readonly model and task inspection without configuration mutation', async ({
    page,
  }) => {
    const fixture = await mockDecisionRuntimeCatalog(page, { readonly: true })
    await gotoDecisionModels(page, '/models/decision')
    await page.getByPlaceholder('Search decision models…').fill('Decision 2.0 Kai')
    await expect(
      page.getByRole('button', { name: 'Deploy Decision 2.0 Kai 0.6B', exact: true }),
    ).toBeDisabled()
    const bindings = page.getByRole('region', { name: 'Task bindings' })
    await expect(bindings).toContainText('Task difficulty')
    await expect(bindings.getByRole('button', { name: 'Row actions' })).toHaveCount(0)
    await page.goto('/models/decision?deployment=primary')
    const replicas = page.getByRole('region', { name: 'Replicas' })
    await expect(replicas.getByLabel('Replica 1 device', { exact: true })).toBeDisabled({
      timeout: 30_000,
    })
    await expect(replicas.getByRole('button', { name: 'Add replica' })).toBeDisabled()
    expect(fixture.runtimeRequests).toHaveLength(0)
  })

  test('keeps catalog and task binding controls usable at mobile width with keyboard dismissal', async ({
    page,
  }, testInfo) => {
    await page.setViewportSize({ width: 390, height: 844 })
    const fixture = await mockDecisionRuntimeCatalog(page)
    await gotoDecisionModels(page, '/models/decision')
    await page
      .locator('#decision-catalog')
      .getByRole('radio', { name: 'Decision 2.0', exact: true })
      .click()
    await expect(
      page.getByRole('button', { name: 'Deploy Decision 2.0 Kai 0.6B', exact: true }),
    ).toBeVisible()
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await page.getByRole('row').filter({ hasText: 'Task difficulty' }).click()
    const consumer = page.getByRole('combobox', { name: 'Model for Task difficulty' })
    await consumer.click()
    // Escape closes the open list, not the dialog; it is only open once the list shows.
    await expect(page.getByRole('listbox')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('listbox')).toHaveCount(0)
    await expect(consumer).toBeFocused()
    await expect(consumer).toHaveAttribute('aria-expanded', 'false')
    await consumer.click()
    await page.getByRole('option', { name: /Decision-1.0-Lex-0.6B/ }).click()
    const apply = page.getByRole('dialog').getByRole('button', { name: 'Apply', exact: true })
    await expect(apply).toBeInViewport()
    await page.screenshot({
      path: testInfo.outputPath('decision-catalog-mobile.png'),
      fullPage: true,
    })
    await page.getByRole('dialog').getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(consumer).toHaveCount(0)
    expect(fixture.runtimeRequests).toEqual([])
  })
})
