import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const artifact = 'vllm-sr/Vela-2.0-4B'
const config = {
  version: 'v0.5',
  global: {
    model_catalog: {
      system: { decision_model: { deployment: 'primary' } },
      deployments: {
        primary: {
          provider: 'model_runtime',
          artifact,
          replicas: [{ device: 'cpu' }, { device: 'cpu' }],
        },
      },
    },
  },
}
const instance = {
  ownership: 'managed',
  controller_available: true,
  can_switch: true,
  observed_mode: 'router',
  active_deployment: 'primary',
  model: artifact,
}
const inventory = {
  deployments: [
    {
      name: 'primary',
      repo: artifact,
      served_name: 'primary',
      managed: true,
      state: 'ready',
      ready: true,
      profile: 'exact',
      desired_replicas: 2,
      ready_replicas: 2,
    },
  ],
}
const tasks = {
  default_deployment: 'primary',
  tasks: [{ id: 'domain', title: 'Domain classification' }],
  deployments: [
    {
      deployment: 'primary',
      model: artifact,
      ready: true,
      native_question_types: ['choice'],
      tasks: [{ task_id: 'domain', supported: true }],
    },
  ],
  bindings: [
    {
      task_id: 'domain',
      consumer: 'domain',
      key: 'domain',
      recipe: 'default',
      deployment: 'primary',
      model: artifact,
      source: 'default',
      ready: true,
      editable: true,
      path: ['recipes', '0', 'model_bindings', 'domain'],
      binding: 'primary',
    },
  ],
}
const reply = (value: unknown) => ({ contentType: 'application/json', body: JSON.stringify(value) })

async function mockObservations(page: Page) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/all', (route) => route.fulfill(reply(config)))
  await page.route('**/api/router/config/global', (route) => route.fulfill(reply(config.global)))
  await page.route('**/api/instance', (route) => route.fulfill(reply(instance)))
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill(reply(inventory)),
  )
  await page.route('**/api/router/api/v1/config/hash', (route) =>
    route.fulfill(reply({ activation_status: 'active' })),
  )
  await page.route('**/api/decision-model/tasks', (route) => route.fulfill(reply(tasks)))
}

const defaultCard = (page: Page) => page.getByRole('region', { name: 'Default decision model' })
const deployButton = (page: Page) =>
  page.getByRole('button', { name: 'Deploy Vela 2.0 9B', exact: true })

async function gotoDecisionModels(page: Page) {
  await page.goto('/models/decision')
  await expect(page.getByRole('heading', { name: 'Catalog', exact: true })).toBeVisible({
    timeout: 30_000,
  })
}

test('shows live model and replica health while configuration and task bindings load independently', async ({
  page,
}) => {
  await mockObservations(page)
  let releaseConfig!: () => void
  const configPending = new Promise<void>((resolve) => {
    releaseConfig = resolve
  })
  let releaseTasks!: () => void
  const tasksPending = new Promise<void>((resolve) => {
    releaseTasks = resolve
  })
  let globalReads = 0
  await page.route('**/api/router/config/global', async (route) => {
    globalReads++
    await configPending
    await route.fulfill(reply(config.global))
  })
  await page.route('**/api/router/config/all', async (route) => {
    await configPending
    await route.fulfill(reply(config))
  })
  await page.route('**/api/decision-model/tasks', async (route) => {
    await tasksPending
    await route.fulfill(reply(tasks))
  })
  try {
    await gotoDecisionModels(page)
    const card = defaultCard(page)
    await expect(card.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
    await expect(card).toContainText('2/2')
    await expect(page.getByRole('region', { name: 'Task bindings' })).not.toContainText(
      'Domain classification',
    )
    await expect(deployButton(page)).toBeDisabled()
    releaseConfig()
    await expect(deployButton(page)).toBeEnabled()
    releaseTasks()
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'Domain classification',
    )
    await page.goto('/models/decision?deployment=primary')
    await expect(page.getByRole('region', { name: 'Replicas' })).toContainText(
      'Configure replicas',
      { timeout: 30_000 },
    )
    expect(globalReads).toBe(2)
  } finally {
    releaseConfig()
    releaseTasks()
  }
})

test('keeps last observations on read failure and limits only the operations that depend on them', async ({
  page,
}) => {
  await mockObservations(page)
  await gotoDecisionModels(page)
  await expect(deployButton(page)).toBeEnabled()
  await expect(page.getByText('Router mode', { exact: true })).toBeVisible()
  await expect(page.getByRole('group', { name: 'Instance mode', exact: true })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Apply mode', exact: true })).toHaveCount(0)
  await page.route('**/api/instance', (route) =>
    route.fulfill({ status: 503, ...reply({ error: 'Instance temporarily unreachable' }) }),
  )
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill({ status: 503, ...reply({ error: 'Runtime temporarily unreachable' }) }),
  )
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  const card = defaultCard(page)
  await expect(card.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
  await expect(card).toContainText('Runtime last observed')
  await expect(card).toContainText('2/2')
  await expect(page.getByText('Last observed: Router mode', { exact: true })).toBeVisible()
  await expect(deployButton(page)).toBeEnabled()
  await expect(
    page
      .getByRole('region', { name: 'Task bindings' })
      .getByRole('button', { name: 'Row actions' }),
  ).toBeVisible()
  await page.route('**/api/instance', (route) => route.fulfill(reply(instance)))
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect(page.getByText('Router mode', { exact: true })).toBeVisible()
})

test('does not disable edits during an ordinary refresh, but marks failed configuration and tasks stale', async ({
  page,
}) => {
  await mockObservations(page)
  await gotoDecisionModels(page)
  await expect(deployButton(page)).toBeEnabled()
  let release!: () => void
  const pending = new Promise<void>((resolve) => {
    release = resolve
  })
  let reads = 0
  await page.route('**/api/router/config/all', async (route) => {
    reads++
    await pending
    await route.fulfill({ status: 503, ...reply({ error: 'Configuration unavailable' }) })
  })
  await page.route('**/api/router/config/global', async (route) => {
    await pending
    await route.fulfill({ status: 503, ...reply({ error: 'Effective configuration unavailable' }) })
  })
  await page.route('**/api/decision-model/tasks', async (route) => {
    await pending
    await route.fulfill({ status: 503, ...reply({ error: 'Task discovery unavailable' }) })
  })
  try {
    await page.getByRole('button', { name: 'Refresh', exact: true }).click()
    await expect.poll(() => reads).toBe(1)
    await expect(deployButton(page)).toBeEnabled()
    release()
    await expect(deployButton(page)).toBeDisabled()
    await expect(defaultCard(page)).toContainText('Model configuration last observed')
    const bindings = page.getByRole('region', { name: 'Task bindings' })
    await expect(bindings).toContainText('Domain classification')
    await expect(bindings).toContainText('Task bindings last observed')
    await expect(bindings.getByRole('button', { name: 'Row actions' })).toHaveCount(0)
    await defaultCard(page).getByRole('button', { name: 'Default deployment actions' }).click()
    await page.getByRole('menuitem', { name: 'Deployment details' }).click()
    await expect(
      page.getByRole('region', { name: 'Replicas' }).getByRole('button', { name: 'Add replica' }),
    ).toBeDisabled()
  } finally {
    release()
  }
})

test('keeps observation cards readable on desktop and mobile', async ({ page }, testInfo) => {
  await mockObservations(page)
  await gotoDecisionModels(page)
  await expect(deployButton(page)).toBeEnabled()
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 })
    await expect(
      defaultCard(page).getByRole('heading', { name: 'Vela-2.0-4B', exact: true }),
    ).toBeVisible()
    await expect(defaultCard(page)).toContainText('2/2')
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(
      width,
    )
    await page.screenshot({
      path: testInfo.outputPath(`observations-${width}.png`),
      fullPage: true,
    })
  }
})

test('reports Engine mode without offering a Dashboard mode switch', async ({ page }) => {
  await mockObservations(page)
  await page.route('**/api/instance', (route) =>
    route.fulfill(reply({ ...instance, observed_mode: 'engine' })),
  )
  const modeWrites: string[] = []
  page.on('request', (request) => {
    if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/instance/deploy') {
      modeWrites.push(request.url())
    }
  })
  await gotoDecisionModels(page)
  await expect(page.getByText('Engine mode', { exact: true })).toBeVisible()
  await expect(deployButton(page)).toBeEnabled()
  await expect(page.getByRole('group', { name: 'Instance mode', exact: true })).toHaveCount(0)
  await expect(
    page.getByRole('button', { name: /Apply mode|Enable routing|Switch.*mode/i }),
  ).toHaveCount(0)
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect(page.getByText('Engine mode', { exact: true })).toBeVisible()
  expect(modeWrites).toEqual([])
})

test('treats a disconnected pending operation as history rather than blocking model management', async ({
  page,
}) => {
  await page.clock.install()
  await mockObservations(page)
  await page.route('**/api/instance', (route) =>
    route.fulfill(
      reply({
        ...instance,
        operation: {
          id: 'operation-1',
          phase: 'preparing',
          target_mode: 'router',
          started_at: new Date().toISOString(),
        },
      }),
    ),
  )
  await gotoDecisionModels(page)
  await expect(deployButton(page)).toBeDisabled()
  await expect(page.getByText('Deployment in progress', { exact: true })).toBeVisible()
  const bindings = page.getByRole('region', { name: 'Task bindings' })
  await expect(bindings.getByRole('button', { name: 'Row actions' })).toHaveCount(0)
  await page.route('**/api/instance', (route) =>
    route.fulfill({ status: 503, ...reply({ error: 'Controller disconnected' }) }),
  )
  await page.clock.fastForward(3100)
  await expect(deployButton(page)).toBeEnabled()
  await expect(bindings.getByRole('button', { name: 'Row actions' })).toBeVisible()
  await expect(page.getByText('Last observed: preparing', { exact: true })).toBeVisible()
  await expect(page.getByText('Deployment in progress', { exact: true })).toHaveCount(0)
  await defaultCard(page).getByRole('button', { name: 'Default deployment actions' }).click()
  await page.getByRole('menuitem', { name: 'Deployment details' }).click()
  await expect(
    page.getByRole('region', { name: 'Replicas' }).getByRole('button', { name: 'Add replica' }),
  ).toBeEnabled()
})
