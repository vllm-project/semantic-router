import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const artifact = 'vllm-sr/Vela-2.0-4B'
const config = {
  version: 'v0.3',
  global: {
    model_catalog: {
      system: { decision_model: { deployment: 'primary' } },
      deployments: {
        primary: {
          provider: 'model_runtime',
          artifact,
          profile: 'exact',
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
      consumer: 'category',
      recipe: '',
      deployment: 'primary',
      model: artifact,
      source: 'default',
      ready: true,
      editable: true,
      path: ['routing', 'model_bindings', 'domain'],
      binding: { deployment: 'primary', contract: 'decision.v1' },
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
    await page.goto('/decision-model')
    await expect(page.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
    const replicas = page.getByRole('region', { name: 'Model replicas' })
    await expect(replicas).toContainText('2 / 2')
    await expect(replicas).toContainText('Loading replica configuration')
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'Loading task bindings',
    )
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeDisabled()
    await page.getByLabel('Search models').fill('4B')
    releaseConfig()
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
    await expect(replicas.getByText('Configure replicas')).toBeVisible()
    releaseTasks()
    await expect(page.getByRole('region', { name: 'Task bindings' })).toContainText(
      'Domain classification',
    )
    expect(globalReads).toBe(1)
  } finally {
    releaseConfig()
    releaseTasks()
  }
})

test('keeps last observations on read failure and limits only the operations that depend on them', async ({
  page,
}) => {
  await mockObservations(page)
  await page.goto('/decision-model')
  const deploy = page.getByRole('button', { name: 'Deploy selection' })
  await expect(deploy).toBeEnabled()
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
  await expect(page.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
  const replicas = page.getByRole('region', { name: 'Model replicas' })
  await expect(replicas).toContainText('Replica health last observed')
  await expect(replicas).toContainText('2 / 2')
  await expect(page.getByText('Last observed: Router mode', { exact: true })).toBeVisible()
  await expect(deploy).toBeEnabled()
  await expect(
    page.getByRole('region', { name: 'Task bindings' }).getByRole('button', { name: 'Change' }),
  ).toBeVisible()
  await page.route('**/api/instance', (route) => route.fulfill(reply(instance)))
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect(page.getByText('Router mode', { exact: true })).toBeVisible()
})

test('does not disable edits during an ordinary refresh, but marks failed configuration and tasks stale', async ({
  page,
}) => {
  await mockObservations(page)
  await page.goto('/decision-model')
  const deploy = page.getByRole('button', { name: 'Deploy selection' })
  await expect(deploy).toBeEnabled()
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
    await expect(deploy).toBeEnabled()
    release()
    await expect(deploy).toBeDisabled()
    const catalog = page.getByRole('region', { name: 'Choose a decision model' })
    await expect(catalog).toContainText('Model configuration last observed')
    const bindings = page.getByRole('region', { name: 'Task bindings' })
    await expect(bindings).toContainText('Domain classification')
    await expect(bindings).toContainText('Task bindings last observed')
    await expect(bindings.getByRole('button', { name: 'Change' })).toHaveCount(0)
    const replicas = page.getByRole('region', { name: 'Model replicas' })
    await replicas.getByText('Configure replicas').click()
    await expect(replicas.getByRole('button', { name: 'Add replica' })).toBeDisabled()
  } finally {
    release()
  }
})

test('keeps observation cards readable on desktop and mobile', async ({ page }, testInfo) => {
  await mockObservations(page)
  await page.goto('/decision-model')
  await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 })
    await expect(page.getByRole('heading', { name: 'Vela-2.0-4B', exact: true })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Model replicas' })).toContainText('2 / 2')
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
  await page.goto('/decision-model')
  await expect(page.getByText('Engine mode', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Deploy selection' })).toBeEnabled()
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
  await page.goto('/decision-model')
  await expect(page.getByRole('button', { name: 'Deploying…', exact: true })).toBeDisabled()
  await expect(page.getByText('Deployment in progress.', { exact: false })).toBeVisible()
  const replicas = page.getByRole('region', { name: 'Model replicas' })
  await replicas.getByText('Configure replicas').click()
  await expect(replicas.getByRole('button', { name: 'Add replica' })).toBeDisabled()
  const bindings = page.getByRole('region', { name: 'Task bindings' })
  await expect(bindings.getByRole('button', { name: 'Change' })).toHaveCount(0)
  await page.route('**/api/instance', (route) =>
    route.fulfill({ status: 503, ...reply({ error: 'Controller disconnected' }) }),
  )
  await page.clock.fastForward(3100)
  await expect(page.getByRole('button', { name: 'Deploy selection', exact: true })).toBeEnabled()
  await expect(replicas.getByRole('button', { name: 'Add replica' })).toBeEnabled()
  await expect(bindings.getByRole('button', { name: 'Change' })).toBeVisible()
  await expect(page.getByText('Last observed: preparing', { exact: true })).toBeVisible()
  await expect(page.getByText('Deployment in progress.', { exact: false })).toHaveCount(0)
})
