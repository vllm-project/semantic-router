import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

async function mockDecisionModelManager(
  page: Page,
  options: {
    readonly?: boolean
    engine?: boolean
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
      system: { decision_model: 'Vela-2.0-4B', pii_classifier: 'models/custom-pii' },
    },
  }
  let observed = 'Vela-2.0-4B'
  let pending = false
  const requests: unknown[] = []
  const reply = (data: unknown) => ({
    status: 200,
    contentType: 'application/json',
    body: JSON.stringify(data),
  })
  await page.route('**/api/router/config/global', (route) => route.fulfill(reply(global)))
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill(
      reply({
        version: 'v0.3',
        global,
        providers: { models: [] },
        routing: {
          signals: { decision: [{ name: 'task', question: { type: 'choice' } }] },
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
    activate: () => {
      observed = global.model_catalog.system.decision_model
      pending = false
    },
  }
}

test.describe('Decision model management', () => {
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

  test('opens from System and shows model, hardware, runtime and binding details', async ({
    page,
  }) => {
    await mockDecisionModelManager(page)
    await page.goto('/status')
    await page.getByRole('button', { name: 'System', exact: true }).click()
    const menu = page.getByRole('navigation', { name: 'System' })
    await menu.getByRole('tab', { name: /Runtime/ }).click()
    await menu.getByRole('link', { name: 'Decision Model', exact: true }).click()
    await expect(page).toHaveURL('/decision-model')
    await expect(page.getByRole('heading', { name: 'Decision Model', exact: true })).toBeVisible()
    await expect(page.getByRole('radio')).toHaveCount(5)
    await expect(page.getByRole('radio', { name: /Vela 2.0 4B/ })).toBeChecked()
    const status = page.getByRole('region', { name: 'Deployment status' })
    await expect(status).toContainText('Matching runtime ready')
    await expect(status).toContainText('Router')
    const deployments = page.getByRole('region', { name: 'Model runtime deployments' })
    for (const value of [
      'revision-123',
      'rocm:0',
      'torch',
      'exact',
      'model-runtime-1',
      'safety (classification)',
    ]) {
      await expect(deployments).toContainText(value)
    }
    await expect(page.getByRole('region', { name: 'Bindings and questions' })).toContainText(
      'task (choice)',
    )
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
    await expect(page.getByRole('region', { name: 'Bindings and questions' })).toContainText(
      'models/custom-pii',
    )
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
    await mockDecisionModelManager(page, { readonly: true })
    await page.goto('/decision-model')
    await expect(page.getByRole('heading', { name: 'Decision Model', exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeDisabled()
    await expect(page.getByRole('radio').first()).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Refresh', exact: true })).toBeEnabled()
  })

  test('does not apply router model settings to a standalone engine', async ({ page }) => {
    const fixture = await mockDecisionModelManager(page, { engine: true })
    await page.goto('/decision-model')
    await expect(page.getByRole('region', { name: 'Deployment status' })).toContainText('Engine')
    await expect(page.getByText(/This deployment is a standalone engine/)).toBeVisible()
    await expect(page.getByRole('button', { name: 'Deploy selected model' })).toBeDisabled()
    await expect(page.getByRole('radio').first()).toBeDisabled()
    expect(fixture.requests).toEqual([])
  })
})
