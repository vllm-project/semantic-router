import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

const schemaIndex = {
  contract_version: 'vllm-sr/config-schema/v1',
  config_version: 'v0.3',
  schema_id: 'https://vllm-sr.ai/schemas/router-config-v0.3.schema.json',
  default_view: 'index',
  sections: [
    {
      path: 'global',
      title: 'Global configuration',
      description: 'Shared Router services and behavior.',
      required: false,
      href: '/config/router/schema?view=section&path=global',
    },
  ],
  surfaces: {
    algorithm: {
      count: 1,
      names: ['multi_factor'],
      href_template: '/config/router/schema?view=surface&kind=algorithm&name={name}',
    },
  },
}

const globalSection = {
  $ref: '#/$defs/CanonicalGlobal',
  $defs: {
    CanonicalGlobal: {
      type: 'object',
      title: 'Global configuration',
      required: ['router'],
      properties: {
        router: {
          type: 'object',
          description: 'Runtime routing behavior.',
        },
      },
    },
  },
  'x-vllm-sr-view': { view: 'section', path: 'global' },
}

const algorithmSurface = {
  $ref: '#/$defs/MultiFactorAlgorithm',
  $defs: {
    MultiFactorAlgorithm: {
      type: 'object',
      title: 'Multi-factor selection',
      properties: {
        quality_weight: {
          type: 'number',
          description: 'Relative quality objective weight.',
          minimum: 0,
          maximum: 1,
        },
      },
    },
  },
  'x-vllm-sr-view': {
    view: 'surface',
    kind: 'algorithm',
    name: 'multi_factor',
  },
  'x-vllm-sr-surface': {
    display_name: 'Multi-factor',
    description: 'Balances quality, cost, and latency.',
  },
}

async function mockSchemaReference(page: Page) {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/schema?*', async (route) => {
    const url = new URL(route.request().url())
    const view = url.searchParams.get('view')
    const body =
      view === 'index' ? schemaIndex : view === 'section' ? globalSection : algorithmSurface
    await route.fulfill({
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'X-Vllm-Sr-Schema-Source': 'runtime',
        'X-Vllm-Sr-Schema-Match': 'true',
      },
      body: JSON.stringify(body),
    })
  })
}

test('progressively loads shareable config schema views', async ({ page }) => {
  await mockSchemaReference(page)
  const detailRequests: string[] = []
  page.on('request', (request) => {
    if (
      request.url().includes('/api/router/config/schema?view=') &&
      !request.url().endsWith('view=index')
    ) {
      detailRequests.push(request.url())
    }
  })

  await page.goto('/config/reference')
  await expect(page.getByTestId('config-schema-reference-page')).toBeVisible()
  await expect(page.getByText('Deployed Router', { exact: true })).toBeVisible()
  expect(detailRequests).toHaveLength(0)

  await page.getByRole('button', { name: /Global configuration/ }).click()
  await expect(page).toHaveURL(/\/config\/reference\?section=global$/)
  await expect(page.getByRole('heading', { name: 'global' })).toBeVisible()
  await expect(page.getByText('Runtime routing behavior.', { exact: true })).toBeVisible()

  await page.getByRole('button', { name: /multi_factor/ }).click()
  await expect(page).toHaveURL(/surface=algorithm%3Amulti_factor$/)
  await expect(page.getByRole('heading', { name: 'Multi-factor' })).toBeVisible()
  await expect(
    page.getByText('Balances quality, cost, and latency.', { exact: true }),
  ).toBeVisible()
  expect(detailRequests).toHaveLength(2)
})
