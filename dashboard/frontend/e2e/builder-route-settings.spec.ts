import { expect } from '@playwright/test'

import { mockAuthenticatedAppShell, test } from './support/compiler'

// Uses the production Go compiler through its HTTP handler; only server data is a fixture.
const config = `version: v0.5
providers:
  models:
    - name: model-a
      reasoning:
        family: qwen3
      backend_refs:
        - name: local
          endpoint: localhost:8000
          protocol: http
    - name: model-b
      backend_refs:
        - name: local
          endpoint: localhost:8001
          protocol: http
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_injection
          predicate: {gte: 0.8}
    decisions:
      - name: guard_route
        description: Route prompt attacks to a safe model.
        priority: 120
        tier: 2
        rules:
          operator: AND
          on_unknown: no_match
          conditions:
            - type: jailbreak
              name: prompt_injection
        action:
          type: route
          destination: model-a
        modelRefs:
          - model: model-a
            use_reasoning: true
            reasoning_mode: enabled
        candidateIterations:
          - variable: candidate
            source: models
            models:
              - model: model-b
            outputs:
              - type: model
                value: candidate
        retention:
          ttl_turns: 4
`

test('keeps route settings outside the form when a Builder route is saved', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/config/yaml', (route) => route.fulfill({ body: config }))
  await page.goto('/builder')

  await page.getByRole('button', { name: /^guard_route\b/ }).click()
  await page.getByRole('button', { name: 'Save', exact: true }).click()

  await page.getByRole('radio', { name: 'YAML', exact: true }).click()
  const compiled = page.locator('pre').filter({ hasText: 'name: guard_route' })
  for (const setting of [
    'tier: 2',
    'on_unknown: no_match',
    'destination: model-a',
    'reasoning_mode: enabled',
    'variable: candidate',
    'ttl_turns: 4',
  ]) {
    await expect(compiled).toContainText(setting)
  }
})
