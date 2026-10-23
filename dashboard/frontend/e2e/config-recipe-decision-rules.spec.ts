import { expect, test, type Page } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'

// A named recipe's nested rule is edited through the shared decision-rule
// editor, saved, and reloaded from the saved body. The sibling recipe, the
// entrypoints and the decision's non-rule policy must come back unchanged.

const nestedRules = {
  operator: 'AND',
  conditions: [
    {
      operator: 'OR',
      conditions: [
        { type: 'classifier', name: 'intent', label: 'a', predicate: { gte: 0.6 } },
        { type: 'keyword', name: 'urgent' },
      ],
    },
    { operator: 'NOT', conditions: [{ type: 'keyword', name: 'spam' }] },
  ],
}

const balancedRoute = {
  name: 'balanced-route',
  description: 'Nested balanced route',
  priority: 100,
  tier: 2,
  rules: nestedRules,
  modelRefs: [{ model: 'model-a', use_reasoning: false, lora_name: 'adapter-a' }],
  algorithm: { type: 'confidence', confidence: { confidence_method: 'avg_logprob' } },
  plugins: [{ type: 'system_prompt', configuration: { system_prompt: 'Be brief.' } }],
  annotations: { owner: 'routing-team' },
  emits: [{ kind: 'retention', retention: { ttl_turns: 2 } }],
}

const privacyRecipe = {
  name: 'privacy',
  description: 'Private objective',
  routing: {
    signals: { pii: [{ name: 'private-pii', threshold: 0.8 }] },
    decisions: [
      {
        name: 'private-route',
        description: 'Private route',
        priority: 100,
        rules: { operator: 'AND', conditions: [{ type: 'pii', name: 'private-pii' }] },
        modelRefs: [{ model: 'model-a', use_reasoning: false }],
      },
    ],
  },
}

const entrypoints = [
  { model_names: ['vllm-sr/balanced'], recipe: 'balanced' },
  { model_names: ['vllm-sr/private'], recipe: 'privacy' },
]

const initialConfig = {
  version: 'v0.3',
  providers: { defaults: { default_model: 'model-a' }, models: [{ name: 'model-a' }] },
  routing: {
    modelCards: [{ name: 'model-a' }],
    signals: {
      keywords: [{ name: 'root-only', operator: 'OR', keywords: ['root'], case_sensitive: false }],
    },
    decisions: [],
  },
  entrypoints,
  recipes: [
    {
      name: 'balanced',
      description: 'Balanced objective',
      routing: {
        signals: {
          classifiers: [{ name: 'intent', type: 'local', labels: ['a', 'b'] }],
          keywords: [
            { name: 'urgent', operator: 'OR', keywords: ['urgent'], case_sensitive: false },
            { name: 'spam', operator: 'OR', keywords: ['spam'], case_sensitive: false },
          ],
        },
        decisions: [
          {
            name: 'balanced-first',
            description: 'First',
            priority: 300,
            rules: { type: 'keyword', name: 'urgent' },
            modelRefs: [{ model: 'model-a', use_reasoning: false }],
          },
          balancedRoute,
          {
            name: 'balanced-last',
            description: 'Last',
            priority: 10,
            rules: {},
            modelRefs: [{ model: 'model-a', use_reasoning: false }],
          },
        ],
      },
    },
    privacyRecipe,
  ],
}

type SavedConfig = typeof initialConfig

const editedExpression =
  '(classifier("intent", label: "b", predicate: { gte: 0.7 }, on_error: "match") OR keyword("urgent")) AND NOT keyword("spam")'

async function mockStatefulConfig(page: Page, saved: SavedConfig[]) {
  await mockAuthenticatedAppShell(page)
  // config/all serves the last saved body, so a reload reads what was written.
  await page.route('**/api/router/config/all', (route) =>
    route.fulfill({ json: saved.at(-1) ?? initialConfig }),
  )
  await page.route('**/api/router/config/global', (route) => route.fulfill({ json: {} }))
  await page.route('**/api/router/config/update', async (route) => {
    saved.push(route.request().postDataJSON() as SavedConfig)
    await route.fulfill({ json: { success: true } })
  })
}

async function openBalancedRouteEditor(page: Page) {
  await page.goto('/config/decisions')
  await page.getByLabel('Routing profile').selectOption('balanced')
  await page.getByRole('button', { name: 'Edit balanced-route' }).click()
  return page.getByRole('dialog', { name: 'Edit Decision: balanced-route' })
}

test('edits a named recipe nested rule and keeps siblings and policy', async ({ page }) => {
  // Edits, saves, reloads and reopens the editor; under parallel load this
  // exceeds the default 30 s budget.
  test.slow()
  const saved: SavedConfig[] = []
  await mockStatefulConfig(page, saved)
  const editor = await openBalancedRouteEditor(page)

  const result = editor.locator('code').filter({ hasText: 'intent' })
  await expect(result).toHaveText(
    '(classifier("intent", label: "a", predicate: { gte: 0.6 }) OR keyword("urgent")) AND NOT keyword("spam")',
  )

  // The signal picker lists only the balanced recipe's signals.
  await editor.locator('.react-flow__node').filter({ hasText: 'urgent' }).dblclick()
  let leafDialog = page.getByRole('dialog', { name: 'Edit Signal' })
  await expect(leafDialog.getByRole('button', { name: 'spam', exact: true })).toBeVisible()
  await expect(leafDialog.getByRole('button', { name: 'root-only' })).toHaveCount(0)
  await leafDialog.getByLabel('Signal Type').selectOption('pii')
  await expect(leafDialog.getByRole('button', { name: 'private-pii' })).toHaveCount(0)
  await leafDialog.getByRole('button', { name: 'Cancel' }).click()

  // Edit the classifier leaf's label, predicate and on_error.
  await editor.locator('.react-flow__node').filter({ hasText: 'intent' }).dblclick()
  leafDialog = page.getByRole('dialog', { name: 'Edit Signal' })
  await expect(leafDialog.getByLabel('Label')).toHaveValue('a')
  await leafDialog.getByLabel('Label').fill('b')
  await leafDialog.getByLabel('gte', { exact: true }).fill('0.7')
  await leafDialog.getByLabel('On error').selectOption('match')
  await leafDialog.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(result).toHaveText(editedExpression)

  // An incomplete raw expression blocks the save instead of saving a prefix.
  await editor.getByLabel('Edit raw expression').check()
  const raw = editor.getByLabel('Raw expression', { exact: true })
  await expect(raw).toHaveValue(editedExpression)
  await raw.fill('(classifier("intent", label: "b") OR keyword("urgent")')
  const rawAlert = editor.getByRole('alert').filter({ hasText: 'Saving is blocked until the expression is complete' })
  await expect(rawAlert).toContainText('at column')
  await expect(result).toHaveText(editedExpression)
  await editor.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(
    editor.getByRole('alert').filter({ hasText: 'Fix the rule tree before saving' }),
  ).toContainText('Raw expression is incomplete')
  expect(saved).toHaveLength(0)

  // Restoring the complete expression keeps the explicit nested tree.
  await raw.fill(editedExpression)
  await expect(rawAlert).toHaveCount(0)
  await editor.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => saved.length).toBe(1)

  const body = saved[0]
  const balanced = body.recipes.find((recipe) => recipe.name === 'balanced')
  const savedRoute = balanced?.routing.decisions?.find((entry) => entry.name === 'balanced-route')
  expect(savedRoute?.rules).toEqual({
    operator: 'AND',
    conditions: [
      {
        operator: 'OR',
        conditions: [
          { type: 'classifier', name: 'intent', label: 'b', predicate: { gte: 0.7 }, on_error: 'match' },
          { type: 'keyword', name: 'urgent' },
        ],
      },
      { operator: 'NOT', conditions: [{ type: 'keyword', name: 'spam' }] },
    ],
  })
  // Non-rule decision policy and the decision order are unchanged.
  const policy = Object.fromEntries(Object.entries(balancedRoute).filter(([key]) => key !== 'rules'))
  expect(savedRoute).toMatchObject(policy)
  expect(balanced?.routing.decisions?.map((entry) => entry.name)).toEqual([
    'balanced-first',
    'balanced-route',
    'balanced-last',
  ])
  // The sibling recipe and the entrypoints are written back untouched.
  expect(body.recipes.find((recipe) => recipe.name === 'privacy')).toEqual(privacyRecipe)
  expect(body.entrypoints).toEqual(entrypoints)

  // Reload from the saved body and confirm the rule and its metadata.
  await page.reload()
  const reopened = await openBalancedRouteEditor(page)
  await expect(reopened.locator('code').filter({ hasText: 'intent' })).toHaveText(editedExpression)
  await expect(reopened.locator('.react-flow__node')).toHaveCount(6)
  await expect(reopened.locator('.react-flow__node').filter({ hasText: 'intent' })).toContainText(
    'gte 0.7',
  )
  await expect(reopened.locator('.react-flow__node').filter({ hasText: 'intent' })).toContainText(
    'on_error match',
  )
})

test('leaving raw mode with an incomplete draft restores the rule from before raw editing', async ({ page }) => {
  const saved: SavedConfig[] = []
  await mockStatefulConfig(page, saved)
  const editor = await openBalancedRouteEditor(page)
  const original =
    '(classifier("intent", label: "a", predicate: { gte: 0.6 }) OR keyword("urgent")) AND NOT keyword("spam")'
  const result = editor.locator('code').filter({ hasText: 'urgent' })

  await editor.getByLabel('Edit raw expression').check()
  const raw = editor.getByLabel('Raw expression', { exact: true })
  // A complete prefix is committed while typing; the incomplete draft is not.
  await raw.fill('keyword("urgent")')
  await expect(result).toHaveText('keyword("urgent")')
  await raw.fill('keyword("urgent") AND')
  await expect(editor.getByRole('alert').filter({ hasText: 'Saving is blocked' })).toBeVisible()

  await editor.getByLabel('Edit raw expression').uncheck()
  await expect(result).toHaveText(original)
  await editor.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(() => saved.length).toBe(1)
  const savedRoute = saved[0].recipes
    .find((recipe) => recipe.name === 'balanced')
    ?.routing.decisions?.find((entry) => entry.name === 'balanced-route')
  expect(savedRoute?.rules).toEqual(nestedRules)
})

test('Escape leaves the fullscreen rule editor without closing the decision dialog', async ({ page }) => {
  const saved: SavedConfig[] = []
  await mockStatefulConfig(page, saved)
  const editor = await openBalancedRouteEditor(page)

  // Make an unsaved edit so closing the dialog would lose it.
  await editor.locator('.react-flow__node').filter({ hasText: 'urgent' }).click({ button: 'right' })
  await page.getByRole('menuitem', { name: 'Wrap with NOT' }).click()
  const edited =
    '(classifier("intent", label: "a", predicate: { gte: 0.6 }) OR NOT keyword("urgent")) AND NOT keyword("spam")'
  await expect(editor.locator('code').filter({ hasText: 'intent' })).toHaveText(edited)

  await editor.getByTitle('Fullscreen').click()
  const exit = page.getByTitle('Exit fullscreen (Esc)')
  await expect(exit.first()).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(exit).toHaveCount(0)
  await expect(editor).toBeVisible()
  await expect(editor.locator('code').filter({ hasText: 'intent' })).toHaveText(edited)

  // Undo still reaches the editor after the fullscreen remount.
  await editor.getByTitle('Fullscreen').click()
  await expect(exit.first()).toBeVisible()
  await page.keyboard.press('Control+z')
  await page.keyboard.press('Escape')
  await expect(exit).toHaveCount(0)
  await expect(editor).toBeVisible()
  await expect(editor.locator('code').filter({ hasText: 'intent' })).toHaveText(
    '(classifier("intent", label: "a", predicate: { gte: 0.6 }) OR keyword("urgent")) AND NOT keyword("spam")',
  )
})

test('Tab stays inside the fullscreen rule editor', async ({ page }) => {
  const saved: SavedConfig[] = []
  await mockStatefulConfig(page, saved)
  const editor = await openBalancedRouteEditor(page)
  await editor.getByTitle('Fullscreen').click()
  // The overlay header's close button; the canvas toolbar repeats the title.
  const exit = page.getByTitle('Exit fullscreen (Esc)').first()
  await expect(exit).toBeVisible()

  // The decision dialog's own focus trap must not pull focus behind the overlay.
  const focusInOverlay = () =>
    exit.evaluate((button) => {
      const overlay = button.parentElement?.parentElement
      const active = document.activeElement
      return Boolean(overlay && active && active !== document.body && overlay.contains(active))
    })
  await exit.focus()
  for (const key of ['Tab', 'Tab', 'Shift+Tab', 'Shift+Tab', 'Shift+Tab']) {
    await page.keyboard.press(key)
    expect(await focusInOverlay()).toBe(true)
  }
})

test('a rule the Router rejects keeps the dialog open with the Router error', async ({ page }) => {
  const saved: SavedConfig[] = []
  await mockStatefulConfig(page, saved)
  // The update handler parses the config with the Router's validators and
  // answers 400 with their error, as the dashboard backend does.
  const routerError =
    'Config validation failed: routing recipe "balanced": decision "balanced-route": rules.conditions[1]: NOT requires exactly one child condition, got 2'
  await page.route('**/api/router/config/update', (route) =>
    route.fulfill({ status: 400, contentType: 'text/plain', body: routerError }),
  )
  const editor = await openBalancedRouteEditor(page)

  await editor.getByRole('button', { name: 'Save', exact: true }).click()
  await expect(editor.getByText(routerError)).toBeVisible()
  await expect(editor).toBeVisible()
})
