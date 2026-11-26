import { expect, test, type Page } from '@playwright/test'
import { join } from 'node:path'
import { mockAuthenticatedAppShell } from './support/auth'

const types = ['choice', 'score', 'noul', 'span', 'set']
const response = {
  model: 'vela-test',
  answers: {
    task: {
      type: 'choice',
      choice: 'code',
      probabilities: { code: 0.93, writing: 0.05, other: 0.02 },
      confidence: 0.93,
    },
    difficulty: { type: 'score', score: 1.64, probabilities: { '0': 0.06, '1': 0.24, '2': 0.7 } },
    needs_reasoning: { type: 'noul', noul: 0.88 },
    entities: { type: 'span' },
    needs: { type: 'set' },
  },
  spans: {
    entities: [{ label: 'person', start: 5, end: 14, text: 'Maya Chen', probability: 0.96 }],
  },
  sets: {
    needs: {
      selected: ['coding', 'reasoning'],
      probabilities: { coding: 0.94, reasoning: 0.91, creative: 0.09 },
    },
  },
  thresholds: { needs: 0.5, entities: 0.5 },
  span_heads: { entities: 'router' },
  usage: { input_tokens: 84, output_tokens: 17 },
  meta: { compute_ms: 24.2, profile: 'exact', engine: 'native', device: 'cpu' },
}

async function mockPlayground(
  page: Page,
  options: {
    reader?: boolean
    classicOnly?: boolean
    unready?: boolean
    sameIdentity?: boolean
    repo?: string
  } = {},
) {
  await mockAuthenticatedAppShell(
    page,
    options.reader
      ? {
          user: {
            id: 'reader',
            email: 'reader@example.com',
            name: 'Reader',
            role: 'read',
            permissions: ['config.read'],
          },
        }
      : {},
  )
  await page.route('**/api/decision-model/capabilities', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        serving_mode: 'router',
        timeout_ms: 30000,
        deployments: [
          {
            id: '@vela/auto',
            model: options.sameIdentity ? '@vela/auto' : 'vela-test',
            repo: options.repo,
            ready: !options.unready,
            question_types: options.classicOnly ? types.slice(0, 3) : types,
            surfaces: ['decisions'],
            ...(options.unready ? { unavailable_reason: 'The runtime is still loading.' } : {}),
          },
        ],
      }),
    }),
  )
  await page.route('**/api/decision-model/tasks', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ tasks: [], deployments: [], bindings: [] }),
    }),
  )
  await page.route('**/api/decision-model/routes', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ available: true, routes: [] }),
    }),
  )
  const requests: Record<string, unknown>[] = []
  await page.route('**/api/decision-model/test', (route) => {
    requests.push(route.request().postDataJSON())
    return route.fulfill({ contentType: 'application/json', body: JSON.stringify(response) })
  })
  await page.goto('/models/decision?tab=playground')
  await expect(page.getByRole('heading', { name: 'Answers', exact: true })).toBeVisible({
    timeout: 30_000,
  })
  await expect(page.getByRole('combobox', { name: 'Runtime target', exact: true })).toContainText(
    options.repo || (options.sameIdentity ? '@vela/auto' : 'vela-test'),
  )
  return requests
}

const runButton = (page: Page) => page.getByRole('button', { name: /^Run (test|\d+ questions)/ })
const state = (page: Page) => page.getByRole('textbox', { name: 'State', exact: true })

async function addType(page: Page, label: string) {
  await page.getByRole('button', { name: 'Add question', exact: true }).click()
  await page.getByRole('menuitem', { name: new RegExp(`^${label}`) }).click()
}

async function switchType(page: Page, question: string, type: string) {
  await page.getByRole('button', { name: new RegExp(`^Question type for ${question}`) }).click()
  await page.getByRole('menuitemradio', { name: new RegExp(`^${type}`) }).click()
}

async function loadExample(page: Page, label: string) {
  await page.getByRole('combobox', { name: 'Load example', exact: true }).click()
  await page.getByRole('option', { name: new RegExp(label) }).click()
}

test('names the actual runtime model while keeping the deployment as the request target', async ({
  page,
}) => {
  const requests = await mockPlayground(page, { sameIdentity: true, repo: 'vllm-sr/Vela-2.0-4B' })
  const target = page.getByRole('combobox', { name: 'Runtime target', exact: true })
  await expect(target).toContainText('vllm-sr/Vela-2.0-4B')
  await target.click()
  const option = page.getByRole('option', { name: /vllm-sr\/Vela-2.0-4B/ })
  await expect(option).toContainText('Deployment: @vela/auto')
  await option.click()
  await runButton(page).click()
  await expect(page.getByRole('article', { name: 'Result for task' })).toBeVisible()
  expect(requests).toHaveLength(1)
  expect(requests[0]).toMatchObject({ deployment: '@vela/auto' })
})

test('shows runtime deployments before task observations in monitoring', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/api/v1/inventory/model-runtime', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ deployments: [] }),
    }),
  )
  await page.route('**/api/decision-model/tasks', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ tasks: [], deployments: [], bindings: [] }),
    }),
  )
  await page.route('**/embedded/prometheus/api/v1/query?*', (route) => {
    const query = new URL(route.request().url()).searchParams.get('query') ?? ''
    const result = query.includes('sr_systemone_stage_total')
      ? [
          {
            metric: {
              algorithm: 'escalate',
              stage: 'fast',
              model: 'decision-kai',
              outcome: 'accepted',
            },
            value: [0, '12'],
          },
          {
            metric: {
              algorithm: 'escalate',
              stage: 'fast',
              model: 'decision-kai',
              outcome: 'rejected',
            },
            value: [0, '3'],
          },
        ]
      : [{ metric: {}, value: [0, '0.2'] }]
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({ status: 'success', data: { resultType: 'vector', result } }),
    })
  })
  await page.goto('/models/decision?tab=monitoring')
  await expect(
    page.getByRole('heading', { name: 'Model runtime deployments', exact: true }),
  ).toBeVisible({ timeout: 30_000 })
  const headings = page.getByRole('heading', {
    name: /^(Model runtime deployments|Automatic routing|Task observations)$/,
  })
  await expect(headings).toHaveText([
    'Model runtime deployments',
    'Automatic routing',
    'Task observations',
  ])
  const auto = page.getByRole('region', { name: 'Automatic routing', exact: true })
  await expect(auto).toContainText('not answer accuracy')
  await expect(auto.getByText('Stage activity', { exact: true })).toBeVisible()
  await expect(auto.locator('.recharts-yAxis')).toContainText(/escalate \/ fast ·\s*decision-kai/)
  if (process.env.PLAYGROUND_SCREENSHOT_DIR)
    await page.screenshot({
      path: join(process.env.PLAYGROUND_SCREENSHOT_DIR, 'systemone-auto-monitoring.png'),
      fullPage: true,
    })
})

test('runs a native request and presents real probabilities with the native JSON one tab away', async ({
  page,
}) => {
  const requests = await mockPlayground(page)
  await expect(page.getByText(/^(Router|Engine) mode$/)).toHaveCount(0)
  await expect(page.getByText('From context to a decision')).toBeVisible()
  await runButton(page).click()
  const result = page.getByRole('article', { name: 'Result for task' })
  await expect(result.getByRole('meter', { name: 'code' })).toHaveAttribute('aria-valuenow', '93')
  await expect(result).toContainText('93.0% confidence')
  expect(requests).toHaveLength(1)
  expect(requests[0]).toMatchObject({
    deployment: '@vela/auto',
    request: {
      options: { return_meta: true },
      questions: {
        task: {
          type: 'choice',
          choices: [
            { key: 'code', description: 'Programming or debugging' },
            { key: 'writing', description: 'Writing or editing prose' },
            { key: 'other', description: 'Any other task' },
          ],
        },
      },
    },
  })
  await page.getByRole('tab', { name: 'Response' }).click()
  await expect(page.getByRole('tabpanel', { name: 'Response' }).locator('pre')).toContainText(
    '"model": "vela-test"',
  )
  await expect(page.getByRole('button', { name: 'Copy response' })).toBeVisible()
  await page.getByRole('tab', { name: 'Answers' }).click()
  await page
    .getByRole('textbox', { name: 'Instructions for task', exact: true })
    .fill('My custom question')
  await switchType(page, 'task', 'Score')
  await switchType(page, 'task', 'Choice')
  await expect(
    page.getByRole('textbox', { name: 'Instructions for task', exact: true }),
  ).toHaveValue('My custom question')
  await expect(
    page.getByText(
      'Showing the last submitted request. Run again to see results for your changes.',
    ),
  ).toBeVisible()
})

test('renders a five-type batch, highlights Unicode spans, and fits desktop and mobile', async ({
  page,
}) => {
  const pageErrors: string[] = []
  page.on('pageerror', (error) => pageErrors.push(error.message))
  await page.setViewportSize({ width: 1440, height: 1000 })
  await mockPlayground(page)
  await loadExample(page, 'Code & reasoning')
  await addType(page, 'Span')
  await addType(page, 'Set')
  await state(page).fill('Hi 👋 Maya Chen! Write and explain a Python sorting algorithm.')
  await page.getByRole('button', { name: /^Run 5 questions/ }).click()
  await expect(page.getByRole('article', { name: 'Result for difficulty' })).toContainText('1.64')
  await expect(page.getByRole('article', { name: 'Result for needs_reasoning' })).toContainText(
    '88.0%',
  )
  await expect(page.getByRole('article', { name: 'Result for needs', exact: true })).toContainText(
    'coding, reasoning selected',
  )
  const spans = page.getByRole('article', { name: 'Result for entities' })
  await expect(spans.getByLabel('Highlighted spans').locator('mark')).toContainText('Maya Chen')
  await expect(spans.getByRole('cell', { name: '5–14', exact: true })).toBeVisible()
  await expect(page.getByRole('article')).toHaveCount(5)
  const inputBounds = await page.getByRole('region', { name: 'State', exact: true }).boundingBox()
  const questionBounds = await page
    .getByRole('region', { name: 'Questions', exact: true })
    .boundingBox()
  const resultBounds = await page.getByRole('tabpanel', { name: 'Answers' }).boundingBox()
  expect(inputBounds!.x + inputBounds!.width).toBeLessThan(resultBounds!.x)
  expect(questionBounds!.x + questionBounds!.width).toBeLessThan(resultBounds!.x)
  if (process.env.PLAYGROUND_SCREENSHOT_DIR)
    await page.screenshot({
      path: join(process.env.PLAYGROUND_SCREENSHOT_DIR, 'systemone-desktop.png'),
      fullPage: true,
    })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await expect(page.getByRole('button', { name: /^Run 5 questions/ })).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Deploy' })).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Monitoring' })).toBeVisible()
  if (process.env.PLAYGROUND_SCREENSHOT_DIR)
    await page.screenshot({
      path: join(process.env.PLAYGROUND_SCREENSHOT_DIR, 'systemone-mobile.png'),
      fullPage: true,
    })
  expect(pageErrors).toEqual([])
})

test('uses discovered question support and runtime readiness to gate inference', async ({
  page,
}) => {
  const requests = await mockPlayground(page, { classicOnly: true, sameIdentity: true })
  await expect(page.getByRole('combobox', { name: 'Runtime target', exact: true })).toHaveText(
    '@vela/auto',
  )
  await page.getByRole('button', { name: /^Question type for task/ }).click()
  await expect(page.getByRole('menuitemradio', { name: /^Span/ })).toBeDisabled()
  await expect(page.getByRole('menuitemradio', { name: /^Set/ })).toBeDisabled()
  await page.keyboard.press('Escape')
  expect(requests).toHaveLength(0)
  await mockPlayground(page, { unready: true })
  await expect(runButton(page)).toBeDisabled()
  await expect(page.getByText('The runtime is still loading.')).toBeVisible()
})

test('allows a configuration reader to author and inspect without running inference', async ({
  page,
}) => {
  const requests = await mockPlayground(page, { reader: true })
  await page.getByLabel('Question 1 name', { exact: true }).fill('my_question')
  await expect(runButton(page)).toBeDisabled()
  await expect(
    page.getByText('Your account needs evaluation.run permission to send a test.'),
  ).toBeVisible()
  await page.getByRole('tab', { name: 'Request' }).click()
  await expect(page.getByRole('tabpanel', { name: 'Request' }).locator('pre')).toContainText(
    'my_question',
  )
  expect(requests).toHaveLength(0)
})

test('shows partial question failures and native service errors without inventing results', async ({
  page,
}) => {
  await mockPlayground(page)
  await page.route('**/api/decision-model/test', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        ...response,
        answers: {
          task: {
            type: 'choice',
            error: 'invalid_question',
            message: 'This question exceeds the model option limit.',
          },
        },
      }),
    }),
  )
  await runButton(page).click()
  await expect(page.getByRole('alert')).toContainText(
    'This question exceeds the model option limit.',
  )
  await expect(page.getByRole('meter')).toHaveCount(0)
  await page.route('**/api/decision-model/test', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({
        error: { code: 'unavailable', message: 'The model runtime is unavailable.' },
      }),
    }),
  )
  await runButton(page).click()
  await expect(page.getByRole('alert')).toContainText('The model runtime is unavailable.')
  await expect(page.getByRole('article')).toHaveCount(0)
  await expect(page.getByLabel('Question 1 name', { exact: true })).toHaveValue('task')
})

test('cancels the browser wait and discards a late response', async ({ page }) => {
  await mockPlayground(page)
  let resolve: (() => void) | undefined
  const deferred = new Promise<void>((done) => {
    resolve = done
  })
  await page.route('**/api/decision-model/test', async (route) => {
    await deferred
    await route
      .fulfill({ contentType: 'application/json', body: JSON.stringify(response) })
      .catch(() => {})
  })
  await runButton(page).click()
  await expect(page.getByText('Your model is thinking…')).toBeVisible()
  await expect(state(page)).toBeDisabled()
  await page.getByRole('button', { name: 'Cancel request' }).click()
  await expect(
    page.getByRole('status').filter({ hasText: 'Stopped waiting for this request.' }),
  ).toBeVisible()
  resolve?.()
  await expect(page.getByRole('article')).toHaveCount(0)
  await expect(runButton(page)).toBeEnabled()
})

test('validates structured input and sends explicit span state fields unchanged', async ({
  page,
}) => {
  const requests = await mockPlayground(page)
  await page.getByRole('radio', { name: 'Structured', exact: true }).click()
  await state(page).fill('{')
  await expect(runButton(page)).toBeDisabled()
  await expect(page.getByText('The structured state is not valid JSON.')).toBeVisible()
  await state(page).fill('{"request":"Who is this?","answer":"Maya"}')
  await switchType(page, 'task', 'Span')
  await page.getByText('Question settings', { exact: true }).first().click()
  await page.getByLabel('State field', { exact: false }).fill('answer')
  await runButton(page).click()
  await expect(page.getByText('Response received')).toBeVisible()
  expect(requests[0]).toMatchObject({
    request: {
      state: { request: 'Who is this?', answer: 'Maya' },
      questions: { task: { type: 'span', over: 'answer' } },
    },
  })
})

test('refreshes failed discovery instead of accepting stale runtime capabilities', async ({
  page,
}) => {
  await mockPlayground(page)
  await page.route('**/api/decision-model/capabilities', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({
        error: { code: 'unavailable', message: 'Cannot discover model capabilities.' },
      }),
    }),
  )
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Cannot discover model capabilities.')
  await expect(runButton(page)).toBeDisabled()
})

test('supports select keyboard navigation and escape without changing drafts', async ({ page }) => {
  await mockPlayground(page)
  const example = page.getByRole('combobox', { name: 'Load example', exact: true })
  await state(page).fill('Keep my draft while I explore examples.')
  await example.focus()
  await example.press('Enter')
  await expect(page.getByRole('listbox')).toBeVisible()
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('Escape')
  await expect(example).toHaveAttribute('aria-expanded', 'false')
  await expect(example).toBeFocused()
  await expect(state(page)).toHaveValue('Keep my draft while I explore examples.')
  await example.press('Enter')
  await page.getByRole('option', { name: /Customer support/ }).press('Enter')
  await expect(state(page)).toHaveValue(/charged twice/)
  await expect(example).toBeFocused()
  const runtime = page.getByRole('combobox', { name: 'Runtime target', exact: true })
  await runtime.focus()
  await runtime.press('Enter')
  await expect(page.getByRole('option', { name: /vela-test/ })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await page.keyboard.press('Escape')
  await expect(runtime).toBeFocused()
})

test('loads the language span example and sends its labels unchanged', async ({ page }) => {
  const requests = await mockPlayground(page)
  await loadExample(page, 'Entity extraction')
  await expect(state(page)).toHaveValue(/Python/)
  await expect(
    page.getByRole('textbox', { name: 'entities label 3 key', exact: true }),
  ).toHaveValue('language')
  await expect(
    page.getByRole('textbox', { name: 'entities label 3 description', exact: true }),
  ).toHaveValue('The programming language')
  await runButton(page).click()
  await expect(page.getByText('Response received')).toBeVisible()
  expect(requests[0]).toMatchObject({
    request: {
      state: expect.stringContaining('Python'),
      questions: {
        entities: {
          type: 'span',
          labels: expect.arrayContaining([
            { key: 'language', description: 'The programming language' },
          ]),
        },
      },
    },
  })
})

for (const colorScheme of ['light', 'dark'] as const) {
  test(`keeps example menus readable and inside the mobile viewport with ${colorScheme} OS preference`, async ({
    browser,
  }) => {
    const context = await browser.newContext({
      viewport: { width: 390, height: 844 },
      hasTouch: true,
      colorScheme,
    })
    const page = await context.newPage()
    await mockPlayground(page)
    await page.getByRole('combobox', { name: 'Load example', exact: true }).tap()
    const menu = page.getByRole('listbox')
    await expect(menu).toBeVisible()
    const bounds = await menu.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390)
    expect(bounds!.y).toBeGreaterThanOrEqual(0)
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(844)
    await page.getByRole('option', { name: /Entity extraction/ }).tap()
    await expect(
      page.getByRole('textbox', { name: 'entities label 3 key', exact: true }),
    ).toHaveValue('language')
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await context.close()
  })
}

test('runs auto through its recipe and preserves the selected example across target changes', async ({
  page,
}) => {
  await mockPlayground(page)
  const submissions: Record<string, unknown>[] = []
  await page.route('**/api/decision-model/routes', (route) => {
    if (route.request().method() === 'POST') {
      submissions.push(route.request().postDataJSON())
      return route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          ...response,
          model: 'vllm-sr/auto',
          routing: {
            recipe: 'decisions',
            decision: 'task',
            algorithm: 'escalate',
            stage: 'strong',
            selected_model: 'decision-eos',
            quality: 'uncalibrated',
            model_calls: 3,
          },
        }),
      })
    }
    return route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        available: true,
        routes: [
          {
            model: 'vllm-sr/auto',
            recipe: 'decisions',
            algorithms: ['escalate'],
            question_types: ['choice', 'score', 'noul'],
            execution_timeout_ms: 120000,
          },
        ],
      }),
    })
  })
  await loadExample(page, 'Code')
  const example = await page
    .getByRole('combobox', { name: 'Load example', exact: true })
    .textContent()
  const input = await state(page).inputValue()
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  const target = page.getByRole('combobox', { name: 'Runtime target', exact: true })
  await target.click()
  await expect(page.getByText('Automatic routes', { exact: true })).toBeVisible()
  await expect(page.getByText('Direct models', { exact: true })).toBeVisible()
  await page.getByRole('option', { name: /vllm-sr\/auto/ }).click()
  await expect(target).toHaveText('vllm-sr/auto')
  await expect(page.getByRole('combobox', { name: 'Load example', exact: true })).toHaveText(
    example!,
  )
  await expect(state(page)).toHaveValue(input)
  await page.getByRole('button', { name: /^Question type for task/ }).click()
  await expect(page.getByRole('menuitemradio', { name: /^Span/ })).toBeDisabled()
  await page.keyboard.press('Escape')
  await runButton(page).click()
  const outcome = page.getByRole('region', { name: 'Auto routing outcome' })
  await expect(outcome).toContainText('decision-eos')
  await expect(outcome).toContainText('strong')
  await expect(outcome).toContainText('Operator thresholds')
  await expect(outcome).toContainText('not measured accuracy')
  expect(submissions[0]).toMatchObject({ model: 'vllm-sr/auto', request: { state: input } })
  expect(submissions[0]).not.toHaveProperty('deployment')
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
})

test('route discovery failure does not block direct inference', async ({ page }) => {
  const direct = await mockPlayground(page)
  await page.route('**/api/decision-model/routes', (route) =>
    route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({ error: { message: 'Auto discovery unavailable' } }),
    }),
  )
  await page.getByRole('button', { name: 'Refresh', exact: true }).click()
  await expect(page.getByText('Auto discovery unavailable')).toBeVisible()
  await runButton(page).click()
  await expect(page.getByText('Response received')).toBeVisible()
  expect(direct).toHaveLength(1)
})

test('slow direct discovery does not delay auto routing', async ({ page }) => {
  await mockPlayground(page)
  let release!: () => void
  const pending = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/api/decision-model/capabilities', async (route) => {
    await pending
    await route.fulfill({ status: 503, body: '{}' })
  })
  await page.route('**/api/decision-model/routes', (route) =>
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        available: true,
        routes: [
          {
            model: 'vllm-sr/auto',
            recipe: 'decisions',
            algorithms: ['systemone'],
            question_types: ['choice', 'score', 'noul'],
            execution_timeout_ms: 5000,
          },
        ],
      }),
    }),
  )
  try {
    await page.reload()
    await expect(page.getByRole('combobox', { name: 'Runtime target', exact: true })).toHaveText(
      'vllm-sr/auto',
      { timeout: 30_000 },
    )
    await expect(runButton(page)).toBeEnabled()
  } finally {
    release()
  }
})
