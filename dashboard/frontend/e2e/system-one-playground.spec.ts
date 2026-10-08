import { expect, test, type Page } from '@playwright/test'
import { join } from 'node:path'
import { mockAuthenticatedAppShell } from './support/auth'

const types = ['choice', 'score', 'noul', 'span', 'set']
const response = {
  model: 'vela-test',
  answers: {
    choice_1: {
      type: 'choice',
      choice: 'code',
      probabilities: { code: 0.93, writing: 0.05, other: 0.02 },
      confidence: 0.93,
    },
    score_1: { type: 'score', score: 1.64, probabilities: { '0': 0.06, '1': 0.24, '2': 0.7 } },
    noul_1: { type: 'noul', noul: 0.88 },
    span_4: { type: 'span' },
    set_5: { type: 'set' },
  },
  spans: { span_4: [{ label: 'person', start: 5, end: 14, text: 'Maya Chen', probability: 0.96 }] },
  sets: {
    set_5: {
      selected: ['coding', 'reasoning'],
      probabilities: { coding: 0.94, reasoning: 0.91, creative: 0.09 },
    },
  },
  thresholds: { set_5: 0.5, span_4: 0.5 },
  span_heads: { span_4: 'router' },
  usage: { input_tokens: 84, output_tokens: 17 },
  meta: { compute_ms: 24.2, profile: 'exact', engine: 'native', device: 'cpu' },
}

async function mockPlayground(
  page: Page,
  options: { reader?: boolean; classicOnly?: boolean; unready?: boolean } = {},
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
            model: 'vela-test',
            ready: !options.unready,
            question_types: options.classicOnly ? types.slice(0, 3) : types,
            surfaces: ['decisions'],
            ...(options.unready ? { unavailable_reason: 'The runtime is still loading.' } : {}),
          },
        ],
      }),
    }),
  )
  const requests: Record<string, unknown>[] = []
  await page.route('**/api/decision-model/test', (route) => {
    requests.push(route.request().postDataJSON())
    return route.fulfill({ contentType: 'application/json', body: JSON.stringify(response) })
  })
  await page.goto('/decision-model/playground')
  await expect(
    page.getByRole('heading', { name: 'Decision Model Test', exact: true }),
  ).toBeVisible()
  await expect(page.getByLabel('Runtime target', { exact: true })).toContainText('vela-test')
  return requests
}

async function addType(page: Page, label: string) {
  await page.getByRole('button', { name: '+ Add question', exact: true }).click()
  await page
    .getByLabel('Add a question type')
    .getByRole('button', { name: new RegExp(label) })
    .click()
}

test('runs a native request and presents real probabilities with API details collapsed', async ({
  page,
}) => {
  const requests = await mockPlayground(page)
  await expect(page.getByText('From context to a decision')).toBeVisible()
  await page.getByRole('button', { name: 'Run test', exact: true }).click()
  const result = page.getByRole('article', { name: 'Result for choice_1' })
  await expect(result.getByRole('meter', { name: 'code' })).toHaveAttribute('aria-valuenow', '93')
  await expect(result).toContainText('93.0% confidence')
  expect(requests).toHaveLength(1)
  expect(requests[0]).toMatchObject({
    deployment: '@vela/auto',
    request: {
      options: { return_meta: true },
      questions: {
        choice_1: {
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
  const inspector = page
    .locator('details')
    .filter({ has: page.getByText('Inspect response', { exact: true }) })
  await expect(inspector).not.toHaveAttribute('open', '')
  await inspector.locator('summary').click()
  await expect(inspector.locator('pre')).toContainText('"model": "vela-test"')
  await page.evaluate(() =>
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: undefined }),
  )
  await inspector.getByRole('button', { name: 'Copy JSON' }).click()
  await expect(inspector.getByRole('button', { name: 'Copy unavailable' })).toBeVisible()
  await page.getByRole('textbox', { name: 'Instructions', exact: true }).fill('My custom question')
  await page
    .getByLabel('Question type')
    .getByRole('button', { name: 'Score', exact: false })
    .click()
  await page
    .getByLabel('Question type')
    .getByRole('button', { name: 'Choice', exact: false })
    .click()
  await expect(page.getByRole('textbox', { name: 'Instructions', exact: true })).toHaveValue(
    'My custom question',
  )
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
  await page.getByLabel('Load example', { exact: true }).selectOption('coding')
  await addType(page, 'Span')
  await addType(page, 'Set')
  await page
    .getByRole('textbox', { name: 'Input context', exact: true })
    .fill('Hi 👋 Maya Chen! Write and explain a Python sorting algorithm.')
  await page.getByRole('button', { name: 'Run 5 questions', exact: true }).click()
  await expect(page.getByRole('article', { name: 'Result for score_1' })).toContainText('1.64')
  await expect(page.getByRole('article', { name: 'Result for noul_1' })).toContainText('88.0%')
  await expect(page.getByRole('article', { name: 'Result for set_5' })).toContainText('✓ reasoning')
  const spans = page.getByRole('article', { name: 'Result for span_4' })
  await expect(spans.getByLabel('Highlighted spans').locator('mark')).toContainText('Maya Chen')
  await expect(spans.getByRole('cell', { name: '5–14', exact: true })).toBeVisible()
  await expect(page.getByRole('article')).toHaveCount(5)
  if (process.env.PLAYGROUND_SCREENSHOT_DIR)
    await page.screenshot({
      path: join(process.env.PLAYGROUND_SCREENSHOT_DIR, 'systemone-desktop.png'),
      fullPage: true,
    })
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await expect(page.getByRole('button', { name: 'Run 5 questions' })).toBeVisible()
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
  const requests = await mockPlayground(page, { classicOnly: true })
  await expect(
    page.getByLabel('Question type').getByRole('button', { name: 'Span', exact: false }),
  ).toBeDisabled()
  await expect(
    page.getByLabel('Question type').getByRole('button', { name: 'Set', exact: false }),
  ).toBeDisabled()
  expect(requests).toHaveLength(0)
  await mockPlayground(page, { unready: true })
  await expect(page.getByRole('button', { name: 'Run test', exact: true })).toBeDisabled()
  await expect(page.getByText('The runtime is still loading.')).toBeVisible()
})

test('allows a configuration reader to author and inspect without running inference', async ({
  page,
}) => {
  const requests = await mockPlayground(page, { reader: true })
  await page.getByLabel('Question name', { exact: true }).fill('my_question')
  await expect(page.getByRole('button', { name: 'Run test', exact: true })).toBeDisabled()
  await expect(
    page.getByText('Your account needs evaluation.run permission to send a test.'),
  ).toBeVisible()
  await page.getByText('Preview request', { exact: true }).click()
  await expect(page.locator('pre')).toContainText('my_question')
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
          choice_1: {
            type: 'choice',
            error: 'invalid_question',
            message: 'This question exceeds the model option limit.',
          },
        },
      }),
    }),
  )
  await page.getByRole('button', { name: 'Run test', exact: true }).click()
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
  await page.getByRole('button', { name: 'Run test', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('The model runtime is unavailable.')
  await expect(page.getByRole('article')).toHaveCount(0)
  await expect(page.getByLabel('Question name', { exact: true })).toHaveValue('choice_1')
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
  await page.getByRole('button', { name: 'Run test', exact: true }).click()
  await expect(page.getByText('Your model is thinking')).toBeVisible()
  await expect(page.getByRole('textbox', { name: 'Input context', exact: true })).toBeDisabled()
  await page.getByRole('button', { name: 'Cancel request' }).click()
  await expect(page.getByRole('status')).toContainText('Stopped waiting for this request.')
  resolve?.()
  await expect(page.getByRole('article')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Run test', exact: true })).toBeEnabled()
})

test('validates structured input and sends explicit span state fields unchanged', async ({
  page,
}) => {
  const requests = await mockPlayground(page)
  await page.getByRole('button', { name: 'Structured', exact: true }).click()
  await page.getByRole('textbox', { name: 'Input context', exact: true }).fill('{')
  await expect(page.getByRole('button', { name: 'Run test', exact: true })).toBeDisabled()
  await expect(page.getByText('The structured state is not valid JSON.')).toBeVisible()
  await page
    .getByRole('textbox', { name: 'Input context', exact: true })
    .fill('{"request":"Who is this?","answer":"Maya"}')
  await page.getByLabel('Question type').getByRole('button', { name: 'Span', exact: false }).click()
  await page.getByText('Question settings', { exact: true }).click()
  await page.getByLabel('State field', { exact: false }).fill('answer')
  await page.getByRole('button', { name: 'Run test', exact: true }).click()
  await expect(page.getByText('Response received')).toBeVisible()
  expect(requests[0]).toMatchObject({
    request: {
      state: { request: 'Who is this?', answer: 'Maya' },
      questions: { choice_1: { type: 'span', over: 'answer' } },
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
  await expect(page.getByRole('button', { name: 'Run test', exact: true })).toBeDisabled()
})
