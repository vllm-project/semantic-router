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
  options: {
    reader?: boolean
    classicOnly?: boolean
    unready?: boolean
    sameIdentity?: boolean
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
    page.getByRole('heading', { name: 'Decision Playground', exact: true }),
  ).toBeVisible()
  await expect(page.getByRole('combobox', { name: 'Runtime target', exact: true })).toContainText(
    options.sameIdentity ? '@vela/auto' : 'vela-test',
  )
  return requests
}

async function addType(page: Page, label: string) {
  await page.getByRole('button', { name: '+ Add question', exact: true }).click()
  await page
    .getByLabel('Add a question type')
    .getByRole('button', { name: new RegExp(label) })
    .click()
}

async function loadExample(page: Page, label: string) {
  await page.getByRole('combobox', { name: 'Load example', exact: true }).click()
  await page.getByRole('option', { name: new RegExp(label) }).click()
}

test('runs a native request and presents real probabilities with API details collapsed', async ({
  page,
}) => {
  const requests = await mockPlayground(page)
  await expect(page.getByText('Build / System One', { exact: true })).toBeVisible()
  await expect(page.getByText(/^(Router|Engine) mode$/)).toHaveCount(0)
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
  await loadExample(page, 'Code & reasoning')
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
  const inputBounds = await page
    .getByRole('region', { name: 'Input context', exact: true })
    .boundingBox()
  const questionBounds = await page
    .getByRole('region', { name: 'Questions', exact: true })
    .boundingBox()
  const resultBounds = await page
    .getByRole('region', { name: 'Results', exact: true })
    .boundingBox()
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
  await expect(page.getByRole('button', { name: 'Run 5 questions' })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Decision Models', exact: false })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Decision Monitoring', exact: false })).toBeVisible()
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

test('supports select keyboard navigation, escape, tab, and outside dismissal without changing drafts', async ({
  page,
}) => {
  await mockPlayground(page)
  const example = page.getByRole('combobox', { name: 'Load example', exact: true })
  const context = page.getByRole('textbox', { name: 'Input context', exact: true })
  await context.fill('Keep my draft while I explore examples.')
  await example.focus()
  await example.press('ArrowDown')
  await expect(example).toHaveAttribute('aria-expanded', 'true')
  await expect(example).toBeFocused()
  await expect(page.getByRole('option', { name: /Code & reasoning/ })).toHaveAttribute(
    'data-active',
    'true',
  )
  await example.press('End')
  await expect(page.getByRole('option', { name: /Entity extraction/ })).toHaveAttribute(
    'data-active',
    'true',
  )
  await example.press('Home')
  await example.press('ArrowDown')
  await expect(page.getByRole('option', { name: /Customer support/ })).toHaveAttribute(
    'data-active',
    'true',
  )
  await example.press('Escape')
  await expect(example).toHaveAttribute('aria-expanded', 'false')
  await expect(example).toBeFocused()
  await expect(context).toHaveValue('Keep my draft while I explore examples.')
  await example.press('Enter')
  await example.press('Tab')
  await expect(example).toHaveAttribute('aria-expanded', 'false')
  await expect(example).not.toBeFocused()
  await example.click()
  await page.getByRole('heading', { name: 'Input context', exact: true }).click()
  await expect(example).toHaveAttribute('aria-expanded', 'false')
  await example.focus()
  await example.press('c')
  await expect(example).toHaveAttribute('aria-expanded', 'true')
  await example.press('ArrowDown')
  await example.press('Enter')
  await expect(context).toHaveValue(/charged twice/)
  await expect(example).toBeFocused()

  const runtime = page.getByRole('combobox', { name: 'Runtime target', exact: true })
  await runtime.focus()
  await runtime.press(' ')
  await expect(page.getByRole('option', { name: /vela-test/ })).toHaveAttribute(
    'aria-selected',
    'true',
  )
  await runtime.press('Escape')
  await expect(runtime).toBeFocused()
})

test('loads the language span example and sends its labels unchanged', async ({ page }) => {
  const requests = await mockPlayground(page)
  await loadExample(page, 'Entity extraction')
  await expect(page.getByRole('textbox', { name: 'Input context', exact: true })).toHaveValue(
    /Python/,
  )
  await expect(page.getByRole('textbox', { name: 'label 3 key', exact: true })).toHaveValue(
    'language',
  )
  await expect(page.getByRole('textbox', { name: 'label 3 description', exact: true })).toHaveValue(
    'The programming language',
  )
  await page.getByRole('button', { name: 'Run test', exact: true }).click()
  await expect(page.getByText('Response received')).toBeVisible()
  expect(requests[0]).toMatchObject({
    request: {
      state: expect.stringContaining('Python'),
      questions: {
        span_1: {
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
    const menu = page.getByRole('listbox', { name: 'Load example', exact: true })
    await expect(menu).toBeVisible()
    const bounds = await menu.boundingBox()
    expect(bounds).not.toBeNull()
    expect(bounds!.x).toBeGreaterThanOrEqual(0)
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(390)
    expect(bounds!.y).toBeGreaterThanOrEqual(0)
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(844)
    const contrast = await menu.evaluate((element) => {
      const luminance = (color: string) => {
        const rgb = color
          .match(/[\d.]+/g)!
          .slice(0, 3)
          .map(Number)
          .map((channel) => {
            const value = channel / 255
            return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
          })
        return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
      }
      const background = luminance(getComputedStyle(element).backgroundColor)
      return Array.from(element.querySelectorAll('strong, small')).map((text) => {
        const foreground = luminance(getComputedStyle(text).color)
        return (Math.max(foreground, background) + 0.05) / (Math.min(foreground, background) + 0.05)
      })
    })
    expect(contrast.every((ratio) => ratio >= 4.5)).toBe(true)
    if (process.env.PLAYGROUND_SCREENSHOT_DIR)
      await page.screenshot({
        path: join(
          process.env.PLAYGROUND_SCREENSHOT_DIR,
          `systemone-example-menu-${colorScheme}.png`,
        ),
      })
    await page.getByRole('option', { name: /Entity extraction/ }).tap()
    await expect(page.getByRole('textbox', { name: 'label 3 key', exact: true })).toHaveValue(
      'language',
    )
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true)
    await context.close()
  })
}
