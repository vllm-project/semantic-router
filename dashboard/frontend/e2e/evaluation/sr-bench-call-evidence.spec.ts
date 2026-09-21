import { expect, test, type Locator, type Page, type Route } from '@playwright/test'
import { mockAuthenticatedAppShell } from '../support/auth'

const target = {
  id: 'single',
  kind: 'single',
  model: 'fixture/subject-model',
  base_url: 'http://localhost:8000/v1',
}
const hostile =
  '<img src="https://evidence.invalid/image" onerror="window.evidenceExecuted=true">' +
  '<script>window.evidenceExecuted=true</script> [open](javascript:alert(1))'
const hiddenReasoning = 'HIDDEN-REASONING-MUST-NOT-BE-DISPLAYED'
const wrongMessages = 'LEGACY-MESSAGES-MUST-NOT-REPLACE-EFFECTIVE-REQUEST'
const tools = Array.from({ length: 12 }, (_, index) => ({
  id: `tool-${index + 1}`,
  type: 'function',
  function: {
    name: `inspect_${index + 1}`,
    arguments:
      index === 11
        ? '{ malformed: arguments'
        : JSON.stringify({ command: `synthetic-command-${index + 1}`, options: { dry_run: true } }),
  },
}))
const messages = Array.from({ length: 12 }, (_, index) => ({
  role: index % 2 === 0 ? 'user' : 'assistant',
  content: `Saved message ${index + 1}${index === 0 ? `: ${hostile}` : ''}`,
  ...(index === 1 ? { tool_calls: tools, reasoning_content: hiddenReasoning } : {}),
}))
const subject = {
  id: 'subject-call',
  case_id: 'synthetic-task',
  target_id: 'single',
  role: 'subject',
  status: 'completed',
  model: target.model,
  cost_usd: 0.0123,
  latency_s: 4.5,
  ttft_s: 0.25,
  finish_reason: 'tool_calls',
  output_complete: true,
  usage: { input_tokens: 123, cached_input_tokens: 0, cache_write_tokens: 7, output_tokens: 42 },
  native_output: {
    policy: 'native',
    source: 'provider_render',
    model: target.model,
    input_tokens: 2621,
    context_window: 32768,
    max_output_tokens: 30147,
    configured_max_output_tokens: 32768,
    provider_model_observed: true,
  },
  request: {
    messages: [{ role: 'user', content: wrongMessages }],
    effective_body: {
      model: target.model,
      messages,
      max_tokens: 777,
      temperature: 0,
      top_p: 1,
      reasoning_effort: 'xhigh',
      parallel_tool_calls: false,
    },
  },
  final: `Visible synthetic response. ${hostile}`,
  reasoning: hiddenReasoning,
  tool_calls: tools,
}
const simulator = {
  id: 'simulator-call',
  case_id: 'synthetic-task',
  target_id: 'single',
  role: 'simulator',
  status: 'completed',
  model: 'fixture/simulator-model',
  cost_usd: 0,
  latency_s: 0,
  finish_reason: 'stop',
  usage: { input_tokens: 0, cached_input_tokens: null, output_tokens: 0 },
  request: {
    effective_body: {
      messages: [{ role: 'user', content: 'Only simulator input.' }],
      temperature: 0,
    },
  },
  final: 'Only simulator response.',
}

async function fixture(page: Page) {
  await mockAuthenticatedAppShell(page)
  const run = {
    id: 'call-evidence-run',
    status: 'completed',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:01:00Z',
    progress: { total: 1, completed: 1, failed: 0 },
    manifest: {
      version: 'sr-bench-1.0',
      name: 'Synthetic saved call evidence',
      mode: 'live',
      profile: 'smoke',
      seed: 42,
      targets: [target],
      sampling: { temperature: 0 },
      output_policy: 'native',
      limits: {},
      cases: [{ id: 'synthetic-task', benchmark: 'tau3' }],
    },
  }
  const state = {
    writes: [] as string[],
    unexpected: [] as string[],
    external: [] as string[],
    holdSubject: null as null | ((route: Route) => Promise<void>),
  }
  await page.route('https://evidence.invalid/**', async (route) => {
    state.external.push(route.request().url())
    await route.abort()
  })
  await page.route('**/api/sr-bench/v1/**', async (route) => {
    const request = route.request()
    const endpoint = new URL(request.url()).pathname.replace('/api/sr-bench/v1', '')
    if (request.method() !== 'GET') state.writes.push(endpoint)
    let body: unknown
    if (endpoint === '/catalog') body = { version: 'sr-bench-1.0', profiles: [], benchmarks: [] }
    else if (endpoint === '/targets') body = { targets: [target] }
    else if (endpoint === '/datasets') body = { datasets: [] }
    else if (endpoint === '/runs') body = { runs: [run] }
    else if (endpoint === `/runs/${run.id}`) body = run
    else if (endpoint === `/runs/${run.id}/report`)
      body = {
        version: 'sr-bench-1.0',
        run_id: run.id,
        status: 'completed',
        summary: { targets: [], wall_time_s: 60 },
        benchmarks: [],
        limitations: [],
        provenance: {},
      }
    else if (endpoint === `/runs/${run.id}/results`)
      body = { results: [], total: 0, next_cursor: null, limit: 100 }
    else if (endpoint === `/runs/${run.id}/events`) body = { events: [] }
    else if (endpoint === `/runs/${run.id}/calls`)
      body = {
        calls: [subject, simulator].map((call) => ({
          id: call.id,
          case_id: call.case_id,
          target_id: call.target_id,
          role: call.role,
          status: call.status,
          model: call.model,
          cost_usd: call.cost_usd,
          latency_s: call.latency_s,
        })),
        total: 2,
        next_cursor: null,
        limit: 100,
      }
    else if (endpoint === `/runs/${run.id}/calls/${subject.id}`) {
      if (state.holdSubject) return state.holdSubject(route)
      body = subject
    } else if (endpoint === `/runs/${run.id}/calls/${simulator.id}`) body = simulator
    else {
      state.unexpected.push(endpoint)
      return route.fulfill({ status: 404, json: { error: 'Unexpected fixture request' } })
    }
    await route.fulfill({ json: body })
  })
  await page.goto(`/evaluation?view=runs&run=${run.id}`)
  await expect(page.getByRole('heading', { name: run.manifest.name, exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Calls', exact: true }).click()
  return state
}

const callPanel = (page: Page) => page.getByRole('tabpanel', { name: 'Calls', exact: true })
const section = (page: Page, name: string) =>
  callPanel(page)
    .locator('section')
    .filter({ has: page.getByRole('heading', { name, exact: true }) })
    .last()
const value = (scope: Locator, name: string) =>
  scope
    .locator('dt')
    .filter({ hasText: new RegExp(`^${name}$`) })
    .locator('..')
    .locator('dd')
const conversation = (page: Page) =>
  callPanel(page)
    .locator('details')
    .filter({
      has: page.locator('summary').filter({ hasText: /^Conversation sent with this call$/ }),
    })

async function openCall(page: Page, id = subject.id) {
  await page.getByRole('button', { name: id, exact: true }).click()
  await expect(
    page.getByRole('heading', { name: `Call evidence: ${id}`, exact: true }),
  ).toBeVisible()
  await expect(section(page, 'Visible response')).toBeVisible()
}

test('native call details distinguish actual output capacity from request settings and preserve four buckets', async ({
  page,
}) => {
  const state = await fixture(page)
  await openCall(page)
  const budget = section(page, 'Native output budget')
  await expect(value(budget, 'Input tokens')).toHaveText('2,621')
  await expect(value(budget, 'Context window')).toHaveText('32,768')
  await expect(value(budget, 'Available output tokens')).toHaveText('30,147')
  await expect(value(budget, 'Registered output maximum')).toHaveText('32,768')
  await expect(budget).not.toContainText('777')
  const settings = section(page, 'Recorded request settings')
  await expect(settings).toContainText('xhigh')
  await expect(value(settings, 'Temperature')).toHaveText('0')
  const usage = section(page, 'Token usage')
  await expect(value(usage, 'Input tokens')).toHaveText('123')
  await expect(value(usage, 'Cached input tokens')).toHaveText('0')
  await expect(value(usage, 'Cache write tokens')).toHaveText('7')
  await expect(value(usage, 'Output tokens')).toHaveText('42')
  await expect(callPanel(page)).toContainText(target.model)
  await expect(
    callPanel(page)
      .locator('span')
      .filter({ hasText: /^Role$/ })
      .locator('..')
      .locator('strong'),
  ).toHaveText('subject')
  await expect(callPanel(page)).toContainText('$0.0123')
  await expect(callPanel(page)).toContainText('4.5 s')
  await expect(conversation(page)).not.toHaveAttribute('open', '')
  expect(state.writes).toEqual([])
  expect(state.unexpected).toEqual([])
})

test('tools and exact sent messages paginate independently and reset when a call is reopened', async ({
  page,
}) => {
  const state = await fixture(page)
  await openCall(page)
  const toolSection = section(page, 'Tool calls')
  await expect(toolSection).toContainText('inspect_1')
  await expect(toolSection).not.toContainText('inspect_11')
  await expect(toolSection.getByRole('navigation', { name: 'Tool calls pages' })).toContainText(
    '1–10 of 12',
  )
  await toolSection.getByRole('button', { name: 'Next tool calls', exact: true }).click()
  await expect(toolSection).toContainText('inspect_11')
  await expect(toolSection.getByText('synthetic-command-1', { exact: true })).toHaveCount(0)
  await toolSection
    .locator('summary')
    .filter({ hasText: /^inspect_12/ })
    .click()
  await expect(toolSection.getByText('{ malformed: arguments', { exact: true })).toBeVisible()

  const sent = conversation(page)
  await sent.locator(':scope > summary').click()
  await expect(sent).toContainText('Saved message 1:')
  await expect(sent).not.toContainText('Saved message 11')
  await expect(sent).not.toContainText(wrongMessages)
  await expect(sent).not.toContainText(hiddenReasoning)
  await expect(sent.getByRole('navigation', { name: 'Tool calls pages' })).toContainText(
    '1–10 of 12',
  )
  await sent.getByRole('button', { name: 'Next tool calls', exact: true }).click()
  await expect(sent).toContainText('inspect_12')
  await sent.getByRole('button', { name: 'Next messages', exact: true }).click()
  await expect(sent).toContainText('Saved message 11')
  await expect(sent).not.toContainText('Saved message 1:')
  await expect(sent.getByRole('navigation', { name: 'Messages pages' })).toContainText(
    '11–12 of 12',
  )

  await page.getByRole('button', { name: 'Back to calls', exact: true }).click()
  await openCall(page, simulator.id)
  await expect(callPanel(page)).not.toContainText('Visible synthetic response.')
  await expect(section(page, 'Visible response')).toContainText(simulator.final)
  await page.getByRole('button', { name: 'Back to calls', exact: true }).click()
  await openCall(page)
  await section(page, 'Tool calls')
    .locator('summary')
    .filter({ hasText: /^inspect_1function$/ })
    .click()
  await expect(
    section(page, 'Tool calls').getByText('synthetic-command-1', { exact: true }),
  ).toBeVisible()
  await expect(conversation(page)).not.toHaveAttribute('open', '')
  await conversation(page).locator(':scope > summary').click()
  await expect(conversation(page)).toContainText('Saved message 1:')
  await expect(conversation(page)).not.toContainText('Saved message 11')
  expect(state.writes).toEqual([])
})

test('simulator evidence keeps missing usage distinct from recorded zero', async ({ page }) => {
  const state = await fixture(page)
  await openCall(page, simulator.id)
  await expect(
    callPanel(page)
      .locator('span')
      .filter({ hasText: /^Role$/ })
      .locator('..')
      .locator('strong'),
  ).toHaveText('simulator')
  await expect(callPanel(page)).toContainText(simulator.model)
  const usage = section(page, 'Token usage')
  await expect(value(usage, 'Input tokens')).toHaveText('0')
  await expect(value(usage, 'Output tokens')).toHaveText('0')
  await expect(value(usage, 'Cached input tokens')).toHaveText('Not recorded')
  await expect(value(usage, 'Cache write tokens')).toHaveText('Not recorded')
  await expect(section(page, 'Visible response')).toHaveText(/Only simulator response\./)
  expect(state.writes).toEqual([])
})

test('saved text is inert and desktop and mobile call details fit the viewport', async ({
  page,
}, testInfo) => {
  const state = await fixture(page)
  await openCall(page)
  await conversation(page).locator(':scope > summary').click()
  const response = section(page, 'Visible response')
  await expect(response.locator('pre')).toContainText(hostile)
  await expect(response.locator('img, script, a, iframe')).toHaveCount(0)
  await expect(conversation(page).locator('img, script, a, iframe')).toHaveCount(0)
  await expect(callPanel(page).getByText(hiddenReasoning, { exact: false })).toHaveCount(0)
  const toolSection = section(page, 'Tool calls')
  await toolSection
    .locator('summary')
    .filter({ hasText: /^inspect_1function$/ })
    .click()
  await expect(toolSection.locator('dl').first()).toContainText('synthetic-command-1')
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 })
    await expect
      .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
      .toBe(true)
    await callPanel(page).screenshot({ path: testInfo.outputPath(`call-evidence-${width}.png`) })
  }
  expect(await page.evaluate(() => 'evidenceExecuted' in window)).toBe(false)
  expect(state.external).toEqual([])
  expect(state.writes).toEqual([])
})

test('a delayed detail from a previous selection cannot overwrite the current call', async ({
  page,
}) => {
  const state = await fixture(page)
  let release: () => void = () => {}
  let entered = false
  let settled = false
  state.holdSubject = async (route) => {
    entered = true
    await new Promise<void>((resolve) => {
      release = resolve
    })
    await route.fulfill({ json: subject }).catch(() => {})
    settled = true
  }
  await page.getByRole('button', { name: subject.id, exact: true }).click()
  await expect.poll(() => entered).toBe(true)
  await page.getByRole('button', { name: 'Back to calls', exact: true }).click()
  await openCall(page, simulator.id)
  release()
  await expect.poll(() => settled).toBe(true)
  await expect(section(page, 'Visible response')).toContainText(simulator.final)
  await expect(callPanel(page)).not.toContainText('Visible synthetic response.')
  await expect(callPanel(page)).not.toContainText('inspect_1')
  await expect(
    page.getByRole('heading', { name: `Call evidence: ${simulator.id}`, exact: true }),
  ).toBeVisible()
  expect(state.writes).toEqual([])
})

test('malformed retained request entries remain inspectable without hiding valid neighbors', async ({
  page,
}) => {
  await fixture(page)
  await page.route(
    '**/api/sr-bench/v1/runs/call-evidence-run/calls/subject-call',
    async (route) => {
      await route.fulfill({
        json: {
          ...subject,
          status: 'failed',
          final: null,
          tool_calls: [
            null,
            { type: 'function', function: { name: 'malformed_arguments', arguments: {} } },
          ],
          request: {
            effective_body: {
              messages: [
                {
                  role: 'user',
                  content: [
                    null,
                    { type: 'text', text: 'Valid neighboring text' },
                    { type: 'image_url', image_url: { url: 'https://evidence.invalid/image' } },
                  ],
                },
                { role: 'assistant', content: null, tool_calls: [null] },
                { role: 'assistant', content: null, tool_calls: 'invalid calls' },
              ],
            },
          },
        },
      })
    },
  )
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await openCall(page)
  await expect(section(page, 'Visible response')).toContainText('No visible final text recorded.')
  await expect(section(page, 'Tool calls')).toContainText('Invalid recorded tool call.')
  await section(page, 'Tool calls').locator('summary').click()
  await expect(section(page, 'Tool calls')).toContainText('Invalid recorded tool arguments.')
  await conversation(page).locator(':scope > summary').click()
  await expect(conversation(page)).toContainText('Invalid recorded content part.')
  await expect(conversation(page)).toContainText('Valid neighboring text')
  await expect(conversation(page)).toContainText('Invalid recorded tool calls.')
  await expect(conversation(page).locator('img')).toHaveCount(0)
  expect(errors).toEqual([])
})
