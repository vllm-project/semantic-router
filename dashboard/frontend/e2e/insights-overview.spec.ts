import { expect, test, type Page } from '@playwright/test'

import {
  LONG_SESSION,
  mockHomeOverview,
  mockLongRun,
  mockLongSession,
  mockOverview,
} from './support/insightsOverview'

const HARNESSES = ['Claude Code', 'Codex', 'OpenCode', 'Cursor', 'OpenClaw']

const fitsWidth = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)

test('opens the Breakdown on three cards of one family, with money in one currency', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await mockOverview(page, { mixedCurrency: true })
  await page.goto('/observe/requests')

  // the summary row reads USD only and counts what plain units priced, never adding it
  const summary = page.getByRole('region', { name: 'Requests summary' })
  await expect(summary).toContainText('$14.09')
  await expect(summary).toContainText('2,352 calls unpriced')
  for (const gone of ['more currenc', 'Price units', '80.30']) {
    await expect(summary).not.toContainText(gone)
  }
  // the saving reads against what these requests compared with, and leads with "Cost only",
  // so a narrow tile cuts the share; the Router's own calls sit on their own line
  await expect(summary.getByText('Saved vs configured model', { exact: true })).toBeVisible()
  await expect(summary).toContainText('$18.42')
  await expect(summary.getByText('Cost only · 57.7% less than the configured model')).toBeVisible()
  await expect(summary.getByText('Router’s own calls: $0.61')).toBeVisible()
  await summary.getByRole('button', { name: 'How the estimate is made' }).focus()
  await expect(page.getByRole('tooltip')).toContainText('Priced: 1,322 of 1,464 requests')
  await summary.getByRole('button', { name: 'How the estimate is made' }).blur()
  await summary.getByText('2,352 calls unpriced').hover()
  await expect(page.getByRole('tooltip')).toContainText('The total is in USD only')
  await page.mouse.move(0, 0)

  await page.getByRole('button', { name: 'Breakdown' }).click()
  const breakdown = page.locator('#requests-breakdown')
  await expect(breakdown.getByRole('group', { name: 'Breakdown figures' })).toHaveCount(0)
  for (const text of ['Input tokens', 'Prompt cache', 'Model cost', 'Cost-only estimate']) {
    await expect(breakdown.getByText(text)).toHaveCount(0)
  }
  const cards = ['Agents', 'Models', 'Decisions'].map((name) =>
    breakdown.getByRole('region', { name, exact: true }),
  )
  const tops: number[] = []
  for (const card of cards) {
    await expect(card).toBeVisible()
    // a header icon in the summary row's style, then rows of one anatomy
    await expect(card.locator('h4 svg')).toHaveCount(1)
    await expect(card.locator('li').first().locator('.h-\\[3px\\]')).toHaveCount(1)
    tops.push(Math.round((await card.boundingBox())!.y))
  }
  expect(new Set(tops).size).toBe(1)

  const agents = breakdown.getByRole('list', { name: 'Requests by agent harness' })
  const rows = agents.locator('li[data-harness]')
  await expect(rows).toHaveCount(6)
  await expect(rows.first()).toHaveAttribute('data-harness', 'claude_code')
  await expect(rows.last()).toHaveAttribute('data-harness', 'api_client')
  for (const name of HARNESSES) {
    await expect(agents.getByRole('img', { name })).toBeVisible()
  }
  await expect(agents.locator('li[data-harness="claude_code"]')).toContainText('28.1%')
  const models = breakdown.getByRole('list', { name: 'Requests by model' })
  await expect(models.locator('li').first()).toContainText('61.6%')
  await expect(page.locator('main')).not.toContainText('N/A')
  await breakdown.screenshot({ path: info.outputPath('breakdown-1440.png') })
  await summary.screenshot({ path: info.outputPath('summary-1440.png') })

  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => fitsWidth(page)).toBe(true)
  await breakdown.screenshot({ path: info.outputPath('breakdown-390.png') })
})

test('expands a long session at once: its agents from the summary, then its turns a page at a time', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  const { release, reads } = await mockLongSession(page)
  // the totals above the Sessions view read only the requests that belong to a session
  const sessionsTotals = page.waitForRequest(
    (request) => request.url().includes('/aggregate') && request.url().includes('has_session=true'),
  )
  await page.goto('/observe/requests?view=sessions')
  await sessionsTotals
  await expect(page.getByRole('region', { name: 'Requests summary' })).toContainText(
    'Totals cover the requests that belong to a session',
  )

  const row = page.locator(`[data-session="${LONG_SESSION}"]`)
  await expect(row).toContainText('Port the billing service to the new queue')
  // one currency in the row: USD, with what older plain-unit prices covered left out
  await expect(row).toContainText('$0.334')
  await expect(row).toContainText('partly unpriced')
  await expect(row).not.toContainText('+ 0.21')

  await row.getByRole('button', { expanded: false }).first().click()
  // the agents and their figures paint before the run's calls arrive
  const tree = row.getByRole('list', { name: 'Agents of this session' })
  await expect(tree).toBeVisible()
  await expect(tree.locator('[data-agent="main"]').first()).toContainText('106 calls')
  await expect(tree.locator('[data-agent="subagent 9"]')).toBeVisible()
  await expect(tree.getByLabel('Loading the turns').first()).toBeVisible()
  await page.screenshot({ path: info.outputPath('session-expanding-1440.png') })

  release()
  const main = tree.locator('[data-agent="main"]').first()
  await expect(main.locator('[data-turn]').first()).toBeVisible()
  await expect(
    main.getByRole('list', { name: 'Main agent turns' }).locator(':scope > li'),
  ).toHaveCount(50)
  await expect(main.getByRole('button', { name: 'Show 50 more' }).first()).toBeVisible()
  await expect(tree.locator('[data-turn="long-31"] [data-event="compaction"]')).toContainText(
    'Compaction',
  )
  await main.getByRole('button', { name: 'Show 50 more' }).first().click()
  await expect(
    main.getByRole('list', { name: 'Main agent turns' }).locator(':scope > li'),
  ).toHaveCount(100)
  await page.screenshot({ path: info.outputPath('session-expanded-1440.png'), fullPage: true })

  // the run page opens on the same read
  await row.getByRole('button', { name: 'Open this agent run' }).click()
  await expect(page.getByRole('list', { name: 'Run timeline' })).toBeVisible()
  expect(reads.count).toBe(1)
})

test('scales the run timeline to many agents and calls', async ({ page }, info) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await mockLongRun(page)
  await page.goto(`/observe/runs?session=${encodeURIComponent(LONG_SESSION)}`)

  const timeline = page.getByRole('list', { name: 'Run timeline' })
  await expect(page.getByText('Calls 1–100 of 160')).toBeVisible()
  await expect(timeline.getByRole('listitem')).toHaveCount(6)
  await page.getByRole('button', { name: 'Show 4 more agents' }).click()
  await expect(timeline.getByRole('listitem')).toHaveCount(10)
  await expect(page.locator('#run-timeline')).toContainText(/25m( \d+s)? idle/)

  // the events strip marks each event apart, compaction ruled through the lanes and named
  const events = page.getByRole('list', { name: 'Run events' })
  for (const event of ['compaction', 'subagent_start', 'image', 'failure', 'switch']) {
    await expect(events.locator(`[data-event="${event}"]`).first()).toBeVisible()
  }
  await expect(page.locator('[data-compaction-rule]')).toHaveCount(1)
  const compaction = events.getByRole('button', { name: /^Compaction/ })
  await compaction.hover()
  await expect(page.getByRole('tooltip')).toContainText('summarized the conversation')
  await expect(page.getByRole('tooltip')).toContainText('#31')
  await page.mouse.move(0, 0)
  const legend = page.getByLabel('Timeline legend')
  for (const word of [
    'Compaction',
    'Subagent',
    'Image',
    'Guard refusal',
    'Failed',
    'Model switch',
  ]) {
    await expect(legend).toContainText(word)
  }
  // the axis ticks round clock times
  await expect(page.locator('#run-timeline')).toContainText(/09:0\d/)
  await page
    .locator('#run-timeline')
    .screenshot({ path: info.outputPath('run-timeline-card-1440.png') })
  await compaction.click()
  await expect(
    page
      .getByRole('complementary', { name: 'Step details' })
      .locator('[data-event="compaction"]')
      .first(),
  ).toBeVisible()
  await page.mouse.move(0, 0)
  const widths = await timeline
    .getByRole('button')
    .evaluateAll((blocks) => blocks.map((block) => block.getBoundingClientRect().width))
  expect(Math.min(...widths)).toBeGreaterThanOrEqual(6)

  const block = timeline.getByRole('button', { name: /qwen.*, 1 call$/i }).first()
  await block.hover()
  await expect(page.getByRole('tooltip')).toContainText(' in · ')
  await block.click()
  await page.mouse.move(0, 0)
  const details = page.getByRole('complementary', { name: 'Step details' })
  for (const section of ['Routing', 'Usage', 'Outcome']) {
    await expect(details.getByRole('region', { name: section })).toBeVisible()
  }
  const routing = details.getByRole('region', { name: 'Routing' })
  await expect(routing.getByLabel('switched to')).toBeVisible()
  await expect(routing.getByText('Switched model')).toBeVisible()
  await expect(routing.getByText('A better model was allowed')).toBeVisible()
  await expect(routing.getByLabel('Signals').getByText('code request')).toBeVisible()
  await expect(details.getByRole('link', { name: 'Open the replay record' })).toBeVisible()
  await page.screenshot({ path: info.outputPath('run-timeline-1440.png'), fullPage: true })

  const fitMarkers = await events.getByRole('listitem').count()
  await page.getByRole('radio', { name: '4×' }).click()
  const overview = page.getByRole('slider', { name: 'Run overview' })
  await expect(overview).toBeVisible()
  // no marker is lost to the zoom: each keeps its size, and close ones part as the track widens
  await expect(events.locator('[data-event="compaction"]').first()).toBeAttached()
  expect(await events.getByRole('listitem').count()).toBeGreaterThanOrEqual(fitMarkers)
  expect(
    await timeline.evaluate((list) => {
      const track = list.closest('.overflow-x-auto')!
      return track.scrollWidth > track.clientWidth * 3
    }),
  ).toBe(true)
  const box = (await overview.boundingBox())!
  await overview.click({ position: { x: box.width * 0.9, y: box.height / 2 } })
  await expect
    .poll(async () => Number(await overview.getAttribute('aria-valuenow')))
    .toBeGreaterThan(50)
  await page.screenshot({ path: info.outputPath('run-timeline-zoom-1440.png'), fullPage: true })

  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('radio', { name: 'Fit' }).click()
  await expect(overview).toHaveCount(0)
  await expect.poll(() => fitsWidth(page)).toBe(true)
  // on a phone every kind of event in view still has a marker of its own, none overlapping
  for (const event of ['compaction', 'subagent_start', 'image', 'failure', 'switch']) {
    await expect(events.locator(`[data-event="${event}"]`).first()).toBeVisible()
  }
  const boxes = await events
    .getByRole('button')
    .evaluateAll((buttons) => buttons.map((button) => button.getBoundingClientRect()))
  for (let index = 1; index < boxes.length; index += 1) {
    expect(boxes[index].left).toBeGreaterThanOrEqual(boxes[index - 1].right - 1)
  }
  await page.mouse.move(0, 0)
  await page.screenshot({ path: info.outputPath('run-timeline-390.png'), fullPage: true })
  await page
    .locator('#run-timeline')
    .screenshot({ path: info.outputPath('run-timeline-card-390.png') })
})

test('shows the agents in use on Home with their logos and shares', async ({ page }, info) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await mockHomeOverview(page)
  await page.goto('/')

  const agents = page.getByTestId('home-agents')
  await expect(agents.getByRole('heading', { name: 'Agents in use' })).toBeVisible()
  await expect(agents).toContainText('Requests by harness · last 24 hours')
  await expect(agents.locator('li[data-harness]')).toHaveCount(6)
  for (const name of HARNESSES) {
    await expect(agents.getByRole('img', { name })).toBeVisible()
  }
  await expect(agents.locator('li[data-harness="api_client"]')).toContainText('35.5%')
  await expect(page.getByTestId('home-public-models')).toHaveCount(0)
  await page.screenshot({ path: info.outputPath('home-1440.png'), fullPage: true })

  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => fitsWidth(page)).toBe(true)
  await page.screenshot({ path: info.outputPath('home-390.png'), fullPage: true })
})
