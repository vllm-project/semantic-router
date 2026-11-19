import { expect, test, type Page } from '@playwright/test'

import {
  LONG_SESSION,
  mockHomeOverview,
  mockLongRun,
  mockOverview,
} from './support/insightsOverview'

const HARNESSES = ['Claude Code', 'Codex', 'OpenCode', 'Cursor', 'OpenClaw']

const fitsWidth = (page: Page) =>
  page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)

test('lays the breakdown out as one grid of figures and requests by agent harness', async ({
  page,
}, info) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await mockOverview(page)
  await page.goto('/observe/requests')
  await page.getByRole('button', { name: 'Breakdown' }).click()

  const breakdown = page.locator('#requests-breakdown')
  const figures = breakdown.getByRole('group', { name: 'Breakdown figures' })
  for (const label of [
    'Requests',
    'Input tokens',
    'Output tokens',
    'Prompt cache',
    'Model cost',
    'Vs reference',
  ]) {
    await expect(figures.getByText(label, { exact: true })).toBeVisible()
  }
  await expect(figures).toContainText('48.2M')
  await expect(figures).toContainText('-$18.42')
  await expect(figures).toContainText('57.7% below reference')
  const tops = await figures
    .locator(':scope > *')
    .evaluateAll((tiles) => tiles.map((tile) => Math.round(tile.getBoundingClientRect().top)))
  expect(new Set(tops).size).toBe(1)
  for (const text of [
    'How estimates work',
    'How the net saving works',
    'not a quality comparison',
  ]) {
    await expect(breakdown.getByText(text)).toHaveCount(0)
  }
  await breakdown.getByRole('button', { name: 'How the estimate is made' }).hover()
  await expect(page.getByRole('tooltip')).toContainText('Priced: 1,322 of 1,464 requests')
  await page.mouse.move(0, 0)

  const agents = breakdown.getByRole('list', { name: 'Requests by agent harness' })
  const rows = agents.locator('li[data-harness]')
  await expect(rows).toHaveCount(6)
  await expect(rows.last()).toHaveAttribute('data-harness', 'api_client')
  for (const name of HARNESSES) {
    await expect(agents.getByRole('img', { name })).toBeVisible()
  }
  await expect(agents.locator('li[data-harness="claude_code"]')).toContainText('28.1%')
  await expect(page.locator('main')).not.toContainText('N/A')
  await breakdown.screenshot({ path: info.outputPath('breakdown-1440.png') })

  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => fitsWidth(page)).toBe(true)
  await breakdown.screenshot({ path: info.outputPath('breakdown-390.png') })
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
  await expect(timeline).toContainText(/25m( \d+s)? idle/)
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

  await page.getByRole('radio', { name: '4×' }).click()
  const overview = page.getByRole('slider', { name: 'Run overview' })
  await expect(overview).toBeVisible()
  expect(
    await timeline.evaluate((list) => list.scrollWidth > list.parentElement!.clientWidth * 3),
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
  await page.screenshot({ path: info.outputPath('run-timeline-390.png'), fullPage: true })
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
