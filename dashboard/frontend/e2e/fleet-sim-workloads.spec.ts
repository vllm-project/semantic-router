import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

const sampleStats = {
  n_requests: 288,
  duration_s: 42.5,
  arrival_rate_rps: 6.8,
  p50_prompt_tokens: 512,
  p95_prompt_tokens: 2048,
  p99_prompt_tokens: 4096,
  p50_output_tokens: 64,
  p99_output_tokens: 192,
  p50_total_tokens: 640,
  p99_total_tokens: 4288,
  routing_distribution: { 'llama-8b': 0.72, 'llama-70b': 0.28 },
  prompt_histogram: [],
  output_histogram: [],
}

test('opens uploaded trace preview in a dialog from the trace table', async ({ page }) => {
  await mockAuthenticatedAppShell(page)

  await page.route('**/api/fleet-sim/api/workloads', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify([
        {
          name: 'azure',
          description: 'Azure chat traffic',
          path: '/app/data/azure_cdf.json',
        },
      ]),
    })
  })

  await page.route('**/api/fleet-sim/api/workloads/*/stats', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sampleStats),
    })
  })

  await page.route('**/api/fleet-sim/api/traces', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify([
        {
          id: 'trace-1',
          name: 'Batch spike requests',
          format: 'csv',
          upload_time: '2026-03-18T08:31:00Z',
          n_requests: 288,
          stats: sampleStats,
        },
      ]),
    })
  })

  await page.route('**/api/fleet-sim/api/traces/trace-1/sample?*', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        total: 288,
        records: [
          {
            timestamp: '0.53',
            prompt_tokens: '689',
            generated_tokens: '48',
            selected_model: 'llama-8b',
          },
          {
            timestamp: '0.67',
            prompt_tokens: '231',
            generated_tokens: '52',
            selected_model: 'llama-8b',
          },
        ],
      }),
    })
  })

  await page.goto('/fleet-sim/workloads')

  await expect(page.getByRole('heading', { name: 'Trace Intake' })).toBeVisible()
  const traceRow = page.locator('tr').filter({ hasText: 'Batch spike requests' })
  await traceRow.getByRole('button', { name: 'View' }).click()

  const dialog = page.getByRole('dialog')
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('heading', { name: 'Batch spike requests' })).toBeVisible()
  await expect(dialog.getByText(/showing 2 of 288 rows/i)).toBeVisible()
  await expect(dialog.getByText('selected_model')).toBeVisible()

  await page.getByRole('button', { name: 'Close trace preview' }).click()
  await expect(dialog).toBeHidden()
})
