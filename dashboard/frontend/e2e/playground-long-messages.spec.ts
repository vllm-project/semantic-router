import { expect, test } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'

test('previews a long user message without changing its request or copied content', async ({
  page,
}) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/v1/models*', async (route) => {
    await route.fulfill({
      json: {
        object: 'list',
        data: [
          {
            id: 'vllm-sr/auto',
            object: 'model',
            routing: { resolution: 'virtual', selectable: true, default_route: true },
          },
        ],
      },
    })
  })
  const prompt = 'Header marker. ' + ' orchard'.repeat(40000) + '\nUSER_TAIL_MARKER'
  const answer = 'Assistant answer. ' + ' detail'.repeat(300) + ' ASSISTANT_TAIL_MARKER'
  let sentContent: string | undefined
  await page.route('**/api/router/v1/chat/completions', async (route) => {
    const body = route.request().postDataJSON() as {
      messages: Array<{ role: string; content: string }>
    }
    sentContent = body.messages.find((message) => message.role === 'user')?.content
    await route.fulfill({
      json: {
        choices: [
          { index: 0, message: { role: 'assistant', content: answer }, finish_reason: 'stop' },
        ],
      },
    })
  })
  await page.goto('/playground')
  await page.evaluate(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: async (text: string) => {
          ;(window as typeof window & { copiedMessage?: string }).copiedMessage = text
        },
      },
    })
  })
  await page.getByPlaceholder('Ask me anything...').fill(prompt)
  await page.getByRole('button', { name: 'Send message' }).click()
  const user = page.locator('[data-message-role="user"]').last()
  const assistant = page.locator('[data-message-role="assistant"]').last()
  await expect(assistant).toContainText('ASSISTANT_TAIL_MARKER')
  expect(sentContent).toBe(prompt)
  await expect(user.getByRole('button', { name: 'Show more', exact: true })).toHaveAttribute(
    'aria-expanded',
    'false',
  )
  await expect(user).not.toContainText('USER_TAIL_MARKER')
  expect((await user.boundingBox())!.height).toBeLessThan(800)
  await expect(assistant.getByRole('button', { name: 'Show more' })).toHaveCount(0)
  await user.getByRole('button', { name: 'Copy', exact: true }).click()
  await expect
    .poll(() =>
      page.evaluate(() => (window as typeof window & { copiedMessage?: string }).copiedMessage),
    )
    .toBe(prompt)
  await user.getByRole('button', { name: 'Show more', exact: true }).click()
  await expect(user).toContainText('USER_TAIL_MARKER')
  await expect(user.getByRole('button', { name: 'Show less', exact: true })).toHaveAttribute(
    'aria-expanded',
    'true',
  )
  await user.getByRole('button', { name: 'Show less', exact: true }).click()
  await expect(user).not.toContainText('USER_TAIL_MARKER')
  await expect(user.getByRole('button', { name: 'Show more', exact: true })).toHaveAttribute(
    'aria-expanded',
    'false',
  )
  expect(sentContent).toBe(prompt)
})
