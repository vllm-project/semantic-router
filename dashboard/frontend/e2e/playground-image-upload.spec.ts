import { expect, test } from '@playwright/test'

import { mockAuthenticatedAppShell } from './support/auth'
import { openComposerAddMenu } from './support/playground'

const ONE_PIXEL_GIF = Buffer.from('R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw==', 'base64')

const streamBody = (content: string): string =>
  content
    .split('')
    .map(
      (character) =>
        `data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: character } }] })}\n\n`,
    )
    .join('') + 'data: [DONE]\n\n'

test('previews an uploaded image and sends it through the vision route', async ({ page }) => {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/v1/models*', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        object: 'list',
        data: [
          {
            id: 'vllm-sr/auto',
            object: 'model',
            owned_by: 'vllm-semantic-router',
            routing: { resolution: 'virtual', selectable: true, default_route: true },
          },
        ],
      }),
    })
  })

  let requestBody: Record<string, unknown> | null = null
  await page.route('**/api/router/v1/chat/completions', async (route) => {
    requestBody = route.request().postDataJSON() as Record<string, unknown>
    await route.fulfill({
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'x-vsr-selected-model': 'local/vision-backend',
        'x-vsr-selected-modality': 'image',
        'x-vsr-matched-context': 'image_content',
      },
      body: streamBody('The image is a one-pixel test fixture.'),
    })
  })

  await page.goto('/playground')

  const attachImage = async () => {
    const menu = await openComposerAddMenu(page)
    const chooserPromise = page.waitForEvent('filechooser')
    await menu.getByRole('menuitem', { name: 'Attach files' }).click()
    const chooser = await chooserPromise
    await chooser.setFiles({
      name: 'vision.gif',
      mimeType: 'image/gif',
      buffer: ONE_PIXEL_GIF,
    })
  }

  await attachImage()
  const pendingPreview = page.getByAltText('Preview of vision.gif')
  await expect(pendingPreview).toBeVisible()
  await expect
    .poll(() => pendingPreview.evaluate((image: HTMLImageElement) => image.naturalWidth))
    .toBe(1)

  await page.getByRole('button', { name: 'Remove image vision.gif' }).click()
  await expect(pendingPreview).toHaveCount(0)
  await attachImage()

  await page.getByPlaceholder('Ask me anything...').fill('What is visible in this image?')
  await page.getByRole('button', { name: 'Send message' }).click()

  await expect(page.getByText('The image is a one-pixel test fixture.')).toBeVisible()
  const sentPreview = page.getByAltText('vision.gif')
  await expect(sentPreview).toBeVisible()
  await expect
    .poll(() => sentPreview.evaluate((image: HTMLImageElement) => image.naturalWidth))
    .toBe(1)
  const assistantMessage = page.locator('[data-message-role="assistant"]').last()
  await expect(assistantMessage.getByText('local/vision-backend', { exact: true })).toBeVisible()
  await expect(assistantMessage.getByTitle('Context: Image Content')).toBeVisible()

  expect(requestBody).not.toBeNull()
  const messages = requestBody!.messages as Array<{ role: string; content: unknown }>
  expect(messages.at(-1)).toEqual({
    role: 'user',
    content: [
      { type: 'text', text: 'What is visible in this image?' },
      {
        type: 'image_url',
        image_url: { url: expect.stringMatching(/^data:image\/gif;base64,/) },
      },
    ],
  })
})
