import { test, expect } from '@playwright/test'
import { mockAuthenticatedAppShell } from './support/auth'
import { openComposerAddMenu } from './support/playground'

function chatStreamChunk(delta: Record<string, unknown>): string {
  return `data: ${JSON.stringify({ choices: [{ index: 0, delta }] })}\n\n`
}

function chatStreamBody(content: string, reasoning = ''): string {
  const initialLine = chatStreamChunk({ role: 'assistant', content: '' })
  const reasoningLines = reasoning
    ? reasoning.split('').map((char) => chatStreamChunk({ reasoning: char }))
    : []

  const contentLines = content.split('').map((char) => chatStreamChunk({ content: char }))

  return initialLine + [...reasoningLines, ...contentLines].join('') + 'data: [DONE]\n\n'
}

function chatToolCallBody(
  toolName: string,
  args: Record<string, unknown>,
  callId = 'call_open_web_1',
): string {
  return JSON.stringify({
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: callId,
              type: 'function',
              function: {
                name: toolName,
                arguments: JSON.stringify(args),
              },
            },
          ],
        },
        finish_reason: 'tool_calls',
      },
    ],
  })
}

function chatJsonBody(content: string): string {
  return JSON.stringify({
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content,
        },
        finish_reason: 'stop',
      },
    ],
  })
}

async function mockStreamingChatFetch(
  page: import('@playwright/test').Page,
  chunks: string[],
  delayMs = 250,
): Promise<void> {
  await page.evaluate(
    async ({ chunks: streamChunks, delayMs: streamDelayMs }) => {
      const originalFetch = window.fetch.bind(window)
      const encoder = new TextEncoder()

      window.fetch = async (input, init) => {
        const url =
          typeof input === 'string' ? input : input instanceof Request ? input.url : String(input)

        if (!url.includes('/api/router/v1/chat/completions')) {
          return originalFetch(input, init)
        }

        let chunkIndex = 0
        return new Response(
          new ReadableStream({
            start(controller) {
              const pushChunk = () => {
                if (chunkIndex >= streamChunks.length) {
                  controller.close()
                  return
                }

                controller.enqueue(encoder.encode(streamChunks[chunkIndex]))
                chunkIndex += 1
                window.setTimeout(pushChunk, streamDelayMs)
              }

              pushChunk()
            },
          }),
          {
            headers: {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
            },
          },
        )
      }
    },
    { chunks, delayMs },
  )
}

async function mockPlaygroundBootstrap(page: import('@playwright/test').Page): Promise<void> {
  await mockAuthenticatedAppShell(page)
  await page.route('**/api/router/v1/models*', async (route) => {
    await route.fulfill({
      status: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        object: 'list',
        data: [
          {
            id: 'vllm-sr/auto',
            object: 'model',
            owned_by: 'vllm-semantic-router',
            description: 'Intelligent Router for Mixture-of-Models',
            routing: { resolution: 'virtual', selectable: true, default_route: true },
          },
        ],
      }),
    })
  })
}

async function readStoredQueuePrompts(page: import('@playwright/test').Page): Promise<string[]> {
  return page.evaluate(() => {
    const raw = window.localStorage.getItem('sr:playground:queue')
    if (!raw) {
      return []
    }

    const parsed = JSON.parse(raw) as Record<string, Array<{ prompt?: string }>>
    return Object.values(parsed).flatMap((tasks) =>
      Array.isArray(tasks) ? tasks.map((task) => task.prompt || '').filter(Boolean) : [],
    )
  })
}

test.describe('Playground Chat Component', () => {
  test.beforeEach(async ({ page }) => {
    await mockPlaygroundBootstrap(page)
    await page.goto('/playground')
  })

  test('renders chat interface', async ({ page }) => {
    // Verify main elements are present
    await expect(page.getByPlaceholder('Ask me anything...')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'New conversation' })).toBeVisible()
    await expect(page.getByTestId('playground-routing-status')).toHaveCount(0)
    await expect(page.getByTestId('playground-sidebar-shell')).toBeVisible()
    await expect(page.getByTestId('playground-composer-model-select')).toContainText('vllm-sr/auto')
  })

  test('uses explicit routing metadata and keeps the model selector vendor-neutral', async ({
    page,
  }) => {
    await page.unroute('**/api/router/v1/models*')
    await page.route('**/api/router/v1/models*', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          object: 'list',
          data: [
            {
              id: 'vllm-sr/mom-v1-blend',
              object: 'model',
              owned_by: 'vllm-semantic-router',
              description: 'Intelligent Router for Mixture-of-Models',
              routing: { resolution: 'virtual', selectable: true, recipe: 'balanced' },
            },
            {
              id: 'vllm-sr/mom-v1-flash',
              object: 'model',
              owned_by: 'vllm-semantic-router',
              description: 'Latency-first routing profile',
              routing: {
                resolution: 'virtual',
                selectable: true,
                mode: 'future-orchestrator',
                recipe: 'speed-first',
              },
            },
            {
              id: 'vllm-sr/auto',
              object: 'model',
              owned_by: 'vllm-semantic-router',
              description: 'Automatic model routing',
              routing: { resolution: 'virtual', selectable: true, default_route: true },
            },
            {
              id: 'partner/backend',
              object: 'model',
              owned_by: 'upstream-endpoint',
              routing: { resolution: 'passthrough', selectable: false },
            },
          ],
        }),
      })
    })

    await page.reload({ waitUntil: 'domcontentloaded' })
    const selector = page.getByTestId('playground-composer-model-select')
    await expect(selector).toContainText('vllm-sr/auto')
    await expect(selector).not.toContainText('MoM')
    await selector.click()

    await expect(page.getByText('Choose a model', { exact: true })).toBeVisible()
    await expect(page.getByText('speed-first', { exact: true })).toBeVisible()
    await expect(page.getByText('Latency-first routing profile', { exact: true })).toHaveCount(0)
    await expect(page.getByRole('option')).toHaveCount(3)
    await page.getByRole('option', { name: /vllm-sr\/mom-v1-flash/ }).click()
    await expect(selector).toContainText('vllm-sr/mom-v1-flash')
  })

  test('sends ordinary chat through the advertised default alias when Fusion is available', async ({
    page,
  }) => {
    await page.unroute('**/api/router/v1/models*')
    await page.route('**/api/router/v1/models*', async (route) => {
      await route.fulfill({
        json: {
          object: 'list',
          data: [
            { id: 'vllm-sr/fusion', routing: { resolution: 'virtual', selectable: true } },
            {
              id: 'auto',
              routing: { resolution: 'virtual', selectable: true, default_route: true },
            },
            {
              id: 'vllm-sr/auto',
              routing: { resolution: 'virtual', selectable: true, default_route: true },
            },
          ],
        },
      })
    })
    const requests: Record<string, unknown>[] = []
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      const request = route.request().postDataJSON()
      requests.push(request)
      await route.fulfill({
        status: request.model === 'auto' ? 200 : 400,
        contentType: 'application/json',
        body: chatJsonBody('Hello!'),
      })
    })
    await page.reload({ waitUntil: 'domcontentloaded' })
    const selector = page.getByTestId('playground-composer-model-select')
    await expect(selector).toContainText('auto')
    await page.getByPlaceholder('Ask me anything...').fill('Hello!')
    await page.getByRole('button', { name: 'Send message', exact: true }).click()
    await expect(
      page.locator('[data-message-role="assistant"] [data-message-content]'),
    ).toContainText('Hello!')
    expect(requests).toHaveLength(1)
    expect(requests[0].model).toBe('auto')
    await selector.click()
    await expect(page.getByRole('option')).toHaveCount(3)
    await expect(page.getByRole('option', { name: /vllm-sr\/auto/ })).toBeVisible()
    await page.getByRole('option', { name: /vllm-sr\/fusion/ }).click()
    await expect(selector).toContainText('vllm-sr/fusion')
  })

  test('consolidates composer tools into an accessible mobile add menu', async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 700 })

    const trigger = page.getByRole('button', { name: 'Add to prompt' })
    await expect(trigger).toBeVisible()
    await expect(trigger).toHaveAttribute('aria-expanded', 'false')
    await expect(page.getByRole('menu', { name: 'Add to prompt' })).toHaveCount(0)

    await trigger.focus()
    await trigger.press('Enter')

    const menu = page.getByRole('menu', { name: 'Add to prompt' })
    const attachFiles = menu.getByRole('menuitem', { name: 'Attach files' })
    const webSearch = menu.getByRole('menuitemcheckbox', { name: 'Disable Web Search' })

    await expect(menu).toBeVisible()
    await expect(trigger).toHaveAttribute('aria-expanded', 'true')
    await expect(attachFiles).toBeFocused()
    await expect(webSearch).toHaveAttribute('aria-checked', 'true')

    await page.keyboard.press('End')
    await expect(webSearch).toBeFocused()
    await page.keyboard.press('Home')
    await expect(attachFiles).toBeFocused()

    const menuBox = await menu.boundingBox()
    expect(menuBox).not.toBeNull()
    expect(menuBox!.x).toBeGreaterThanOrEqual(0)
    expect(menuBox!.x + menuBox!.width).toBeLessThanOrEqual(320)
    expect(menuBox!.y).toBeGreaterThanOrEqual(0)
    expect(menuBox!.y + menuBox!.height).toBeLessThanOrEqual(700)

    await page.keyboard.press('Escape')
    await expect(menu).toHaveCount(0)
    await expect(trigger).toBeFocused()

    const togglesMenu = await openComposerAddMenu(page)
    await togglesMenu.getByRole('menuitemcheckbox', { name: 'Disable Web Search' }).click()
    await expect(
      togglesMenu.getByRole('menuitemcheckbox', { name: 'Enable Web Search' }),
    ).toHaveAttribute('aria-checked', 'false')
    await page.mouse.click(380, 100)
    await expect(page.getByRole('menu', { name: 'Add to prompt' })).toHaveCount(0)

    const reopenedMenu = await openComposerAddMenu(page)
    const fileChooserPromise = page.waitForEvent('filechooser')
    await reopenedMenu.getByRole('menuitem', { name: 'Attach files' }).click()
    const fileChooser = await fileChooserPromise
    await fileChooser.setFiles({
      name: 'routing-notes.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('Prefer the most suitable heterogeneous model path.'),
    })

    await expect(page.getByTestId('playground-attachment-list')).toContainText('routing-notes.txt')
    await expect(page.getByRole('menu', { name: 'Add to prompt' })).toHaveCount(0)
    await expect(trigger).toBeFocused()

    const layoutWidth = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
    }))
    expect(layoutWidth.scrollWidth).toBeLessThanOrEqual(layoutWidth.innerWidth)
  })

  test('keeps the mobile composer readable with both side panes closed', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })

    const input = page.getByPlaceholder('Ask me anything...')
    await expect.poll(async () => (await input.boundingBox())?.width ?? 0).toBeGreaterThan(250)
    await input.click()
    await input.fill('The composer stays interactive on a phone')
    await expect(page.getByTestId('playground-sidebar-shell')).toHaveCount(0)
    await expect(page.getByTestId('playground-inspector')).toHaveCount(0)

    const layoutWidth = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
    }))
    expect(layoutWidth.scrollWidth).toBeLessThanOrEqual(layoutWidth.innerWidth)

    const composerBox = await page.getByTestId('chat-composer').boundingBox()
    expect(composerBox).not.toBeNull()
    expect((composerBox?.x ?? 0) + (composerBox?.width ?? 0)).toBeLessThanOrEqual(390)

    // A pane opened on a phone floats over the chat and closes with Escape.
    await page.getByRole('button', { name: 'Show conversations' }).click()
    await expect(page.getByTestId('playground-sidebar-shell')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('playground-sidebar-shell')).toHaveCount(0)
  })

  test('uses the live alias and blocks a restored task with a retired model', async ({ page }) => {
    await page.unroute('**/api/router/v1/models*')
    await page.route('**/api/router/v1/models*', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          object: 'list',
          data: [
            {
              id: 'router/production',
              object: 'model',
              owned_by: 'vllm-semantic-router',
              description: 'Intelligent Router for Mixture-of-Models',
              routing: { resolution: 'virtual', selectable: true, default_route: true },
            },
          ],
        }),
      })
    })

    const requestModels: string[] = []
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      requestModels.push((route.request().postDataJSON() as { model?: string }).model ?? '')
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
        body: chatStreamBody('Custom route is live.'),
      })
    })

    await page.reload({ waitUntil: 'domcontentloaded' })
    await expect(page.getByTestId('playground-routing-status')).toHaveCount(0)
    await page.getByPlaceholder('Ask me anything...').fill('Use the effective runtime alias')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect(page.getByText('Custom route is live.')).toBeVisible()
    expect(requestModels).toEqual(['router/production'])

    await page.evaluate(() => {
      const conversationId = 'restored-legacy-conversation'
      window.localStorage.setItem(
        'sr:playground:queue',
        JSON.stringify({
          [conversationId]: [
            {
              id: 'restored-legacy-task',
              conversationId,
              prompt: 'Restore this legacy task',
              createdAt: Date.now(),
              requestOptions: {
                enableWebSearch: false,
                model: 'MoM',
              },
            },
          ],
        }),
      )
    })

    await page.reload({ waitUntil: 'domcontentloaded' })
    await expect(page.getByText(/Queued model "MoM" is no longer available/)).toBeVisible()
    expect(requestModels).toEqual(['router/production'])
  })

  test('blocks sending and offers retry when model discovery fails', async ({ page }) => {
    await page.unroute('**/api/router/v1/models*')
    await page.route('**/api/router/v1/models*', async (route) => {
      await route.fulfill({ status: 503, body: 'router unavailable' })
    })

    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.getByPlaceholder('Ask me anything...').fill('Do not send this request')
    await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Retry discovery' })).toBeVisible()
    const errorRegion = page.getByTestId('playground-error-region')
    const chatArea = errorRegion.locator('xpath=..')
    const [errorBox, chatBox, composerBox] = await Promise.all([
      errorRegion.boundingBox(),
      chatArea.boundingBox(),
      page.getByPlaceholder('Ask me anything...').boundingBox(),
    ])
    expect(errorBox).not.toBeNull()
    expect(chatBox).not.toBeNull()
    expect(composerBox).not.toBeNull()
    if (errorBox && chatBox && composerBox) {
      expect(errorBox.y - chatBox.y).toBeLessThanOrEqual(24)
      expect(errorBox.y + errorBox.height).toBeLessThan(composerBox.y)
    }
  })

  test('rejects legacy discovery entries without routing metadata and never infers an alias', async ({
    page,
  }) => {
    await page.unroute('**/api/router/v1/models*')
    await page.route('**/api/router/v1/models*', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          object: 'list',
          data: [
            {
              id: 'MoM',
              object: 'model',
              owned_by: 'vllm-semantic-router',
              description: 'Intelligent Router for Mixture-of-Models',
            },
          ],
        }),
      })
    })

    let completionRequests = 0
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      completionRequests += 1
      await route.abort()
    })

    await page.reload({ waitUntil: 'domcontentloaded' })
    const composer = page.getByPlaceholder('Ask me anything...')
    await composer.fill('Do not fall back to MoM')
    await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Retry discovery' })).toBeVisible()
    await composer.press('Enter')
    expect(completionRequests).toBe(0)
  })

  test('opens and collapses the left history rail', async ({ page }) => {
    await page.evaluate(() => {
      const now = Date.now()
      window.localStorage.setItem(
        'sr:chat:conversations',
        JSON.stringify([
          {
            id: 'saved-conversation',
            createdAt: now - 300000,
            updatedAt: now,
            payload: [
              {
                id: 'saved-user-message',
                role: 'user',
                content: 'Saved conversation preview',
                timestamp: new Date(now - 120000).toISOString(),
              },
            ],
          },
        ]),
      )
    })

    await page.goto('/playground', { waitUntil: 'domcontentloaded' })

    const shell = page.getByTestId('playground-sidebar-shell')
    await expect(shell).toBeVisible()
    const sidebarItem = shell.getByRole('button', {
      name: 'Saved conversation preview',
      exact: true,
    })
    await expect(sidebarItem).toBeVisible()
    await expect(shell.locator('img')).toHaveCount(0)

    await page.getByRole('button', { name: 'Hide conversations' }).click()
    await expect(sidebarItem).not.toBeVisible()
    // With the pane closed the header offers the new-conversation action.
    await expect(page.getByRole('button', { name: 'New conversation' })).toHaveCount(1)

    await page.getByRole('button', { name: 'Show conversations' }).click()
    await expect(sidebarItem).toBeVisible()

    // The choice holds across reloads.
    await page.getByRole('button', { name: 'Hide conversations' }).click()
    await page.reload({ waitUntil: 'domcontentloaded' })
    await expect(page.getByRole('button', { name: 'Show conversations' })).toBeVisible()
    await expect(page.getByTestId('playground-sidebar-shell')).toHaveCount(0)
  })

  test('renames, searches and confirms deleting a saved conversation', async ({ page }) => {
    await page.evaluate(() => {
      const now = Date.now()
      const conversation = (id: string, content: string, ago: number) => ({
        id,
        createdAt: now - ago - 1000,
        updatedAt: now - ago,
        payload: [{ id: `${id}-u`, role: 'user', content, timestamp: new Date(now).toISOString() }],
      })
      window.localStorage.setItem(
        'sr:chat:conversations',
        JSON.stringify([
          conversation('first', 'Summarize the routing logs', 60_000),
          conversation('second', 'Plan a trip to Kyoto', 3 * 86_400_000),
        ]),
      )
    })
    await page.goto('/playground', { waitUntil: 'domcontentloaded' })

    const shell = page.getByTestId('playground-sidebar-shell')
    await expect(shell.getByText('Today')).toBeVisible()
    await expect(shell.getByText('Previous 7 days')).toBeVisible()

    await shell.getByRole('searchbox', { name: 'Search conversations' }).fill('kyoto')
    await expect(
      shell.getByRole('button', { name: 'Plan a trip to Kyoto', exact: true }),
    ).toBeVisible()
    await expect(
      shell.getByRole('button', { name: 'Summarize the routing logs', exact: true }),
    ).toHaveCount(0)
    await shell.getByRole('searchbox', { name: 'Search conversations' }).fill('')

    await shell
      .getByRole('button', { name: 'Conversation options for Plan a trip to Kyoto' })
      .click()
    await page.getByRole('menuitem', { name: 'Rename' }).click()
    const name = shell.getByRole('textbox', { name: 'Conversation name' })
    await name.fill('Kyoto itinerary')
    await name.press('Enter')
    await expect(shell.getByRole('button', { name: 'Kyoto itinerary', exact: true })).toBeVisible()

    await shell.getByRole('button', { name: 'Conversation options for Kyoto itinerary' }).click()
    await page.getByRole('menuitem', { name: 'Delete' }).click()
    const confirm = page.getByRole('alertdialog')
    await expect(confirm).toContainText('Kyoto itinerary')
    await confirm.getByRole('button', { name: 'Delete' }).click()
    await expect(shell.getByRole('button', { name: 'Kyoto itinerary', exact: true })).toHaveCount(0)
    await expect(
      shell.getByRole('button', { name: 'Summarize the routing logs', exact: true }),
    ).toBeVisible()
  })

  test('can type message', async ({ page }) => {
    const input = page.getByPlaceholder('Ask me anything...')
    await input.fill('Hello, this is a test message')
    await expect(input).toHaveValue('Hello, this is a test message')
  })

  test('grows the composer with multiline input and caps it before the page layout shifts', async ({
    page,
  }) => {
    const input = page.getByPlaceholder('Ask me anything...')
    const initialBox = await input.boundingBox()
    expect(initialBox).not.toBeNull()

    await input.fill(Array.from({ length: 7 }, (_, index) => `Line ${index + 1}`).join('\n'))
    const expandedBox = await input.boundingBox()
    expect(expandedBox).not.toBeNull()
    expect(expandedBox!.height).toBeGreaterThan(initialBox!.height + 50)
    await expect(input).toHaveCSS('overflow-y', 'hidden')

    await input.fill(Array.from({ length: 40 }, (_, index) => `Long line ${index + 1}`).join('\n'))
    const cappedBox = await input.boundingBox()
    expect(cappedBox).not.toBeNull()
    expect(cappedBox!.height).toBeLessThanOrEqual(222)
    await expect(input).toHaveCSS('overflow-y', 'auto')
  })

  test('shows a copy button for user messages and copies their content', async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'application/json',
        },
        body: chatJsonBody('Assistant reply'),
      })
    })

    await page.evaluate(() => {
      let copiedText = ''
      Object.defineProperty(window, '__copiedUserMessage', {
        configurable: true,
        get: () => copiedText,
        set: (value: string) => {
          copiedText = value
        },
      })

      Object.defineProperty(navigator, 'clipboard', {
        configurable: true,
        value: {
          writeText: async (text: string) => {
            ;(window as typeof window & { __copiedUserMessage?: string }).__copiedUserMessage = text
          },
        },
      })
    })

    await page.getByPlaceholder('Ask me anything...').fill('Copy my user message')
    await page.getByRole('button', { name: 'Send message' }).click()

    const userMessage = page.locator('[data-message-role="user"]').last()
    await expect(userMessage).toContainText('Copy my user message')

    await userMessage.getByRole('button', { name: 'Copy' }).click()

    await expect
      .poll(() =>
        page.evaluate(
          () =>
            (window as typeof window & { __copiedUserMessage?: string }).__copiedUserMessage ?? '',
        ),
      )
      .toBe('Copy my user message')
  })

  test('preserves markdown list formatting when citations are present', async ({ page }) => {
    await page.evaluate(() => {
      const now = Date.now()
      window.localStorage.setItem('sr:playground:active-conversation', 'citation-conversation')
      window.localStorage.setItem(
        'sr:chat:conversations',
        JSON.stringify([
          {
            id: 'citation-conversation',
            createdAt: now - 60_000,
            updatedAt: now,
            payload: [
              {
                id: 'citation-user',
                role: 'user',
                content: 'Show me the summary',
                timestamp: new Date(now - 30_000).toISOString(),
              },
              {
                id: 'citation-assistant',
                role: 'assistant',
                content: '## Summary\n- First cited point [1]\n- Second cited point [2]',
                timestamp: new Date(now - 20_000).toISOString(),
                toolResults: [
                  {
                    callId: 'search-call-1',
                    name: 'search_web',
                    content: [
                      {
                        title: 'Source One',
                        url: 'https://example.com/source-one',
                        snippet: 'One',
                      },
                      {
                        title: 'Source Two',
                        url: 'https://example.com/source-two',
                        snippet: 'Two',
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ]),
      )
    })

    await page.goto('/playground', { waitUntil: 'domcontentloaded' })

    const assistantMessage = page.locator('[data-message-role="assistant"]').last()
    await expect(assistantMessage.getByRole('heading', { name: 'Summary' })).toBeVisible()
    await expect(assistantMessage.locator('li')).toHaveCount(2)
    await expect(assistantMessage.getByRole('link', { name: '[1]' })).toHaveAttribute(
      'href',
      'https://example.com/source-one',
    )
    await expect(assistantMessage.getByRole('link', { name: '[2]' })).toHaveAttribute(
      'href',
      'https://example.com/source-two',
    )
  })

  test('renders markdown while the response is still streaming', async ({ page }) => {
    await mockStreamingChatFetch(
      page,
      [
        chatStreamChunk({ role: 'assistant', content: '' }),
        chatStreamChunk({ content: '# Live heading\n' }),
        chatStreamChunk({ content: '\n\nStreaming body text keeps arriving.' }),
        chatStreamChunk({ content: '\n\nMore markdown-friendly content follows.' }),
        'data: [DONE]\n\n',
      ],
      500,
    )

    await page.getByPlaceholder('Ask me anything...').fill('Stream markdown formatting')
    await page.getByRole('button', { name: 'Send message' }).click()

    const assistantMessage = page.locator('[data-message-role="assistant"]').last()

    await expect(assistantMessage.getByRole('heading', { name: 'Live heading' })).toBeVisible({
      timeout: 5000,
    })
    await expect(page.getByRole('button', { name: 'Stop generating' })).toBeVisible()
  })

  test('does not send on Enter while IME composition is active', async ({ page }) => {
    let requestCount = 0
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      requestCount += 1
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'application/json',
        },
        body: chatJsonBody('IME-safe response'),
      })
    })

    const input = page.getByPlaceholder('Ask me anything...')
    await input.fill('你好')
    await input.dispatchEvent('compositionstart')
    await input.press('Enter')

    await page.waitForTimeout(200)
    expect(requestCount).toBe(0)
    await expect(page.locator('[data-message-role="user"]')).toHaveCount(0)

    await input.dispatchEvent('compositionend')
    await input.press('Enter')

    await expect.poll(() => requestCount).toBe(1)
    await expect(page.locator('[data-message-role="user"]').last()).toContainText('你好')
  })

  test('send button disabled when input empty', async ({ page }) => {
    const sendButton = page.getByRole('button', { name: 'Send message' })
    // Button should be disabled when input is empty
    await expect(sendButton).toBeDisabled()

    // Type something
    await page.getByPlaceholder('Ask me anything...').fill('test')

    // Button should be enabled
    await expect(sendButton).toBeEnabled()
  })

  test('new conversation clears messages', async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody('Hello! This is a mock response.'),
      })
    })

    await page.getByPlaceholder('Ask me anything...').fill('Clear me')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect(page.locator('[data-message-role="user"]').last()).toContainText('Clear me', {
      timeout: 10000,
    })

    await page.getByRole('button', { name: 'New conversation' }).click()

    await expect(
      page.locator('[data-message-role="user"]').filter({ hasText: 'Clear me' }),
    ).toHaveCount(0)
    await expect(page.getByRole('heading', { name: /Welcome, Admin/i })).toBeVisible()
  })

  test('keeps streaming in the original session after switching away and shows progress when switching back', async ({
    page,
  }) => {
    await mockStreamingChatFetch(
      page,
      [
        chatStreamChunk({ role: 'assistant', content: '' }),
        chatStreamChunk({ content: 'First visible chunk. ' }),
        chatStreamChunk({ content: 'Continues while hidden. ' }),
        chatStreamChunk({ content: 'Final background chunk.' }),
        'data: [DONE]\n\n',
      ],
      250,
    )

    const input = page.getByPlaceholder('Ask me anything...')
    const sessionAPrompt = 'Keep session A streaming'

    await input.fill(sessionAPrompt)
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(page.getByText('First visible chunk.')).toBeVisible({ timeout: 5000 })
    await expect(page.getByRole('button', { name: 'Stop generating' })).toBeVisible()

    await page.getByRole('button', { name: 'New conversation' }).click()
    await expect(page.getByRole('heading', { name: /Welcome, Admin/i })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Stop generating' })).toHaveCount(0)
    await expect(page.getByText('First visible chunk.')).toHaveCount(0)

    await page.waitForTimeout(900)

    const sidebarShell = page.getByTestId('playground-sidebar-shell')
    const sessionAButton = sidebarShell.getByRole('button', { name: sessionAPrompt, exact: true })
    if (
      (await sessionAButton.count()) === 0 ||
      !(await sessionAButton
        .first()
        .isVisible()
        .catch(() => false))
    ) {
      await page.getByRole('button', { name: 'Show conversations' }).click()
    }

    await expect(sessionAButton).toBeVisible()
    await sessionAButton.click()

    await expect(page.getByText('Continues while hidden.')).toBeVisible({ timeout: 5000 })
    await expect(page.getByText('Final background chunk.')).toBeVisible({ timeout: 5000 })
  })

  test('switching back to a saved session snaps the transcript to the latest messages', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 560 })

    await page.evaluate(() => {
      const now = Date.now()
      const longHistory = Array.from({ length: 12 }, (_, index) => {
        const offset = (12 - index) * 90_000
        return [
          {
            id: `long-user-${index + 1}`,
            role: 'user',
            content: index === 0 ? 'Long history session' : `Long history follow-up ${index + 1}`,
            timestamp: new Date(now - 4_000_000 + offset).toISOString(),
          },
          {
            id: `long-assistant-${index + 1}`,
            role: 'assistant',
            content: Array.from(
              { length: 8 },
              (_, paragraphIndex) =>
                `Long history answer ${index + 1}, paragraph ${paragraphIndex + 1}: the transcript should reopen near the latest content.`,
            ).join('\n\n'),
            timestamp: new Date(now - 4_000_000 + offset + 15_000).toISOString(),
          },
        ]
      }).flat()

      const recentConversation = [
        {
          id: 'recent-user',
          role: 'user',
          content: 'Recent short session',
          timestamp: new Date(now - 30_000).toISOString(),
        },
        {
          id: 'recent-assistant',
          role: 'assistant',
          content: 'Short reply for the currently selected session.',
          timestamp: new Date(now - 15_000).toISOString(),
        },
      ]

      window.localStorage.setItem('sr:playground:active-conversation', 'recent-conversation')
      window.localStorage.setItem(
        'sr:chat:conversations',
        JSON.stringify([
          {
            id: 'recent-conversation',
            createdAt: now - 60_000,
            updatedAt: now - 10_000,
            payload: recentConversation,
          },
          {
            id: 'long-history-conversation',
            createdAt: now - 5_000_000,
            updatedAt: now - 20_000,
            payload: longHistory,
          },
        ]),
      )
    })

    await page.goto('/playground', { waitUntil: 'domcontentloaded' })
    await expect(page.getByText('Short reply for the currently selected session.')).toBeVisible()

    const sidebarShell = page.getByTestId('playground-sidebar-shell')
    const longHistoryButton = sidebarShell.getByRole('button', {
      name: 'Long history session',
      exact: true,
    })
    if (
      (await longHistoryButton.count()) === 0 ||
      !(await longHistoryButton
        .first()
        .isVisible()
        .catch(() => false))
    ) {
      await page.getByRole('button', { name: 'Show conversations' }).click()
    }

    await longHistoryButton.click()
    await expect(
      page.getByText(
        'Long history answer 12, paragraph 8: the transcript should reopen near the latest content.',
      ),
    ).toBeVisible()

    const transcript = page.getByTestId('chat-transcript')
    await expect
      .poll(
        async () => {
          return transcript.evaluate((node) => {
            const container = node as HTMLDivElement
            return container.scrollHeight - container.scrollTop - container.clientHeight
          })
        },
        { timeout: 5000 },
      )
      .toBeLessThan(24)
  })

  test('sends message and receives response (mocked API)', async ({ page }) => {
    // Mock the chat API endpoint
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      const request = route.request()
      const postData = request.postDataJSON()

      // Verify request structure
      expect(postData).toHaveProperty('messages')
      expect(postData).toHaveProperty('model', 'vllm-sr/auto')
      expect(postData).toHaveProperty('stream')

      // Return mock streaming response
      const responseText = 'Hello! This is a mock response.'

      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody(responseText),
      })
    })

    // Type a message
    const input = page.getByPlaceholder('Ask me anything...')
    await input.fill('Hello, how are you?')

    // Send the message
    await page.getByRole('button', { name: 'Send message' }).click()

    // User message should appear
    await expect(page.getByTestId('chat-transcript').getByText('Hello, how are you?')).toBeVisible()

    // Wait for response to appear (the mocked response)
    await expect(page.getByText('Hello! This is a mock response.')).toBeVisible({ timeout: 10000 })

    // Input should be cleared after sending
    await expect(input).toHaveValue('')
  })

  test('routes open_web through the backend proxy', async ({ page }) => {
    let chatRequestCount = 0
    let openWebRequestBody: Record<string, unknown> | null = null

    await page.route('**/api/tools/open-web', async (route) => {
      openWebRequestBody = route.request().postDataJSON() as Record<string, unknown>

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          url: 'https://example.com/article',
          title: 'Example Article',
          content: 'Example content returned by the backend proxy.',
          length: 46,
          truncated: false,
          method: 'direct',
        }),
      })
    })

    await page.route('**/api/router/v1/chat/completions', async (route) => {
      chatRequestCount += 1

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body:
          chatRequestCount === 1
            ? chatToolCallBody('open_web', {
                url: 'https://example.com/article',
                format: 'markdown',
                max_length: 15000,
              })
            : chatJsonBody('The page was fetched through the backend proxy.'),
      })
    })

    await page.getByPlaceholder('Ask me anything...').fill('Open an article for me')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(page.getByText('The page was fetched through the backend proxy.')).toBeVisible({
      timeout: 10000,
    })
    await expect(page.getByText('Example Article')).toBeVisible({ timeout: 10000 })

    expect(openWebRequestBody).toMatchObject({
      url: 'https://example.com/article',
      timeout: 30,
      force_jina: false,
      format: 'markdown',
      max_length: 15000,
      with_images: false,
    })
    expect(chatRequestCount).toBe(2)
  })

  test('sends tool failures back to the model during follow-up loops', async ({ page }) => {
    await page.evaluate(async () => {
      const originalFetch = window.fetch.bind(window)
      const encoder = new TextEncoder()
      let completionRequestCount = 0

      const streamResponse = (chunks: string[]) =>
        new Response(
          new ReadableStream({
            start(controller) {
              for (const chunk of chunks) {
                controller.enqueue(encoder.encode(chunk))
              }
              controller.close()
            },
          }),
          {
            headers: {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
            },
          },
        )

      window.fetch = async (input, init) => {
        const url =
          typeof input === 'string' ? input : input instanceof Request ? input.url : String(input)

        if (url.includes('/api/router/v1/chat/completions')) {
          completionRequestCount += 1

          if (completionRequestCount === 1) {
            return streamResponse([
              'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_open_web_1","type":"function","function":{"name":"open_web","arguments":"{\\"url\\":\\"https://example.com/article\\"}"}}]}}]}\n\n',
              'data: {"choices":[{"index":0,"finish_reason":"tool_calls"}]}\n\n',
              'data: [DONE]\n\n',
            ])
          }

          if (completionRequestCount === 2) {
            const requestBody = typeof init?.body === 'string' ? JSON.parse(init.body) : null
            ;(window as typeof window & { __lastFollowUpRequest?: unknown }).__lastFollowUpRequest =
              requestBody

            return streamResponse([
              'data: {"choices":[{"index":0,"delta":{"content":"Follow',
              ' up stream recovered."}}]}\n\n',
              'data: [DONE]\n\n',
            ])
          }
        }

        if (url.includes('/api/tools/open-web') || url.startsWith('https://r.jina.ai/')) {
          return new Response('upstream failure', {
            status: 500,
            statusText: 'Internal Server Error',
          })
        }

        return originalFetch(input, init)
      }
    })

    await page.getByPlaceholder('Ask me anything...').fill('Trigger a tool failure')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect
      .poll(
        async () => {
          return await page.evaluate(() =>
            (window as typeof window & { __lastFollowUpRequest?: unknown }).__lastFollowUpRequest
              ? 1
              : 0,
          )
        },
        { timeout: 10000 },
      )
      .toBe(1)

    await expect
      .poll(
        async () => {
          const request = await page.evaluate(
            () =>
              (
                window as typeof window & {
                  __lastFollowUpRequest?: { messages?: Array<Record<string, unknown>> }
                }
              ).__lastFollowUpRequest,
          )
          return request?.messages?.find((message) => message.role === 'tool')?.content ?? null
        },
        { timeout: 10000 },
      )
      .not.toBeNull()

    const refreshedFollowUpRequest = await page.evaluate(
      () =>
        (
          window as typeof window & {
            __lastFollowUpRequest?: { messages?: Array<Record<string, unknown>>; model?: string }
          }
        ).__lastFollowUpRequest,
    )

    expect(refreshedFollowUpRequest?.model).toBe('vllm-sr/auto')
    expect(
      refreshedFollowUpRequest?.messages?.some((message) => message.role === 'tool'),
    ).toBeTruthy()
    const resolvedToolMessage = refreshedFollowUpRequest?.messages?.find(
      (message) => message.role === 'tool',
    )
    expect(resolvedToolMessage?.content).toContain('Tool execution failed:')
    expect(resolvedToolMessage?.content).not.toBe('null')
  })

  test('executes the built-in calculate tool and forwards the structured result', async ({
    page,
  }) => {
    await page.evaluate(async () => {
      const originalFetch = window.fetch.bind(window)
      const encoder = new TextEncoder()
      let completionRequestCount = 0

      const streamResponse = (chunks: string[]) =>
        new Response(
          new ReadableStream({
            start(controller) {
              for (const chunk of chunks) {
                controller.enqueue(encoder.encode(chunk))
              }
              controller.close()
            },
          }),
          {
            headers: {
              'Content-Type': 'text/event-stream',
              'Cache-Control': 'no-cache',
            },
          },
        )

      window.fetch = async (input, init) => {
        const url =
          typeof input === 'string' ? input : input instanceof Request ? input.url : String(input)

        if (url.includes('/api/router/v1/chat/completions')) {
          completionRequestCount += 1

          if (completionRequestCount === 1) {
            return streamResponse([
              'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_calculate_1","type":"function","function":{"name":"calculate","arguments":"{\\"expression\\":\\"2 + 2 * 3\\"}"}}]}}]}\n\n',
              'data: {"choices":[{"index":0,"finish_reason":"tool_calls"}]}\n\n',
              'data: [DONE]\n\n',
            ])
          }

          if (completionRequestCount === 2) {
            const requestBody = typeof init?.body === 'string' ? JSON.parse(init.body) : null
            ;(
              window as typeof window & { __lastCalculateFollowUp?: unknown }
            ).__lastCalculateFollowUp = requestBody

            return streamResponse([
              'data: {"choices":[{"index":0,"delta":{"content":"The answer is 8."}}]}\n\n',
              'data: [DONE]\n\n',
            ])
          }
        }

        return originalFetch(input, init)
      }
    })

    await page.getByPlaceholder('Ask me anything...').fill('What is 2 + 2 * 3?')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(page.getByText('The answer is 8.')).toBeVisible({ timeout: 10000 })

    const followUpRequest = await page.evaluate(
      () =>
        (
          window as typeof window & {
            __lastCalculateFollowUp?: { messages?: Array<Record<string, unknown>> }
          }
        ).__lastCalculateFollowUp,
    )

    const toolMessage = followUpRequest?.messages?.find((message) => message.role === 'tool')

    expect(toolMessage).toBeTruthy()
    expect(typeof toolMessage?.content).toBe('string')
    expect(toolMessage?.content).toContain('"formatted_result":"8"')
  })

  test('handles API error gracefully', async ({ page }) => {
    // Mock API to return an error
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal server error' }),
      })
    })

    // Type and send a message
    await page.getByPlaceholder('Ask me anything...').fill('Test error handling')
    await page.getByRole('button', { name: 'Send message' }).click()

    // User message should still appear
    await expect(page.getByTestId('chat-transcript').getByText('Test error handling')).toBeVisible()

    const errorRegion = page.getByTestId('playground-error-region')
    await expect(errorRegion).toBeVisible({ timeout: 5000 })
    await expect(errorRegion).toContainText('The model service is temporarily unavailable')
    await expect(page.getByText('API error:')).toHaveCount(0)
    const technicalDetails = errorRegion.locator('[data-playground-technical-details="true"]')
    await expect(technicalDetails).toBeVisible()
    await expect(technicalDetails).not.toHaveAttribute('open')
  })

  test('stop button appears during streaming', async ({ page }) => {
    // Mock a slow streaming response
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      // Delay response to allow stop button to appear
      await new Promise((resolve) => setTimeout(resolve, 2000))
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
        body: 'data: {"choices":[{"delta":{"content":"Test"}}]}\n\ndata: [DONE]\n\n',
      })
    })

    // Send a message
    await page.getByPlaceholder('Ask me anything...').fill('Test streaming')
    await page.getByRole('button', { name: 'Send message' }).click()

    // Stop button should appear (look for it quickly before response completes)
    await expect(page.getByRole('button', { name: 'Stop generating' })).toBeVisible({
      timeout: 5000,
    })
  })

  test('queues prompts during streaming, restores them after reload, and lets queued tasks be removed', async ({
    page,
  }) => {
    let requestCount = 0
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      requestCount += 1
      await new Promise((resolve) => setTimeout(resolve, 6000))
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody(`Response ${requestCount}`),
      })
    })

    const input = page.getByPlaceholder('Ask me anything...')
    await input.fill('First queued task')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect.poll(() => requestCount).toBe(1)
    const queue = page.getByTestId('playground-task-queue')

    await input.fill('Second queued task')
    await page.getByRole('button', { name: 'Send message' }).click()
    await input.fill('Third queued task')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(queue).toContainText('Second queued task')
    await expect(queue).toContainText('Third queued task')
    await expect(queue).not.toContainText('First queued task')
    await expect
      .poll(() => readStoredQueuePrompts(page))
      .toEqual(['Second queued task', 'Third queued task'])

    await expect
      .poll(() =>
        page.evaluate(() => {
          const queuedConversations = Object.keys(
            JSON.parse(window.localStorage.getItem('sr:playground:queue') || '{}'),
          )
          return queuedConversations.includes(
            window.localStorage.getItem('sr:playground:active-conversation') || '',
          )
        }),
      )
      .toBe(true)

    await page.reload({ waitUntil: 'domcontentloaded' })

    const restoredQueue = page.getByTestId('playground-task-queue')
    await expect(restoredQueue).toContainText('Third queued task', { timeout: 10000 })
    await expect(restoredQueue).not.toContainText('Second queued task')
    await expect(restoredQueue).toContainText('Third queued task')
    await expect.poll(() => readStoredQueuePrompts(page)).toEqual(['Third queued task'])

    const thirdQueuedTask = restoredQueue
      .locator('[data-testid^="playground-task-queue-item-"]')
      .filter({ hasText: 'Third queued task' })
      .first()
    await thirdQueuedTask.getByRole('button', { name: /Remove queued task:/ }).click()

    await expect(restoredQueue).toHaveCount(0)
    await expect.poll(() => readStoredQueuePrompts(page)).toEqual([])
  })

  test('isolates queued work per conversation so a new conversation can run immediately', async ({
    page,
  }) => {
    let requestCount = 0
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      requestCount += 1
      const body = route.request().postDataJSON() as {
        messages?: Array<{ role?: string; content?: string }>
      }
      const prompt =
        [...(body.messages ?? [])].reverse().find((message) => message.role === 'user')?.content ??
        `Request ${requestCount}`

      const delayMs = prompt === 'A first task' ? 6000 : 600
      await new Promise((resolve) => setTimeout(resolve, delayMs))
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody(`Response for ${prompt}`),
      })
    })

    const input = page.getByPlaceholder('Ask me anything...')

    await input.fill('A first task')
    await input.press('Enter')
    await expect.poll(() => requestCount).toBe(1)

    await input.fill('A queued task')
    await input.press('Enter')
    await expect(page.getByTestId('playground-task-queue')).toContainText('A queued task')
    await expect.poll(() => readStoredQueuePrompts(page)).toEqual(['A queued task'])

    await page.getByRole('button', { name: 'New conversation' }).click()
    await input.fill('B direct task')
    await input.press('Enter')

    await expect.poll(() => requestCount).toBe(2)
    await expect(page.getByText('Response for B direct task')).toBeVisible({ timeout: 10000 })
    await expect(page.getByTestId('playground-task-queue')).toHaveCount(0)
    await expect.poll(() => readStoredQueuePrompts(page)).toEqual(['A queued task'])
  })

  test('allows queued prompts to be edited from the overflow menu', async ({ page }) => {
    let requestCount = 0
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      requestCount += 1
      await new Promise((resolve) => setTimeout(resolve, 6000))
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody('Streaming response for queued editing'),
      })
    })

    const input = page.getByPlaceholder('Ask me anything...')
    await input.fill('First queued task')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect.poll(() => requestCount).toBe(1)

    await input.fill('Editable queued task')
    await page.getByRole('button', { name: 'Send message' }).click()

    const queue = page.getByTestId('playground-task-queue')
    const queuedTask = queue
      .locator('[data-testid^="playground-task-queue-item-"]')
      .filter({ hasText: 'Editable queued task' })
      .first()

    await queuedTask.getByRole('button', { name: /More actions for queued task:/ }).click()
    await page.getByRole('menuitem', { name: 'Edit prompt' }).click()

    await expect(input).toHaveValue('Editable queued task')
    await expect(queue).toHaveCount(0)
    await expect.poll(() => readStoredQueuePrompts(page)).toEqual([])
  })

  test('allows queued tasks to be reordered by dragging', async ({ page }) => {
    let requestCount = 0
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      requestCount += 1
      await new Promise((resolve) => setTimeout(resolve, 6000))
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody('Streaming response for queue ordering'),
      })
    })

    const input = page.getByPlaceholder('Ask me anything...')
    await input.fill('First queued task')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect.poll(() => requestCount).toBe(1)
    const queue = page.getByTestId('playground-task-queue')

    await input.fill('Second queued task')
    await page.getByRole('button', { name: 'Send message' }).click()
    await input.fill('Third queued task')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(queue).toContainText('Second queued task')
    await expect(queue).toContainText('Third queued task')
    await expect(queue).not.toContainText('First queued task')
    const secondQueuedTask = queue
      .locator('[data-testid^="playground-task-queue-item-"]')
      .filter({ hasText: 'Second queued task' })
      .first()
    const thirdQueuedTask = queue
      .locator('[data-testid^="playground-task-queue-item-"]')
      .filter({ hasText: 'Third queued task' })
      .first()

    await thirdQueuedTask.dragTo(secondQueuedTask)

    await expect
      .poll(() => readStoredQueuePrompts(page))
      .toEqual(['Third queued task', 'Second queued task'])
  })

  test('anchors the current user turn near the top and respects manual scrolling during streaming', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 560 })

    await page.evaluate(() => {
      const now = Date.now()
      const history = Array.from({ length: 10 }, (_, index) => {
        const offset = (10 - index) * 60_000
        return [
          {
            id: `seed-user-${index + 1}`,
            role: 'user',
            content: `Earlier question ${index + 1}`,
            timestamp: new Date(now - offset).toISOString(),
          },
          {
            id: `seed-assistant-${index + 1}`,
            role: 'assistant',
            content: Array.from(
              { length: 8 },
              (_, paragraphIndex) =>
                `Earlier answer ${index + 1}, paragraph ${paragraphIndex + 1}: seeded history keeps the transcript tall.`,
            ).join('\n\n'),
            timestamp: new Date(now - offset + 15_000).toISOString(),
          },
        ]
      }).flat()

      window.localStorage.setItem('sr:playground:active-conversation', 'seeded-conversation')
      window.localStorage.setItem(
        'sr:chat:conversations',
        JSON.stringify([
          {
            id: 'seeded-conversation',
            createdAt: now - 3600_000,
            updatedAt: now,
            payload: history,
          },
        ]),
      )
    })
    await page.goto('/playground', { waitUntil: 'domcontentloaded' })

    const chunks = [
      chatStreamChunk({ role: 'assistant', content: '' }),
      ...Array.from({ length: 140 }, (_, index) =>
        chatStreamChunk({ content: `Paragraph ${index + 1}: streaming output keeps growing.\n\n` }),
      ),
      'data: [DONE]\n\n',
    ]

    await mockStreamingChatFetch(page, chunks, 25)

    await page.getByPlaceholder('Ask me anything...').fill('Show a long streamed answer')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(page.getByRole('button', { name: 'Stop generating' })).toBeVisible({
      timeout: 5000,
    })
    const currentAssistant = page.locator('[data-message-role="assistant"]').last()
    await expect(currentAssistant).toContainText('Paragraph 40: streaming output keeps growing.', {
      timeout: 10000,
    })

    const transcript = page.locator('[data-testid="chat-transcript"]')
    await expect
      .poll(
        async () => {
          return transcript.evaluate((node) => {
            const container = node as HTMLDivElement
            const userMessages = container.querySelectorAll<HTMLElement>(
              '[data-message-role="user"]',
            )
            const currentQuestion = userMessages[userMessages.length - 1]

            if (!currentQuestion) {
              return Number.POSITIVE_INFINITY
            }

            return (
              currentQuestion.getBoundingClientRect().top - container.getBoundingClientRect().top
            )
          })
        },
        { timeout: 5000 },
      )
      .toBeLessThan(120)

    const scrollTopBeforeManualScroll = await transcript.evaluate(
      (node) => (node as HTMLDivElement).scrollTop,
    )
    const manualScrollTop = await transcript.evaluate((node) => {
      const container = node as HTMLDivElement
      container.style.scrollBehavior = 'auto'
      const target = Math.max(0, container.scrollTop - container.clientHeight * 1.5)
      container.scrollTop = target
      container.dispatchEvent(new WheelEvent('wheel', { bubbles: true, deltaY: -1 }))
      return container.scrollTop
    })
    expect(manualScrollTop).toBeLessThan(scrollTopBeforeManualScroll - 300)

    await expect(currentAssistant).toContainText('Paragraph 140: streaming output keeps growing.', {
      timeout: 10000,
    })
  })

  test('keeps the assistant rail centered and stable during streaming and after completion', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1280, height: 720 })

    await mockStreamingChatFetch(
      page,
      [
        chatStreamChunk({ role: 'assistant', content: '' }),
        chatStreamChunk({ content: 'This starts the streamed response. ' }),
        chatStreamChunk({ content: 'More text arrives while the layout stays stable. ' }),
        chatStreamChunk({
          content: 'The final chunk lands without the message suddenly widening.',
        }),
        'data: [DONE]\n\n',
      ],
      220,
    )

    await page.getByPlaceholder('Ask me anything...').fill('Check the assistant layout rail')
    await page.getByRole('button', { name: 'Send message' }).click()

    const assistantContent = page
      .locator('[data-message-role="assistant"] [data-message-content]')
      .last()

    await expect(assistantContent).toBeVisible({ timeout: 5000 })

    // The rail centers in the chat column between the conversation pane and the trace.
    const measure = () =>
      assistantContent.evaluate((node) => {
        const rect = node.getBoundingClientRect()
        const transcript = node.closest('[data-testid="chat-transcript"]')!.getBoundingClientRect()
        return {
          offset: rect.left + rect.width / 2 - (transcript.left + transcript.width / 2),
          width: rect.width,
        }
      })
    const boxWhileStreaming = await measure()
    expect(boxWhileStreaming.width).toBeGreaterThan(560)
    expect(boxWhileStreaming.width).toBeLessThan(900)
    expect(Math.abs(boxWhileStreaming.offset)).toBeLessThan(60)

    await expect(
      page.getByText('The final chunk lands without the message suddenly widening.'),
    ).toBeVisible({
      timeout: 10000,
    })

    const boxAfterCompletion = await measure()
    expect(boxAfterCompletion.width).toBeGreaterThan(560)
    expect(boxAfterCompletion.width).toBeLessThan(900)
    expect(Math.abs(boxAfterCompletion.offset)).toBeLessThan(60)
    expect(Math.abs(boxAfterCompletion.width - boxWhileStreaming.width)).toBeLessThan(48)
  })

  test('keeps the composer pinned to the bottom on the second turn', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 })

    let requestCount = 0
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      requestCount += 1
      const body =
        requestCount === 1
          ? chatStreamBody('First answer closes out the opening turn.')
          : chatStreamBody(
              Array.from(
                { length: 28 },
                (_, index) => `Second-turn paragraph ${index + 1} keeps the response growing.`,
              ).join('\n\n'),
            )

      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body,
      })
    })

    const input = page.getByPlaceholder('Ask me anything...')
    const composer = page.getByTestId('chat-composer')

    await input.fill('Start the first turn')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect(page.getByText('First answer closes out the opening turn.')).toBeVisible({
      timeout: 10000,
    })

    await input.fill('Start the second turn')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect(
      page.getByText('Second-turn paragraph 16 keeps the response growing.'),
    ).toBeVisible({ timeout: 10000 })

    const composerBox = await composer.boundingBox()
    expect(composerBox).not.toBeNull()
    expect(composerBox!.y + composerBox!.height).toBeGreaterThan(820)

    const secondTurnMessage = page.locator('[data-message-role="user"]').last()
    const secondTurnBox = await secondTurnMessage.boundingBox()
    expect(secondTurnBox).not.toBeNull()
    expect(secondTurnBox!.y + secondTurnBox!.height).toBeLessThan(composerBox!.y - 24)
  })

  test('renders thinking block from streaming reasoning field', async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody('Final streamed answer.', 'Step 1: inspect the prompt.'),
      })
    })

    await page.getByPlaceholder('Ask me anything...').fill('Show your work')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(page.getByText('Final streamed answer.')).toBeVisible({ timeout: 10000 })
    await expect(page.getByText('Step 1: inspect the prompt.')).toBeVisible({ timeout: 10000 })
    await expect(page.getByRole('button', { name: /Reasoning/ })).toBeVisible({ timeout: 10000 })
  })

  test('keeps the waiting animation inside the active conversation rail', async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 800))
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody('Ready.'),
      })
    })

    await page.getByPlaceholder('Ask me anything...').fill('Keep the indicator with this turn')
    await page.getByRole('button', { name: 'Send message' }).click()

    const rail = page.getByTestId('chat-message-rail')
    const thinking = page.getByTestId('chat-thinking')
    await expect(thinking).toBeVisible({ timeout: 5000 })
    await expect(rail.locator('[data-testid="chat-thinking"]')).toHaveCount(1)

    const [railBox, thinkingBox] = await Promise.all([rail.boundingBox(), thinking.boundingBox()])
    expect(railBox).not.toBeNull()
    expect(thinkingBox).not.toBeNull()
    expect(thinkingBox!.x).toBeGreaterThanOrEqual(railBox!.x - 1)
    expect(thinkingBox!.x + thinkingBox!.width).toBeLessThanOrEqual(railBox!.x + railBox!.width + 1)
    await expect(page.getByText('Ready.')).toBeVisible({ timeout: 5000 })
  })

  test('shows streaming reasoning in thinking overlay before completion', async ({ page }) => {
    await mockStreamingChatFetch(page, [
      chatStreamChunk({ role: 'assistant', content: '' }),
      chatStreamChunk({ reasoning: 'The' }),
      chatStreamChunk({ reasoning: ' answer' }),
      chatStreamChunk({ content: 'Done.' }),
      'data: [DONE]\n\n',
    ])

    await page.getByPlaceholder('Ask me anything...').fill('Stream reasoning')
    await page.getByRole('button', { name: 'Send message' }).click()

    const thinkingBlock = page.getByTestId('thinking-block')
    await expect(thinkingBlock).toBeVisible({ timeout: 5000 })
    await expect(page.getByText('Classifying intent')).toHaveCount(0)
    await expect(page.getByText('Selecting route')).toHaveCount(0)
    await expect(page.getByText('Preparing response')).toHaveCount(0)
    await expect(thinkingBlock.getByText('The answer')).toBeVisible({ timeout: 5000 })
    await expect(page.getByText('Done.')).toBeVisible({ timeout: 10000 })
  })

  test('keeps the restored thinking matrix static with reduced motion', async ({ page }) => {
    await page.emulateMedia({ reducedMotion: 'reduce' })
    await page.reload({ waitUntil: 'domcontentloaded' })
    await mockStreamingChatFetch(
      page,
      [
        chatStreamChunk({ role: 'assistant', content: '' }),
        chatStreamChunk({ reasoning: 'Inspecting' }),
        chatStreamChunk({ reasoning: ' route' }),
        chatStreamChunk({ reasoning: ' state' }),
        chatStreamChunk({ content: 'Reduced motion complete.' }),
        'data: [DONE]\n\n',
      ],
      300,
    )

    await page.getByPlaceholder('Ask me anything...').fill('Respect reduced motion')
    await page.getByRole('button', { name: 'Send message' }).click()

    const thinkingBlock = page.getByTestId('thinking-block')
    await expect(thinkingBlock).toBeVisible({ timeout: 5000 })
    await expect(thinkingBlock.getByTestId('thinking-content')).toHaveCSS('animation-name', 'none')
    await expect(page.getByText('Reduced motion complete.')).toBeVisible({ timeout: 10000 })
  })

  test('renders thinking block from non-stream reasoning field', async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          choices: [
            {
              message: {
                role: 'assistant',
                content: 'Final JSON answer.',
                reasoning: 'Step 1: parse message.reasoning.',
              },
            },
          ],
        }),
      })
    })

    await page.getByPlaceholder('Ask me anything...').fill('Return JSON')
    await page.getByRole('button', { name: 'Send message' }).click()

    await expect(page.getByText('Final JSON answer.')).toBeVisible({ timeout: 10000 })
    await expect(page.getByText('Step 1: parse message.reasoning.')).toBeVisible({
      timeout: 10000,
    })
    await expect(page.getByRole('button', { name: /Reasoning/ })).toBeVisible({ timeout: 10000 })
  })

  test('shows the routed decision, signals, model and timings in the routing trace', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'x-vsr-selected-model': 'qwen/qwen3-32b',
          'x-vsr-selected-decision': 'computer_science_route',
          'x-vsr-selected-recipe': 'default',
          'x-vsr-replay-id': 'replay-123',
          'x-vsr-trace':
            'algorithm=confidence, confidence=0.86, routing_latency_ms=38, effective_input_tokens=412, signals=("domain:computer science" "keyword:code_keywords")',
        },
        body: chatStreamBody('Routed answer.'),
      })
    })
    await page.reload({ waitUntil: 'domcontentloaded' })

    const inspector = page.getByTestId('playground-inspector')
    await expect(inspector).toContainText('Nothing to inspect yet')

    await page.getByPlaceholder('Ask me anything...').fill('Why does my effect run twice?')
    await page.getByRole('button', { name: 'Send message' }).click()
    await expect(page.getByText('Routed answer.')).toBeVisible({ timeout: 10000 })

    await expect(inspector).toContainText('Computer Science')
    await expect(inspector).toContainText('computer_science_route')
    await expect(inspector).toContainText('qwen/qwen3-32b')
    await expect(inspector).toContainText('computer science')
    await expect(inspector).toContainText('code_keywords')
    await expect(inspector).toContainText('confidence 0.86')
    await expect(inspector).toContainText('Input tokens')
    await expect(inspector).toContainText('412')
    await expect(inspector.getByRole('link', { name: 'Open in Requests' })).toHaveAttribute(
      'href',
      '/observe/requests/replay-123',
    )
    await inspector.getByRole('tab', { name: 'Headers' }).click()
    await expect(inspector).toContainText('x-vsr-selected-decision')

    await page.getByRole('button', { name: 'Close routing trace' }).click()
    await expect(page.getByTestId('playground-inspector')).toHaveCount(0)
    await page
      .locator('[data-message-role="assistant"]')
      .last()
      .getByRole('button', { name: 'Inspect route' })
      .click()
    await expect(page.getByTestId('playground-inspector')).toContainText('computer_science_route')
  })
})
