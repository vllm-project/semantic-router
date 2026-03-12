import { test, expect } from '@playwright/test';
import { mockAuthenticatedAppShell } from './support/auth';

function chatStreamChunk(delta: Record<string, unknown>): string {
  return `data: ${JSON.stringify({ choices: [{ index: 0, delta }] })}\n\n`;
}

function chatStreamBody(content: string, reasoning = ''): string {
  const initialLine = chatStreamChunk({ role: 'assistant', content: '' });
  const reasoningLines = reasoning
    ? reasoning.split('').map((char) =>
        chatStreamChunk({ reasoning: char })
      )
    : [];

  const contentLines = content.split('').map((char) =>
    chatStreamChunk({ content: char })
  );

  return initialLine + [...reasoningLines, ...contentLines].join('') + 'data: [DONE]\n\n';
}

async function mockStreamingChatFetch(
  page: import('@playwright/test').Page,
  chunks: string[],
  delayMs = 250,
): Promise<void> {
  await page.evaluate(async ({ chunks: streamChunks, delayMs: streamDelayMs }) => {
    const originalFetch = window.fetch.bind(window);
    const encoder = new TextEncoder();

    window.fetch = async (input, init) => {
      const url = typeof input === 'string'
        ? input
        : input instanceof Request
          ? input.url
          : String(input);

      if (!url.includes('/api/router/v1/chat/completions')) {
        return originalFetch(input, init);
      }

      let chunkIndex = 0;
      return new Response(new ReadableStream({
        start(controller) {
          const pushChunk = () => {
            if (chunkIndex >= streamChunks.length) {
              controller.close();
              return;
            }

            controller.enqueue(encoder.encode(streamChunks[chunkIndex]));
            chunkIndex += 1;
            window.setTimeout(pushChunk, streamDelayMs);
          };

          pushChunk();
        },
      }), {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
      });
    };
  }, { chunks, delayMs });
}

async function mockPlaygroundBootstrap(page: import('@playwright/test').Page): Promise<void> {
  await mockAuthenticatedAppShell(page);
}

test.describe('Playground Chat Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('sr:playground:claw-mode', 'false');
    });
    await mockPlaygroundBootstrap(page);
    await page.goto('/playground');
  });

  test('renders chat interface', async ({ page }) => {
    // Verify main elements are present
    await expect(page.getByPlaceholder('Ask me anything...')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Send message' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'New conversation' })).toBeVisible();
  });

  test('can type message', async ({ page }) => {
    const input = page.getByPlaceholder('Ask me anything...');
    await input.fill('Hello, this is a test message');
    await expect(input).toHaveValue('Hello, this is a test message');
  });

  test('send button disabled when input empty', async ({ page }) => {
    const sendButton = page.getByRole('button', { name: 'Send message' });
    // Button should be disabled when input is empty
    await expect(sendButton).toBeDisabled();
    
    // Type something
    await page.getByPlaceholder('Ask me anything...').fill('test');
    
    // Button should be enabled
    await expect(sendButton).toBeEnabled();
  });

  test('new conversation clears messages', async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody('Hello! This is a mock response.'),
      });
    });

    await page.getByPlaceholder('Ask me anything...').fill('Clear me');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.getByText('Clear me')).toBeVisible({ timeout: 10000 });

    await page.getByRole('button', { name: 'New conversation' }).click();

    await expect(page.getByText('Clear me')).not.toBeVisible();
    await expect(page.getByRole('heading', { name: /Hi there, I am MoM/i })).toBeVisible();
  });

  test('sends message and receives response (mocked API)', async ({ page }) => {
    // Mock the chat API endpoint
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      const request = route.request();
      const postData = request.postDataJSON();
      
      // Verify request structure
      expect(postData).toHaveProperty('messages');
      expect(postData).toHaveProperty('model');
      expect(postData).toHaveProperty('stream');
      
      // Return mock streaming response
      const responseText = 'Hello! This is a mock response.';
      
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody(responseText),
      });
    });

    // Type a message
    const input = page.getByPlaceholder('Ask me anything...');
    await input.fill('Hello, how are you?');
    
    // Send the message
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // User message should appear
    await expect(page.getByText('Hello, how are you?')).toBeVisible();
    
    // Wait for response to appear (the mocked response)
    await expect(page.getByText('Hello! This is a mock response.')).toBeVisible({ timeout: 10000 });
    
    // Input should be cleared after sending
    await expect(input).toHaveValue('');
  });

  test('handles API error gracefully', async ({ page }) => {
    // Mock API to return an error
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    // Type and send a message
    await page.getByPlaceholder('Ask me anything...').fill('Test error handling');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // User message should still appear
    await expect(page.getByText('Test error handling')).toBeVisible();
    
    // Error should be displayed (specific API error message)
    await expect(page.getByText('API error:')).toBeVisible({ timeout: 5000 });
  });

  test('stop button appears during streaming', async ({ page }) => {
    // Mock a slow streaming response
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      // Delay response to allow stop button to appear
      await new Promise(resolve => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
        body: 'data: {"choices":[{"delta":{"content":"Test"}}]}\n\ndata: [DONE]\n\n',
      });
    });

    // Send a message
    await page.getByPlaceholder('Ask me anything...').fill('Test streaming');
    await page.getByRole('button', { name: 'Send message' }).click();
    
    // Stop button should appear (look for it quickly before response completes)
    await expect(page.getByRole('button', { name: 'Stop generating' })).toBeVisible({ timeout: 1000 });
  });

  test('anchors the current user turn near the top and respects manual scrolling during streaming', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 560 });

    await page.evaluate(() => {
      const now = Date.now();
      const history = Array.from({ length: 10 }, (_, index) => {
        const offset = (10 - index) * 60_000;
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
                `Earlier answer ${index + 1}, paragraph ${paragraphIndex + 1}: seeded history keeps the transcript tall.`
            ).join('\n\n'),
            timestamp: new Date(now - offset + 15_000).toISOString(),
          },
        ];
      }).flat();

      window.localStorage.setItem(
        'sr:chat:conversations',
        JSON.stringify([
          {
            id: 'seeded-conversation',
            createdAt: now - 3600_000,
            updatedAt: now,
            payload: history,
          },
        ])
      );
    });
    await page.reload();

    const chunks = [
      chatStreamChunk({ role: 'assistant', content: '' }),
      ...Array.from({ length: 140 }, (_, index) =>
        chatStreamChunk({ content: `Paragraph ${index + 1}: streaming output keeps growing.\n\n` })
      ),
      'data: [DONE]\n\n',
    ];

    await mockStreamingChatFetch(page, chunks, 25);

    await page.getByPlaceholder('Ask me anything...').fill('Show a long streamed answer');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.getByRole('button', { name: 'Stop generating' })).toBeVisible({ timeout: 5000 });
    const currentAssistant = page.locator('[data-message-role="assistant"]').last();
    await expect(currentAssistant).toContainText('Paragraph 40: streaming output keeps growing.', { timeout: 10000 });

    const transcript = page.locator('[data-testid="chat-transcript"]');
    await expect.poll(async () => {
      return transcript.evaluate(node => {
        const container = node as HTMLDivElement;
        const userMessages = container.querySelectorAll<HTMLElement>('[data-message-role="user"]');
        const currentQuestion = userMessages[userMessages.length - 1];

        if (!currentQuestion) {
          return Number.POSITIVE_INFINITY;
        }

        return currentQuestion.getBoundingClientRect().top - container.getBoundingClientRect().top;
      });
    }, { timeout: 5000 }).toBeLessThan(120);

    const scrollTopBeforeManualScroll = await transcript.evaluate(node => (node as HTMLDivElement).scrollTop);
    await transcript.hover();
    await page.mouse.wheel(0, -5000);
    await page.waitForTimeout(600);

    const scrollTopAfterManualScroll = await transcript.evaluate(node => (node as HTMLDivElement).scrollTop);
    expect(scrollTopAfterManualScroll).toBeLessThan(scrollTopBeforeManualScroll - 1000);

    await expect(currentAssistant).toContainText('Paragraph 140: streaming output keeps growing.', { timeout: 10000 });
  });

  test('keeps the assistant rail centered and stable during streaming and after completion', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });

    await mockStreamingChatFetch(page, [
      chatStreamChunk({ role: 'assistant', content: '' }),
      chatStreamChunk({ content: 'This starts the streamed response. ' }),
      chatStreamChunk({ content: 'More text arrives while the layout stays stable. ' }),
      chatStreamChunk({ content: 'The final chunk lands without the message suddenly widening.' }),
      'data: [DONE]\n\n',
    ], 220);

    await page.getByPlaceholder('Ask me anything...').fill('Check the assistant layout rail');
    await page.getByRole('button', { name: 'Send message' }).click();

    const assistantContent = page.locator('[data-message-role="assistant"] [data-message-content]').last();

    await expect(assistantContent).toBeVisible({ timeout: 5000 });

    const boxWhileStreaming = await assistantContent.evaluate(node => {
      const rect = node.getBoundingClientRect();
      return {
        center: rect.left + rect.width / 2,
        width: rect.width,
      };
    });
    expect(boxWhileStreaming.width).toBeGreaterThan(560);
    expect(boxWhileStreaming.width).toBeLessThan(900);
    expect(Math.abs(boxWhileStreaming.center - 640)).toBeLessThan(120);

    await expect(page.getByText('The final chunk lands without the message suddenly widening.')).toBeVisible({
      timeout: 10000,
    });

    const boxAfterCompletion = await assistantContent.evaluate(node => {
      const rect = node.getBoundingClientRect();
      return {
        center: rect.left + rect.width / 2,
        width: rect.width,
      };
    });
    expect(boxAfterCompletion.width).toBeGreaterThan(560);
    expect(boxAfterCompletion.width).toBeLessThan(900);
    expect(Math.abs(boxAfterCompletion.center - 640)).toBeLessThan(120);
    expect(Math.abs(boxAfterCompletion.width - boxWhileStreaming.width)).toBeLessThan(48);
  });

  test('renders thinking block from streaming reasoning field', async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: chatStreamBody('Final streamed answer.', 'Step 1: inspect the prompt.'),
      });
    });

    await page.getByPlaceholder('Ask me anything...').fill('Show your work');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.getByText('Final streamed answer.')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Step 1: inspect the prompt.')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Completed Deep Thinking')).toBeVisible({ timeout: 10000 });
  });

  test('shows streaming reasoning in thinking overlay before completion', async ({ page }) => {
    await mockStreamingChatFetch(page, [
      chatStreamChunk({ role: 'assistant', content: '' }),
      chatStreamChunk({ reasoning: 'The' }),
      chatStreamChunk({ reasoning: ' answer' }),
      chatStreamChunk({ content: 'Done.' }),
      'data: [DONE]\n\n',
    ]);

    await page.getByPlaceholder('Ask me anything...').fill('Stream reasoning');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.getByText('Thinking Process:')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('pre').filter({ hasText: 'The answer' })).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('Done.')).toBeVisible({ timeout: 10000 });
  });

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
      });
    });

    await page.getByPlaceholder('Ask me anything...').fill('Return JSON');
    await page.getByRole('button', { name: 'Send message' }).click();

    await expect(page.getByText('Final JSON answer.')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Step 1: parse message.reasoning.')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Completed Deep Thinking')).toBeVisible({ timeout: 10000 });
  });
});
