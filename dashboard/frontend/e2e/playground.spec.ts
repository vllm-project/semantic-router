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
