import { test, expect } from '@playwright/test';

test.describe('Playground Chat Component', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/playground');
  });

  test('renders chat interface', async ({ page }) => {
    // Verify main elements are present
    await expect(page.getByPlaceholder('Type a message')).toBeVisible();
    await expect(page.getByRole('button', { name: '📤' })).toBeVisible();
    await expect(page.getByRole('button', { name: '🗑️' })).toBeVisible();
  });

  test('can type message', async ({ page }) => {
    const input = page.getByPlaceholder('Type a message');
    await input.fill('Hello, this is a test message');
    await expect(input).toHaveValue('Hello, this is a test message');
  });

  test('send button disabled when input empty', async ({ page }) => {
    const sendButton = page.getByRole('button', { name: '📤' });
    // Button should be disabled when input is empty
    await expect(sendButton).toBeDisabled();
    
    // Type something
    await page.getByPlaceholder('Type a message').fill('test');
    
    // Button should be enabled
    await expect(sendButton).toBeEnabled();
  });

  test('clear button clears messages', async ({ page }) => {
    // Initially should show empty state
    await expect(page.getByText('Start a conversation')).toBeVisible();

    // Click clear (should have no effect but shouldn't error)
    await page.getByRole('button', { name: '🗑️' }).click();

    // Empty state should still be visible
    await expect(page.getByText('Start a conversation')).toBeVisible();
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
        body: responseText.split('').map((char) => 
          `data: ${JSON.stringify({choices: [{delta: {content: char}}]})}\n\n`
        ).join('') + 'data: [DONE]\n\n',
      });
    });

    // Type a message
    const input = page.getByPlaceholder('Type a message');
    await input.fill('Hello, how are you?');
    
    // Send the message
    await page.getByRole('button', { name: '📤' }).click();
    
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
    await page.getByPlaceholder('Type a message').fill('Test error handling');
    await page.getByRole('button', { name: '📤' }).click();
    
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
    await page.getByPlaceholder('Type a message').fill('Test streaming');
    await page.getByRole('button', { name: '📤' }).click();
    
    // Stop button should appear (look for it quickly before response completes)
    await expect(page.getByRole('button', { name: '⏹️' })).toBeVisible({ timeout: 1000 });
  });
});

