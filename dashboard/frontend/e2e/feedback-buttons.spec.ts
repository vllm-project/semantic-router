import { test, expect } from '@playwright/test';

/**
 * FeedbackButtons E2E tests.
 *
 * FeedbackButtons render inside the ChatComponent when an assistant message
 * has a model ID (from `x-vsr-selected-model` header). We mock the chat
 * completions endpoint to produce a response that triggers FeedbackButtons,
 * then test the feedback submission flow.
 */

const MOCK_MODEL = 'llama3.2:3b';

function chatStreamBody(content: string, model: string): string {
  const lines = content.split('').map((char) =>
    `data: ${JSON.stringify({
      choices: [{ delta: { content: char } }],
      model,
    })}\n\n`
  );
  return lines.join('') + 'data: [DONE]\n\n';
}

test.describe('FeedbackButtons', () => {
  test.beforeEach(async ({ page }) => {
    await page.route('**/api/router/v1/chat/completions', async (route) => {
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'x-vsr-selected-model': MOCK_MODEL,
          'x-vsr-selected-decision': 'tech',
        },
        body: chatStreamBody('Hello, this is a test response.', MOCK_MODEL),
      });
    });

    await page.goto('/playground');
  });

  test('thumbs up sends correct feedback payload', async ({ page }) => {
    let feedbackPayload: Record<string, unknown> | null = null;

    await page.route('**/api/router/api/v1/feedback', async (route) => {
      feedbackPayload = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ success: true, message: 'Feedback recorded' }),
      });
    });

    const input = page.getByPlaceholder(/ask me anything|type a message/i);
    await input.fill('What is machine learning?');
    await page.getByRole('button', { name: /send|📤/i }).click();
    await expect(page.getByText('Hello, this is a test response.')).toBeVisible({ timeout: 10000 });

    const thumbsUp = page.getByRole('button', { name: /good response/i });
    await expect(thumbsUp).toBeVisible({ timeout: 5000 });
    await thumbsUp.click();

    expect(feedbackPayload).not.toBeNull();
    expect(feedbackPayload!.winner_model).toBe(MOCK_MODEL);
    expect(feedbackPayload!.decision_name).toBe('tech');
    await expect(page.getByText('Feedback Sent!')).toBeVisible({ timeout: 3000 });
  });

  test('handles feedback API error gracefully', async ({ page }) => {
    await page.route('**/api/router/api/v1/feedback', async (route) => {
      await route.fulfill({
        status: 500,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ error: { message: 'Server unavailable' } }),
      });
    });

    const input = page.getByPlaceholder(/ask me anything|type a message/i);
    await input.fill('Test error handling');
    await page.getByRole('button', { name: /send|📤/i }).click();
    await expect(page.getByText('Hello, this is a test response.')).toBeVisible({ timeout: 10000 });

    const thumbsUp = page.getByRole('button', { name: /good response/i });
    await expect(thumbsUp).toBeVisible({ timeout: 5000 });
    await thumbsUp.click();

    await expect(page.getByText(/server unavailable/i)).toBeVisible({ timeout: 5000 });
  });

  test('thumbs down in single-model mode selects visually without error', async ({ page }) => {
    const input = page.getByPlaceholder(/ask me anything|type a message/i);
    await input.fill('Test thumbs down');
    await page.getByRole('button', { name: /send|📤/i }).click();
    await expect(page.getByText('Hello, this is a test response.')).toBeVisible({ timeout: 10000 });

    const thumbsDown = page.getByRole('button', { name: /bad response/i });
    await expect(thumbsDown).toBeVisible({ timeout: 5000 });
    await thumbsDown.click();

    // Should select visually (aria-pressed=true), no error message
    await expect(thumbsDown).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByRole('alert')).not.toBeVisible();
  });

  test('can toggle from up to down (only submits once)', async ({ page }) => {
    let feedbackCallCount = 0;

    await page.route('**/api/router/api/v1/feedback', async (route) => {
      feedbackCallCount++;
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ success: true }),
      });
    });

    const input = page.getByPlaceholder(/ask me anything|type a message/i);
    await input.fill('Test toggle');
    await page.getByRole('button', { name: /send|📤/i }).click();
    await expect(page.getByText('Hello, this is a test response.')).toBeVisible({ timeout: 10000 });

    const thumbsUp = page.getByRole('button', { name: /good response/i });
    const thumbsDown = page.getByRole('button', { name: /bad response/i });
    await expect(thumbsUp).toBeVisible({ timeout: 5000 });

    // Click thumbs up first
    await thumbsUp.click();
    await expect(page.getByText('Feedback Sent!')).toBeVisible({ timeout: 3000 });
    expect(feedbackCallCount).toBe(1);

    // Now toggle to thumbs down - should change visual but NOT call API again
    await thumbsDown.click();
    await expect(thumbsDown).toHaveAttribute('aria-pressed', 'true');
    await expect(thumbsUp).toHaveAttribute('aria-pressed', 'false');
    expect(feedbackCallCount).toBe(1); // Still 1 - no second API call

    // Toggle back to thumbs up - still no extra API call (UP → DOWN → UP)
    await thumbsUp.click();
    await expect(thumbsUp).toHaveAttribute('aria-pressed', 'true');
    await expect(thumbsDown).toHaveAttribute('aria-pressed', 'false');
    expect(feedbackCallCount).toBe(1); // Still 1 - deterministic single submission
  });

  test('buttons disabled during loading', async ({ page }) => {
    await page.route('**/api/router/api/v1/feedback', async (route) => {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ success: true }),
      });
    });

    const input = page.getByPlaceholder(/ask me anything|type a message/i);
    await input.fill('Test disabled');
    await page.getByRole('button', { name: /send|📤/i }).click();
    await expect(page.getByText('Hello, this is a test response.')).toBeVisible({ timeout: 10000 });

    const thumbsUp = page.getByRole('button', { name: /good response/i });
    await expect(thumbsUp).toBeVisible({ timeout: 5000 });
    await thumbsUp.click();

    // During loading, buttons should be disabled
    await expect(thumbsUp).toBeDisabled();
  });
});
