import { expect, test } from '@playwright/test';
import { mockAuthenticatedSession } from './support/auth';

const evalUser = {
  id: 'user-eval-1',
  email: 'eval@example.com',
  name: 'Eval User',
  role: 'read',
  permissions: ['config.read', 'evaluation.read', 'evaluation.run', 'evaluation.write', 'logs.read', 'topology.read'],
};

test.describe('Evaluation page', () => {
  test('loads evaluation page and shows signal/system workflow UI', async ({ page }) => {
    await mockAuthenticatedSession(page, { user: evalUser });

    await page.route('**/api/evaluation/tasks', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify([]),
        });
      } else {
        await route.continue();
      }
    });
    await page.route('**/api/evaluation/datasets', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain: [{ name: 'mmlu-pro-en', description: 'MMLU-Pro (English)', dimension: 'domain', level: 'router' }],
          fact_check: [{ name: 'fact-check-en', description: 'Fact Check (English)', dimension: 'fact_check', level: 'router' }],
          user_feedback: [{ name: 'feedback-en', description: 'User Feedback (English)', dimension: 'user_feedback', level: 'router' }],
          accuracy: [{ name: 'mmlu-pro', description: 'MMLU-Pro system accuracy', dimension: 'accuracy', level: 'mom' }],
        }),
      });
    });

    await page.goto('/evaluation');
    await expect(page.getByRole('heading', { name: /evaluation/i })).toBeVisible();
    await expect(page.getByText(/signal and system level/i)).toBeVisible();
    await expect(page.getByRole('tab', { name: /tasks/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /create/i })).toBeVisible();
    await expect(page.getByRole('tab', { name: /history/i })).toBeVisible();
  });

  test('create tab shows signal and system level options', async ({ page }) => {
    await mockAuthenticatedSession(page, { user: evalUser });

    await page.route('**/api/evaluation/tasks', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({ status: 200, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([]) });
      } else {
        await route.continue();
      }
    });
    await page.route('**/api/evaluation/datasets', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          domain: [],
          fact_check: [],
          user_feedback: [],
          accuracy: [],
        }),
      });
    });

    await page.goto('/evaluation');
    await page.getByRole('tab', { name: /create/i }).click();
    await expect(page.getByText(/evaluation level/i)).toBeVisible();
    await expect(page.getByText(/signal level/i)).toBeVisible();
    await expect(page.getByText(/system level/i)).toBeVisible();
  });
});
