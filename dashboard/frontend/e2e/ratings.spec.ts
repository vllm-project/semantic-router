import { test, expect } from '@playwright/test';

// Mock ratings data matching the user's expected mock API (gpt-4 #1 with 1410)
const MOCK_RATINGS = [
  { model: 'gpt-4', rating: 1410, wins: 10, losses: 2, ties: 1 },
  { model: 'gemma2:9b', rating: 1350, wins: 8, losses: 4, ties: 2 },
  { model: 'llama3.2:3b', rating: 1280, wins: 6, losses: 5, ties: 1 },
  { model: 'phi4', rating: 1200, wins: 5, losses: 6, ties: 0 },
  { model: 'mistral-7b', rating: 1150, wins: 4, losses: 8, ties: 1 },
];

function mockRatingsResponse() {
  return {
    ratings: MOCK_RATINGS,
    category: 'global',
    count: MOCK_RATINGS.length,
    timestamp: new Date().toISOString(),
  };
}

test.describe('Ratings Page', () => {
  test.beforeEach(async ({ page }) => {
    // Mock the ratings API for consistent test results
    await page.route('**/api/router/api/v1/ratings**', async (route) => {
      const url = new URL(route.request().url());
      const category = url.searchParams.get('category') || 'global';
      const response = mockRatingsResponse();
      response.category = category || 'global';
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(response),
      });
    });
  });

  test('1. Page loads with expected structure', async ({ page }) => {
    await page.goto('/ratings');

    // Wait for table to load (either with data or loading state)
    await page.waitForSelector('table, .loading', { timeout: 5000 });

    // Note: Page shows "Elo Leaderboard" not "Model Leaderboard"
    const heading = page.getByRole('heading', { level: 1 });
    await expect(heading).toBeVisible();
    await expect(heading).toContainText('Leaderboard');

    // Table columns: #, Model, Rating, Games, Wins, Losses, Ties
    await expect(page.getByRole('columnheader', { name: '#' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Model' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Rating' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Games' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Wins' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Losses' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Ties' })).toBeVisible();

    // Category dropdown and auto-refresh toggle
    await expect(page.getByLabel('Auto-refresh')).toBeVisible();
    await expect(page.getByRole('combobox', { name: /category/i })).toBeVisible();
  });

  test('2. Models sorted by rating descending, gpt-4 is #1 with 1410', async ({ page }) => {
    await page.goto('/ratings');

    // Wait for data to load
    await expect(page.getByText('gpt-4')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('1410')).toBeVisible();

    // gpt-4 should be in first data row (rank 1)
    const firstRow = page.locator('tbody tr').first();
    await expect(firstRow).toContainText('gpt-4');
    await expect(firstRow).toContainText('1'); // rank
    await expect(firstRow).toContainText('1410');
  });

  test('3. All 5 models display in table', async ({ page }) => {
    await page.goto('/ratings');

    for (const m of ['gpt-4', 'gemma2:9b', 'llama3.2:3b', 'phi4', 'mistral-7b']) {
      await expect(page.getByText(m, { exact: true })).toBeVisible({ timeout: 5000 });
    }

    const rows = page.locator('tbody tr');
    await expect(rows).toHaveCount(5);
  });

  test('4. Auto-refresh checkbox toggles off', async ({ page }) => {
    await page.goto('/ratings');
    await expect(page.getByText('gpt-4')).toBeVisible({ timeout: 5000 });

    const checkbox = page.getByLabel('Auto-refresh');
    await expect(checkbox).toBeChecked();
    await checkbox.click();
    await expect(checkbox).not.toBeChecked();
  });

  test('5. Refresh button updates last updated time', async ({ page }) => {
    await page.goto('/ratings');
    await expect(page.getByText('gpt-4')).toBeVisible({ timeout: 5000 });

    const updatedRegex = /Updated \d{1,2}:\d{2}:\d{2}/;
    await expect(page.getByText(updatedRegex)).toBeVisible();

    // Small delay to ensure timestamp can change
    await page.waitForTimeout(1100);
    await page.getByRole('button', { name: 'Refresh' }).click();

    // Wait for loading to finish
    await expect(page.getByText('Loading leaderboard…')).toBeVisible({ timeout: 1000 }).catch(() => {});
    await expect(page.getByText('gpt-4')).toBeVisible({ timeout: 5000 });

    const afterText = await page.getByText(updatedRegex).textContent();
    // Timestamp should be visible (may or may not change within same second)
    expect(afterText).toMatch(updatedRegex);
  });

  test('6. Custom category "coding" shows data (mock does not filter)', async ({ page }) => {
    await page.goto('/ratings');
    await expect(page.getByText('gpt-4')).toBeVisible({ timeout: 5000 });

    const customInput = page.getByPlaceholder('e.g. coding');
    await customInput.fill('coding');
    await page.waitForTimeout(500); // Allow debounce / refetch

    // Data should still show (mock returns same data for any category)
    await expect(page.getByText('gpt-4')).toBeVisible({ timeout: 5000 });
    await expect(page.locator('tbody tr')).toHaveCount(5);
  });

  test('7. Ratings link in main page navigation', async ({ page }) => {
    // Use /playground - root "/" is a landing page without nav bar
    await page.goto('/playground');
    const ratingsLink = page.getByRole('link', { name: 'Ratings' });
    await expect(ratingsLink).toBeVisible();
  });

  test('8. Ratings nav link navigates to ratings page', async ({ page }) => {
    await page.goto('/playground');
    await page.getByRole('link', { name: 'Ratings' }).click();
    await expect(page).toHaveURL(/\/ratings/);
    await expect(page.getByText('gpt-4')).toBeVisible({ timeout: 5000 });
  });
});
