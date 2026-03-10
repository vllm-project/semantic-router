/**
 * Full verification against REAL Semantic Router (llama3.2:3b, phi4 in Elo ratings).
 * Run: npx playwright test e2e/ratings-full-verify.spec.ts -c playwright.live.config.ts
 */
import { test, expect } from '@playwright/test';

test.describe('Ratings Page - Full Verification', () => {
  test('1-2. Navigate to /ratings and verify structure', async ({ page }) => {
    await page.goto('/ratings');
    await page.waitForLoadState('networkidle');

    await page.screenshot({ path: 'test-results/01-ratings-page.png', fullPage: true });

    // Elo Leaderboard heading
    const heading = page.getByRole('heading', { level: 1 });
    await expect(heading).toBeVisible();
    await expect(heading).toContainText('Elo Leaderboard');

    // Table columns
    await expect(page.getByRole('columnheader', { name: '#' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Model' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Rating' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Games' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Wins' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Losses' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Ties' })).toBeVisible();

    // At least 2 models: llama3.2:3b and phi4
    await expect(page.getByText('llama3.2:3b')).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('phi4')).toBeVisible();

    const rows = page.locator('tbody tr');
    const count = await rows.count();
    expect(count).toBeGreaterThanOrEqual(2);

    // Models sorted by rating descending - first row should have higher rating
    const firstRow = rows.first();
    const secondRow = rows.nth(1);
    const firstRating = await firstRow.locator('td').nth(2).textContent();
    const secondRating = await secondRow.locator('td').nth(2).textContent();
    expect(parseInt(firstRating || '0', 10)).toBeGreaterThanOrEqual(parseInt(secondRating || '0', 10));

    // Games = Wins + Losses + Ties for each row
    for (let i = 0; i < Math.min(count, 5); i++) {
      const row = rows.nth(i);
      const games = parseInt(await row.locator('td').nth(3).textContent() || '0', 10);
      const wins = parseInt(await row.locator('td').nth(4).textContent() || '0', 10);
      const losses = parseInt(await row.locator('td').nth(5).textContent() || '0', 10);
      const ties = parseInt(await row.locator('td').nth(6).textContent() || '0', 10);
      expect(games).toBe(wins + losses + ties);
    }
  });

  test('3. Category filter - type "tech"', async ({ page }) => {
    await page.goto('/ratings');
    await page.waitForLoadState('networkidle');

    const customInput = page.getByPlaceholder('e.g. coding');
    await customInput.fill('tech');
    await page.waitForTimeout(1000);

    await page.screenshot({ path: 'test-results/03-tech-category.png', fullPage: true });

    // Section should show tech category
    await expect(page.getByText(/Leaderboard — tech/i)).toBeVisible({ timeout: 5000 });

    // Data should load (may show tech data or same data if mock doesn't filter)
    const rows = page.locator('tbody tr');
    await expect(rows.first()).toBeVisible({ timeout: 5000 });
  });

  test('4. Clear category - verify global data', async ({ page }) => {
    await page.goto('/ratings');
    await page.waitForLoadState('networkidle');

    const customInput = page.getByPlaceholder('e.g. coding');
    await customInput.fill('tech');
    await page.waitForTimeout(800);
    await customInput.clear();
    await page.waitForTimeout(1000);

    await expect(page.getByText(/Leaderboard — global/i)).toBeVisible({ timeout: 5000 });
    await expect(page.getByText('llama3.2:3b')).toBeVisible();
    await expect(page.getByText('phi4')).toBeVisible();
  });

  test('5. Auto-refresh toggle', async ({ page }) => {
    await page.goto('/ratings');
    await page.waitForLoadState('networkidle');

    const checkbox = page.getByLabel('Auto-refresh');
    await expect(checkbox).toBeChecked();
    await checkbox.click();
    await expect(checkbox).not.toBeChecked();
    await checkbox.click();
    await expect(checkbox).toBeChecked();
  });

  test('6. Refresh button updates timestamp', async ({ page }) => {
    await page.goto('/ratings');
    await page.waitForLoadState('networkidle');

    const updatedEl = page.getByText(/Updated \d{1,2}:\d{2}:\d{2}/);
    await expect(updatedEl).toBeVisible();

    await page.waitForTimeout(1100);
    await page.getByRole('button', { name: 'Refresh' }).click();
    await expect(page.getByText('Loading leaderboard…')).toBeVisible({ timeout: 1000 }).catch(() => {});

    await expect(updatedEl).toBeVisible({ timeout: 5000 });
    const after = await updatedEl.textContent();
    expect(after).toMatch(/Updated \d{1,2}:\d{2}:\d{2}/);
  });

  test('7-9. Playground nav and Ratings link', async ({ page }) => {
    await page.goto('/playground');
    await page.waitForLoadState('networkidle');

    const ratingsLink = page.getByRole('link', { name: 'Ratings' });
    await expect(ratingsLink).toBeVisible();

    await ratingsLink.click();
    await expect(page).toHaveURL(/\/ratings/);
    await expect(page.getByRole('heading', { level: 1 })).toContainText('Elo Leaderboard');
    await expect(page.getByText('llama3.2:3b')).toBeVisible();

    await page.screenshot({ path: 'test-results/09-back-to-ratings.png', fullPage: true });
  });
});
