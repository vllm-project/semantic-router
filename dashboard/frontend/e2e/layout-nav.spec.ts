import { expect, test, type Page } from '@playwright/test';

const setupState = {
  setupMode: false,
  listenerPort: 8000,
  models: 1,
  decisions: 1,
  hasModels: true,
  hasDecisions: true,
  canActivate: true,
};

const settingsResponse = {
  readonlyMode: false,
  setupMode: false,
  platform: '',
  envoyUrl: '',
};

async function mockCommon(page: Page) {
  await page.route('**/api/setup/state', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(setupState) });
  });

  await page.route('**/api/settings', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(settingsResponse) });
  });

  await page.route('**/api/mcp/tools', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ tools: [] }) });
  });

  await page.route('**/api/mcp/servers', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) });
  });
}

test.describe('Layout top navigation', () => {
  test('moves ClawOS into the secondary group and merges analysis and operations', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockCommon(page);

    await page.goto('/playground');

    const globalNav = page.getByRole('navigation', { name: 'Global navigation' });
    const primaryGroup = page.getByRole('group', { name: 'Primary navigation' });
    const secondaryGroup = page.getByRole('group', { name: 'Secondary navigation' });

    await expect(primaryGroup.getByRole('link', { name: 'Dashboard' })).toBeVisible();
    await expect(primaryGroup.getByRole('link', { name: 'Playground' })).toBeVisible();
    await expect(primaryGroup.getByRole('link', { name: 'Brain' })).toBeVisible();
    await expect(primaryGroup.getByRole('link', { name: 'DSL' })).toBeVisible();
    await expect(primaryGroup.getByRole('button', { name: 'Manager' })).toBeVisible();
    await expect(primaryGroup.getByRole('link', { name: 'ClawOS' })).toHaveCount(0);

    await expect(secondaryGroup.getByRole('link', { name: 'ClawOS' })).toBeVisible();
    await expect(secondaryGroup.getByRole('button', { name: 'Command' })).toBeVisible();
    await expect(globalNav.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);
    await expect(globalNav.getByRole('button', { name: 'Operations', exact: true })).toHaveCount(0);

    await secondaryGroup.getByRole('button', { name: 'Command' }).click();

    const menu = page.getByRole('menu', { name: 'Command' });
    await expect(menu.getByText('Analysis')).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Evaluation' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Replay' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Ratings' })).toBeVisible();
    await expect(menu.getByText('Operations')).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'ML Setup' })).toHaveCount(0);
    await expect(menu.getByRole('menuitem', { name: 'Router Config' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'MCP Servers' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Status' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Logs' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Grafana' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Tracing' })).toBeVisible();
  });
});
