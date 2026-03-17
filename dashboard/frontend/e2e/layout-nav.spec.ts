import { expect, test, type Page } from '@playwright/test';
import { mockAuthenticatedSession } from './support/auth';

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

const readUser = {
  id: 'user-read-1',
  email: 'viewer@example.com',
  name: 'Viewer User',
  role: 'read',
  permissions: [
    'config.read',
    'evaluation.read',
    'logs.read',
    'mcp.read',
    'openclaw.read',
    'tools.use',
    'topology.read',
  ],
};

const configResponse = {
  version: 'v0.3',
  listeners: [{ name: 'public', address: '0.0.0.0', port: 8801 }],
  providers: {
    defaults: {
      default_model: 'test-model',
      reasoning_families: {
        qwen3: {
          type: 'reasoning_effort',
          parameter: 'reasoning_effort',
        },
      },
    },
    models: [
      {
        name: 'test-model',
        provider_model_id: 'test-model',
        backend_refs: [
          {
            name: 'endpoint1',
            endpoint: '127.0.0.1:8000',
            protocol: 'http',
          },
        ],
      },
    ],
  },
  routing: {
    modelCards: [{ name: 'test-model' }],
    signals: {
      domains: [{ name: 'business', description: 'Business' }],
    },
    decisions: [
      {
        name: 'business-route',
        priority: 1,
        rules: {
          operator: 'AND',
          conditions: [{ type: 'domain', name: 'business' }],
        },
        modelRefs: [{ model: 'test-model' }],
      },
    ],
  },
  global: {
    router: { strategy: 'priority' },
    services: {
      response_api: { enabled: true },
    },
  },
};

const rawGlobalYaml = `router:
  strategy: priority
services:
  response_api:
    enabled: true
`;

const statusResponse = {
  overall: 'healthy',
  deployment_type: 'local',
  services: [
    {
      name: 'router',
      status: 'healthy',
      healthy: true,
    },
  ],
  models: {
    models: [],
    summary: {
      loaded_models: 0,
      total_models: 0,
    },
  },
};

async function mockCommon(
  page: Page,
  options: {
    user?: {
      id: string;
      email: string;
      name: string;
      role?: string;
      permissions?: string[];
    };
  } = {},
) {
  await mockAuthenticatedSession(page, options);

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

  await page.route('**/api/router/config/all', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(configResponse) });
  });

  await page.route('**/api/router/config/global', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(configResponse.global),
    });
  });

  await page.route('**/api/router/config/global/raw', async route => {
    await route.fulfill({ status: 200, contentType: 'text/yaml', body: rawGlobalYaml });
  });

  await page.route('**/api/status', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(statusResponse),
    });
  });

  await page.route('**/api/auth/bootstrap/can-register', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ canRegister: false }),
    });
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
    await expect(primaryGroup.getByRole('link', { name: 'Users' })).toHaveCount(0);

    await expect(secondaryGroup.getByRole('link', { name: 'Users' })).toBeVisible();
    await expect(secondaryGroup.getByRole('link', { name: 'ClawOS' })).toBeVisible();
    await expect(secondaryGroup.getByRole('button', { name: 'System' })).toBeVisible();
    await expect(globalNav.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);
    await expect(globalNav.getByRole('button', { name: 'Operations', exact: true })).toHaveCount(0);

    await secondaryGroup.getByRole('button', { name: 'System' }).click();

    const menu = page.getByRole('menu', { name: 'System' });
    await expect(menu.getByText('Analysis')).toBeVisible();
    const menuItems = menu.getByRole('menuitem');
    await expect(menuItems.nth(0)).toHaveText('Global Config');
    await expect(menuItems.nth(1)).toHaveText('Evaluation');
    await expect(menu.getByRole('menuitem', { name: 'Global Config' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Evaluation' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Replay' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Ratings' })).toBeVisible();
    await expect(menu.getByText('Operations')).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'ML Setup' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'MCP Servers' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Status' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Logs' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Grafana' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Tracing' })).toBeVisible();
  });

  test('hides ML Setup for read users and redirects direct access back to the dashboard', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockCommon(page, { user: readUser });

    await page.goto('/dashboard');

    await expect(page.getByRole('button', { name: 'ML Setup', exact: true })).toHaveCount(0);

    const secondaryGroup = page.getByRole('group', { name: 'Secondary navigation' });
    await secondaryGroup.getByRole('button', { name: 'System' }).click();

    const menu = page.getByRole('menu', { name: 'System' });
    await expect(menu.getByRole('menuitem', { name: 'ML Setup' })).toHaveCount(0);
    await expect(menu.getByRole('menuitem', { name: 'MCP Servers' })).toBeVisible();

    await page.goto('/ml-setup');

    await expect(page).toHaveURL(/\/dashboard$/);
    await expect(page.getByRole('button', { name: 'ML Setup', exact: true })).toHaveCount(0);
  });

  test('opens a centered account dialog and allows logging out from the header', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockCommon(page, {
      user: {
        id: 'user-admin-2',
        email: 'ada@example.com',
        name: 'Ada Lovelace',
        role: 'admin',
        permissions: ['config.read', 'config.write', 'users.manage'],
      },
    });

    await page.goto('/dashboard');

    await page.getByRole('button', { name: /Open account details/i }).click();

    const accountDialog = page.getByTestId('layout-account-dialog');
    await expect(accountDialog).toBeVisible();
    await expect(accountDialog).toHaveAttribute('aria-modal', 'true');
    await expect(accountDialog.getByText('Ada Lovelace')).toBeVisible();
    await expect(accountDialog.getByText('ada@example.com')).toBeVisible();
    await expect(accountDialog.getByText('Admin')).toBeVisible();
    await expect(accountDialog.getByText('config.read')).toBeVisible();
    await expect(accountDialog.getByText('config.write')).toBeVisible();
    await expect(accountDialog.getByText('users.manage')).toBeVisible();

    await accountDialog.getByRole('button', { name: 'Logout' }).click();

    await expect(page).toHaveURL(/\/login$/);
    await expect(page.getByRole('heading', { name: 'Sign in', exact: true })).toBeVisible();
  });

  test('defaults /config to Global Config and loads raw global YAML', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockCommon(page);

    await page.goto('/config');

    await expect(page.getByRole('heading', { name: 'Global Config', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Global Config Overview' })).toBeVisible();

    await page.getByRole('button', { name: 'Raw YAML' }).click();
    await expect(page.getByRole('heading', { name: 'Raw Global YAML' })).toBeVisible();
    await expect(page.locator('textarea')).toHaveValue(rawGlobalYaml);
  });
});
