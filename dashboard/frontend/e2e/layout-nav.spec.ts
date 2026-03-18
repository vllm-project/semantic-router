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
      domains: [
        { name: 'business', description: 'Business' },
        { name: 'economics', description: 'Economics' },
      ],
      embeddings: [
        {
          name: 'business_analysis',
          threshold: 0.72,
          candidates: ['business'],
          aggregation_method: 'max',
        },
      ],
      preferences: [
        {
          name: 'structured_delivery',
          threshold: 0.55,
          examples: ['Use a structured answer'],
        },
      ],
      complexity: [
        { name: 'general_reasoning:easy', threshold: 0.25 },
        { name: 'general_reasoning:medium', threshold: 0.5 },
      ],
    },
    decisions: [
      {
        name: 'medium_business',
        priority: 215,
        rules: {
          operator: 'AND',
          conditions: [
            {
              operator: 'OR',
              conditions: [
                { type: 'domain', name: 'business' },
                { type: 'domain', name: 'economics' },
              ],
            },
            {
              operator: 'OR',
              conditions: [
                { type: 'embedding', name: 'business_analysis' },
                { type: 'complexity', name: 'general_reasoning:easy' },
                { type: 'complexity', name: 'general_reasoning:medium' },
                { type: 'preference', name: 'structured_delivery' },
              ],
            },
          ],
        },
        modelRefs: [{ model: 'test-model', use_reasoning: true, reasoning_effort: 'medium' }],
      },
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

const replayRecordsResponse = {
  object: 'router_replay.list',
  count: 2,
  total: 2,
  limit: 100,
  offset: 0,
  has_more: false,
  data: [
    {
      id: 'replay-1',
      timestamp: '2026-03-17T10:00:00Z',
      request_id: 'req-1',
      decision: 'business-route',
      original_model: 'test-model',
      selected_model: 'test-model',
      reasoning_mode: 'off',
      selection_method: 'static',
      signals: {
        domain: ['business'],
        keyword: ['finance'],
      },
      response_status: 200,
      from_cache: false,
      streaming: false,
      prompt_tokens: 1000,
      completion_tokens: 500,
      total_tokens: 1500,
      actual_cost: 0.002,
      baseline_cost: 0.008,
      cost_savings: 0.006,
      currency: 'USD',
      baseline_model: 'premium-model',
    },
    {
      id: 'replay-2',
      timestamp: '2026-03-17T10:05:00Z',
      request_id: 'req-2',
      decision: 'business-route',
      original_model: 'test-model',
      selected_model: 'test-model',
      reasoning_mode: 'on',
      selection_method: 'static',
      signals: {
        domain: ['business'],
        preference: ['low-latency'],
      },
      response_status: 200,
      from_cache: true,
      streaming: true,
    },
  ],
};

const replayAggregateResponse = {
  object: 'router_replay.aggregate',
  record_count: 2,
  summary: {
    total_saved: 0.006,
    baseline_spend: 0.008,
    actual_spend: 0.002,
    currency: 'USD',
    cost_record_count: 1,
    excluded_record_count: 1,
  },
  model_selection: [
    { name: 'test-model', value: 2 },
  ],
  decision_distribution: [
    { name: 'business-route', value: 2 },
  ],
  signal_distribution: [
    { name: 'domain', value: 2 },
    { name: 'keyword', value: 1 },
    { name: 'preference', value: 1 },
  ],
  token_volume: {
    input_tokens: 1000,
    output_tokens: 500,
    total_tokens: 1500,
    excluded_record_count: 1,
  },
  token_breakdown: {
    by_decision: [
      {
        name: 'business-route',
        input_tokens: 1000,
        output_tokens: 500,
        total_tokens: 1500,
      },
    ],
    by_selected_model: [
      {
        name: 'test-model',
        input_tokens: 1000,
        output_tokens: 500,
        total_tokens: 1500,
      },
    ],
  },
  available_decisions: ['business-route'],
  available_models: ['test-model'],
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

  await page.route(/\/api\/router\/v1\/router_replay(?:\?.*)?$/, async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(replayRecordsResponse),
    });
  });

  await page.route(/\/api\/router\/v1\/router_replay\/aggregate(?:\?.*)?$/, async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(replayAggregateResponse),
    });
  });

  await page.route('**/api/router/v1/router_replay/*', async route => {
    const requestURL = new URL(route.request().url());
    const replayID = requestURL.pathname.split('/').pop();
    if (replayID === 'aggregate') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(replayAggregateResponse),
      });
      return;
    }
    const record = replayRecordsResponse.data.find(item => item.id === replayID);

    await route.fulfill({
      status: record ? 200 : 404,
      contentType: 'application/json',
      body: JSON.stringify(record ?? { error: { message: 'not found' } }),
    });
  });
}

test.describe('Layout top navigation', () => {
  test('keeps Insight in primary nav and moves Manager beside System on the right', async ({ page }) => {
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
    await expect(primaryGroup.getByRole('link', { name: 'Insight' })).toBeVisible();
    await expect(primaryGroup.getByRole('button', { name: 'Manager' })).toHaveCount(0);
    await expect(primaryGroup.getByRole('link', { name: 'ClawOS' })).toHaveCount(0);
    await expect(primaryGroup.getByRole('link', { name: 'Users' })).toHaveCount(0);

    await expect(secondaryGroup.getByRole('link', { name: 'Users' })).toBeVisible();
    await expect(secondaryGroup.getByRole('link', { name: 'ClawOS' })).toBeVisible();
    await expect(secondaryGroup.getByRole('button', { name: 'Manager' })).toBeVisible();
    await expect(secondaryGroup.getByRole('button', { name: 'System' })).toBeVisible();
    const secondaryButtons = secondaryGroup.getByRole('button');
    await expect(secondaryButtons.nth(0)).toHaveText(/Manager/);
    await expect(secondaryButtons.nth(1)).toHaveText(/System/);
    await expect(globalNav.getByRole('button', { name: 'Analysis', exact: true })).toHaveCount(0);
    await expect(globalNav.getByRole('button', { name: 'Operations', exact: true })).toHaveCount(0);

    await secondaryGroup.getByRole('button', { name: 'System' }).click();

    const menu = page.getByRole('menu', { name: 'System' });
    await expect(menu.getByText('Analysis')).toBeVisible();
    const menuItems = menu.getByRole('menuitem');
    await expect(menuItems.nth(0)).toHaveText('Global Config');
    await expect(menuItems.nth(1)).toHaveText('Evaluation');
    await expect(menuItems.nth(2)).toHaveText('Ratings');
    await expect(menu.getByRole('menuitem', { name: 'Global Config' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Evaluation' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Replay' })).toHaveCount(0);
    await expect(menu.getByRole('menuitem', { name: 'Ratings' })).toBeVisible();
    await expect(menu.getByText('Operations')).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'ML Setup' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'MCP Servers' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Status' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Logs' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Grafana' })).toBeVisible();
    await expect(menu.getByRole('menuitem', { name: 'Tracing' })).toBeVisible();
  });

  test('loads Insights charts and does not preserve the legacy replay route', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockCommon(page);

    await page.goto('/dashboard');
    await page.getByRole('link', { name: 'Insight' }).click();

    await expect(page).toHaveURL(/\/insights$/);
    await expect(page.getByRole('heading', { name: 'Insights', exact: true })).toBeVisible();
    await expect(page.getByText('Total Saved')).toBeVisible();
    await expect(page.getByText('Saved %')).toBeVisible();
    await expect(page.getByText('Baseline Spend')).toBeVisible();
    await expect(page.getByText('Actual Spend')).toBeVisible();
    await expect(page.getByText('See what the router picked, what signals fired, and how much it saved.')).toBeVisible();
    await expect(page.getByText('Decisions, model picks, token usage, and savings in one view.')).toBeVisible();
    await expect(page.getByText('Replay-backed routing records with spend, savings, and token details per request.')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Model Selection', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Decision Distribution', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Signal Distribution', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Token Volume', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Tokens by Decision', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Tokens by Selected Model', exact: true })).toBeVisible();
    await expect(page.getByText('Input Tokens').first()).toBeVisible();
    await expect(page.getByText('Output Tokens').first()).toBeVisible();
    await expect(page.getByText('Total Tokens').first()).toBeVisible();
    await expect(page.getByText('1 filtered record excluded from cost totals because usage or pricing data is incomplete.')).toBeVisible();
    await expect(page.getByText('1 filtered record excluded from token totals because usage data is incomplete.')).toBeVisible();
    await expect(
      page.getByRole('article').filter({ hasText: 'Total Saved' }).getByRole('strong'),
    ).toHaveText('$0.0060');
    await expect(
      page.getByRole('article').filter({ hasText: 'Saved %' }).getByRole('strong'),
    ).toHaveText('75.0%');
    await expect(page.getByRole('columnheader', { name: 'Actual Cost' })).toBeVisible();
    await expect(page.getByRole('columnheader', { name: 'Saved vs Baseline' })).toBeVisible();

    await page.goto('/replay');
    await expect(page).toHaveURL(/\/dashboard$/);
  });

  test('renders nested canonical decision groups in Brain topology', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mockCommon(page);

    await page.goto('/topology');

    const decisionNode = page
      .locator('[class*="decisionNode"]')
      .filter({ has: page.getByText('medium_business', { exact: true }) })
      .first();

    await expect(decisionNode).toBeVisible();
    await expect(decisionNode).toContainText('AND');
    await expect(decisionNode).toContainText('OR: domain: business | domain: economics');
    await expect(decisionNode).toContainText('OR: embedding: business_analysis');
    await expect(decisionNode).toContainText('preference: structured_delivery');
    await expect(decisionNode.getByText('Referenced signals not configured')).toHaveCount(0);
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
