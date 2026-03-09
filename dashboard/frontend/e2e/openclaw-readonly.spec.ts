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
  readonlyMode: true,
  setupMode: false,
  platform: '',
  envoyUrl: '',
};

const openClawTeam = {
  id: 'team-alpha',
  name: 'Team Alpha',
  vibe: 'Calm',
  role: 'Operations',
  principal: 'Safety first',
  leaderId: 'leader-1',
};

const openClawWorkers = [
  {
    name: 'leader-1',
    teamId: 'team-alpha',
    agentName: 'Leader One',
    agentEmoji: '🦞',
    agentRole: 'Lead',
    agentVibe: 'Calm',
    agentPrinciples: 'Coordinate the team',
    roleKind: 'leader',
  },
  {
    name: 'worker-a',
    teamId: 'team-alpha',
    agentName: 'Worker A',
    agentEmoji: '🤖',
    agentRole: 'Operator',
    agentVibe: 'Precise',
    agentPrinciples: 'Do the work',
    roleKind: 'worker',
  },
];

const openClawRoom = {
  id: 'room-alpha',
  teamId: 'team-alpha',
  name: 'Planning',
};

const openClawStatus = [
  {
    running: true,
    containerName: 'worker-a',
    gatewayUrl: 'http://127.0.0.1:18788',
    port: 18788,
    healthy: true,
    error: '',
    teamId: 'team-alpha',
    teamName: 'Team Alpha',
    agentName: 'Worker A',
    agentEmoji: '🤖',
    agentRole: 'Operator',
    agentVibe: 'Precise',
    agentPrinciples: 'Do the work',
    createdAt: '2026-03-09T00:00:00Z',
  },
];

async function mockReadonlyCommon(page: Page) {
  await page.route('**/api/setup/state', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(setupState) });
  });

  await page.route('**/api/settings', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(settingsResponse) });
  });

  await page.route('**/api/mcp/tools', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ tools: [] }) });
  });
}

async function mockReadonlyOpenClaw(page: Page) {
  await mockReadonlyCommon(page);

  await page.route('**/api/openclaw/status', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(openClawStatus) });
  });

  await page.route('**/api/openclaw/teams', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([openClawTeam]) });
  });

  await page.route('**/api/openclaw/workers', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(openClawWorkers) });
  });

  await page.route('**/api/openclaw/rooms?*', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([openClawRoom]) });
  });

  await page.route('**/api/openclaw/rooms/room-alpha/messages?*', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) });
  });

  await page.route('**/api/openclaw/rooms/room-alpha/stream', async route => {
    await route.fulfill({ status: 200, contentType: 'text/event-stream', body: '' });
  });
}

test.describe('Readonly OpenClaw', () => {
  test('normal playground chat omits claw prompt and tools in readonly mode', async ({ page }) => {
    await mockReadonlyCommon(page);

    let capturedBody: Record<string, unknown> | null = null;
    await page.route('**/api/router/v1/chat/completions', async route => {
      capturedBody = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: 'data: {"choices":[{"delta":{"content":"readonly response"}}]}\n\ndata: [DONE]\n\n',
      });
    });

    await page.goto('/playground');
    await page.getByPlaceholder('Ask me anything...').fill('Describe the current status');
    await page.getByTitle('Send message').click();

    await expect(page.getByText('readonly response')).toBeVisible();
    expect(capturedBody).not.toBeNull();

    const messages = Array.isArray(capturedBody?.messages)
      ? (capturedBody?.messages as Array<{ content?: string }>)
      : [];
    expect(messages.some(message => (message.content || '').includes('witty, humorous Claw Manager'))).toBeFalsy();

    const tools = Array.isArray(capturedBody?.tools)
      ? (capturedBody?.tools as Array<{ function?: { name?: string } }>)
      : [];
    const toolNames = tools.map(tool => tool.function?.name || '');
    expect(toolNames.some(name => name.includes('_claw_'))).toBeFalsy();
  });

  test('room view keeps chat enabled but disables room management controls', async ({ page }) => {
    await mockReadonlyOpenClaw(page);

    await page.goto('/playground');
    await page.getByRole('button', { name: 'Open ClawRoom view' }).click();

    await expect(page.getByRole('button', { name: 'New room' })).toBeDisabled();
    await page.getByRole('button', { name: 'Open sidebar' }).click();
    await expect(page.getByPlaceholder('New room name (optional)')).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Delete room', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Set as leader' })).toBeDisabled();

    const roomInput = page.getByPlaceholder('@all to mention everyone, @leader to assign tasks, or @worker-name');
    await expect(roomInput).toBeEnabled();
    await roomInput.fill('hello team');
    await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  });

  test('openclaw page stays browsable but disables management and embedded dashboard entry', async ({ page }) => {
    await mockReadonlyOpenClaw(page);

    await page.goto('/clawos');

    await page.getByRole('button', { name: /Claw Team/ }).click();
    await expect(page.getByRole('button', { name: 'New Team' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Edit' }).first()).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Delete' }).first()).toBeDisabled();

    await page.getByRole('button', { name: /Claw Worker/ }).click();
    await expect(page.getByRole('button', { name: 'New Worker' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Edit' }).first()).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Delete' }).first()).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Status' }).first()).toBeEnabled();

    await page.getByRole('button', { name: /Claw Dashboard/ }).click();
    await expect(page.getByRole('button', { name: 'Dashboard', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Stop' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Remove' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Refresh Status' })).toBeEnabled();
    await expect(page.locator('iframe[title*="OpenClaw Control UI"]')).toHaveCount(0);
  });
});
