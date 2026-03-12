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

const openClawMessages = [
  {
    id: 'room-msg-user-1',
    roomId: 'room-alpha',
    teamId: 'team-alpha',
    senderType: 'user',
    senderId: 'playground-user',
    senderName: 'You',
    content: '@leader Can you summarize the latest blocker?',
    createdAt: '2026-03-09T00:01:00Z',
  },
  {
    id: 'room-msg-leader-1',
    roomId: 'room-alpha',
    teamId: 'team-alpha',
    senderType: 'leader',
    senderId: 'leader-1',
    senderName: 'leader-1',
    content: 'The blocker is the staging rollout. I am assigning Worker A to verify the queue metrics now.',
    createdAt: '2026-03-09T00:01:12Z',
  },
  {
    id: 'room-msg-worker-1',
    roomId: 'room-alpha',
    teamId: 'team-alpha',
    senderType: 'worker',
    senderId: 'worker-a',
    senderName: 'worker-a',
    content: 'Queue pressure is normal. The issue is isolated to the deploy hook.',
    createdAt: '2026-03-09T00:01:25Z',
  },
];

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
  await mockAuthenticatedSession(page, {
    settings: settingsResponse,
  });

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
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(openClawMessages) });
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
    expect(messages.some(message => (message.content || '').includes('imaginative, sharply observant Claw Manager'))).toBeFalsy();

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
    const sidebar = page.getByTestId('claw-room-sidebar');
    await expect(sidebar).toBeVisible();
    await expect(sidebar.getByLabel('Team')).toHaveValue('team-alpha');
    const teamDetailsButton = page.getByTestId('claw-room-team-details-button');
    await expect(teamDetailsButton).toBeVisible();
    await expect(sidebar).not.toContainText('Collaboration tip:');
    await expect(sidebar).not.toContainText('Mention hints:');
    await expect(page.getByPlaceholder('New room name (optional)')).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Delete room', exact: true })).toBeDisabled();

    await teamDetailsButton.click();

    const dialog = page.getByTestId('claw-room-team-details-dialog');
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText('Team Alpha');
    await expect(dialog).toContainText('Leader One');
    await expect(dialog).toContainText('Worker A');
    await expect(dialog.getByRole('button', { name: 'Set as leader' })).toHaveCount(0);
    await expect(dialog).not.toContainText('@leader-1');
    await expect(dialog).not.toContainText('Coordinate the team');

    const dialogBox = await dialog.boundingBox();
    const viewport = page.viewportSize();

    expect(dialogBox).not.toBeNull();
    expect(viewport).not.toBeNull();

    const dialogCenterX = dialogBox!.x + dialogBox!.width / 2;
    const dialogCenterY = dialogBox!.y + dialogBox!.height / 2;

    expect(Math.abs(dialogCenterX - viewport!.width / 2)).toBeLessThan(100);
    expect(Math.abs(dialogCenterY - viewport!.height / 2)).toBeLessThan(100);

    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await expect(dialog).toHaveCount(0);

    const roomInput = page.getByPlaceholder('@all to mention everyone, @leader to assign tasks, or @worker-name');
    await expect(roomInput).toBeEnabled();
    await roomInput.fill('hello team');
    await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled();
  });

  test('room view keeps a centered transcript rail with claws left and the user right', async ({ page }) => {
    await mockReadonlyOpenClaw(page);

    await page.goto('/playground');
    await page.getByRole('button', { name: 'Open ClawRoom view' }).click();

    const header = page.getByTestId('claw-room-header');
    const transcript = page.getByTestId('claw-room-transcript');
    await expect(header).toBeVisible();
    await expect(transcript).toBeVisible();
    await expect(page.getByText('The blocker is the staging rollout. I am assigning Worker A to verify the queue metrics now.')).toBeVisible();

    const headerBackground = await header.evaluate(element => window.getComputedStyle(element).backgroundColor);
    const headerBorderBottomWidth = await header.evaluate(element => window.getComputedStyle(element).borderBottomWidth);

    expect(headerBackground).toBe('rgba(0, 0, 0, 0)');
    expect(headerBorderBottomWidth).toBe('0px');
    await expect(header).not.toContainText('Room ·');

    const userMessage = transcript.locator('[data-room-message-role="user"] [data-room-message-content]').first();
    const leaderMessage = transcript.locator('[data-room-message-role="leader"] [data-room-message-content]').first();
    const leaderLogo = transcript.locator('[data-room-message-role="leader"] [data-room-sender-logo]').first();

    await expect(userMessage).toBeVisible();
    await expect(leaderMessage).toBeVisible();
    await expect(leaderLogo).toBeVisible();
    await expect(transcript.locator('[data-room-message-role="user"] [data-room-sender-logo]')).toHaveCount(0);
    await expect(transcript.locator('img[alt*="avatar"]')).toHaveCount(0);

    const leaderLogoWidth = await leaderLogo.evaluate(element => window.getComputedStyle(element).width);
    const leaderLogoHeight = await leaderLogo.evaluate(element => window.getComputedStyle(element).height);

    expect(leaderLogoWidth).toBe('16px');
    expect(leaderLogoHeight).toBe('16px');

    const transcriptBox = await transcript.boundingBox();
    const userBox = await userMessage.boundingBox();
    const leaderBox = await leaderMessage.boundingBox();

    expect(transcriptBox).not.toBeNull();
    expect(userBox).not.toBeNull();
    expect(leaderBox).not.toBeNull();

    expect(leaderBox!.x - transcriptBox!.x).toBeLessThan(transcriptBox!.width * 0.3);
    expect(userBox!.x).toBeGreaterThan(leaderBox!.x + 80);
    expect(userBox!.width).toBeLessThan(transcriptBox!.width * 0.8);
  });

  test('room view opens the lower-left account dialog over the full page', async ({ page }) => {
    await mockReadonlyOpenClaw(page);

    await page.goto('/playground');
    await page.getByRole('button', { name: 'Open ClawRoom view' }).click();

    const accountButton = page.getByTestId('playground-account-control');
    await expect(accountButton).toBeVisible();
    await accountButton.click();

    const overlay = page.getByTestId('layout-account-overlay');
    const dialog = page.getByTestId('layout-account-dialog');

    await expect(overlay).toBeVisible();
    await expect(dialog).toBeVisible();

    const overlayBox = await overlay.boundingBox();
    const dialogBox = await dialog.boundingBox();
    const viewport = page.viewportSize();

    expect(overlayBox).not.toBeNull();
    expect(dialogBox).not.toBeNull();
    expect(viewport).not.toBeNull();

    expect(Math.abs(overlayBox!.x)).toBeLessThan(2);
    expect(Math.abs(overlayBox!.y)).toBeLessThan(2);
    expect(Math.abs(overlayBox!.width - viewport!.width)).toBeLessThan(4);
    expect(Math.abs(overlayBox!.height - viewport!.height)).toBeLessThan(4);

    const dialogCenterX = dialogBox!.x + dialogBox!.width / 2;
    const dialogCenterY = dialogBox!.y + dialogBox!.height / 2;

    expect(Math.abs(dialogCenterX - viewport!.width / 2)).toBeLessThan(100);
    expect(Math.abs(dialogCenterY - viewport!.height / 2)).toBeLessThan(120);
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
