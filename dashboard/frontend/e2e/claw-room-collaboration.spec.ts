import { expect, test, type Page } from '@playwright/test'
import { mockAuthenticatedSession } from './support/auth'

const openClawTeam = {
  id: 'team-alpha',
  name: 'Team Alpha',
  vibe: 'Calm',
  role: 'Operations',
  principal: 'Safety first',
  leaderId: 'leader-1',
}

const openClawWorkers = [
  {
    name: 'leader-1',
    teamId: 'team-alpha',
    agentName: 'Leader One',
    agentRole: 'Lead',
    roleKind: 'leader',
  },
  {
    name: 'worker-a',
    teamId: 'team-alpha',
    agentName: 'Worker A',
    agentRole: 'Operator',
    roleKind: 'worker',
  },
]

const openClawRoom = {
  id: 'room-alpha',
  teamId: 'team-alpha',
  name: 'Planning',
}

const initialMessages = [
  {
    id: 'room-msg-user-1',
    roomId: 'room-alpha',
    teamId: 'team-alpha',
    senderType: 'user',
    senderId: 'playground-user',
    senderName: 'You',
    content: 'hello team',
    createdAt: '2026-05-31T00:00:00Z',
  },
]

async function mockCollaborationBootstrap(page: Page) {
  await mockAuthenticatedSession(page)

  await page.route('**/api/setup/state', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        setupMode: false,
        listenerPort: 8700,
        models: 1,
        decisions: 1,
        hasModels: true,
        hasDecisions: true,
        canActivate: true,
      }),
    })
  })

  await page.route('**/api/settings', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ readonlyMode: false, setupMode: false }),
    })
  })

  await page.route('**/api/openclaw/teams', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([openClawTeam]) })
  })

  await page.route('**/api/openclaw/workers', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(openClawWorkers) })
  })

  await page.route('**/api/openclaw/rooms?*', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([openClawRoom]) })
  })

  await page.route('**/api/openclaw/rooms/room-alpha/messages?*', async route => {
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(initialMessages) })
  })
}

async function enableClawRoom(page: Page, waitForWebSocket = true) {
  await page.goto('/playground')
  await page.getByRole('button', { name: 'Enable HireClaw' }).click()
  await page.getByRole('button', { name: 'Open ClawRoom view' }).click()
  await expect(page.getByTestId('claw-room-transcript')).toBeVisible()
  await expect(page.getByText('hello team')).toBeVisible()
  if (waitForWebSocket) {
    await expect(page.getByTestId('claw-room-transport-status')).toContainText('Live')
  }
}

test.describe('Claw room collaboration', () => {
  test('keeps team details focus contained and returns it to the opener', async ({ page }) => {
    await mockCollaborationBootstrap(page)

    await page.addInitScript(() => {
      class MockWebSocket {
        static OPEN = 1
        static CONNECTING = 0
        readyState = MockWebSocket.OPEN
        onopen: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onerror: ((event: Event) => void) | null = null
        onclose: ((event: CloseEvent) => void) | null = null

        constructor() {
          window.setTimeout(() => this.onopen?.(new Event('open')), 0)
        }

        send() {}
        close() {
          this.readyState = 3
        }
      }

      window.WebSocket = MockWebSocket as unknown as typeof WebSocket
    })

    await enableClawRoom(page)
    await page.getByRole('button', { name: 'Open sidebar' }).click()
    const detailsButton = page.getByTestId('claw-room-team-details-button')
    const overflowBeforeDialog = await page.evaluate(() => document.body.style.overflow)
    await detailsButton.click()

    const dialog = page.getByTestId('claw-room-team-details-dialog')
    const headerCloseButton = dialog.getByRole('button', { name: 'Close team details' })
    const footerCloseButton = dialog.getByRole('button', { name: 'Close', exact: true })
    await expect(dialog).toBeVisible()
    await expect(headerCloseButton).toBeFocused()
    expect(await page.evaluate(() => document.body.style.overflow)).toBe('hidden')

    await page.keyboard.press('Shift+Tab')
    await expect(footerCloseButton).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(headerCloseButton).toBeFocused()

    await page.keyboard.press('Escape')
    await expect(dialog).toBeHidden()
    await expect(detailsButton).toBeFocused()
    expect(await page.evaluate(() => document.body.style.overflow)).toBe(overflowBeforeDialog)

    await detailsButton.click()
    await expect(headerCloseButton).toBeFocused()
    const accountTrigger = page.getByRole('button', { name: /Open account details/i })
    await accountTrigger.evaluate((button: HTMLButtonElement) => button.click())
    const accountDialog = page.getByTestId('layout-account-dialog')
    const accountCloseButton = accountDialog.getByRole('button', { name: 'Close account dialog' })
    await expect(accountCloseButton).toBeFocused()

    await headerCloseButton.evaluate((button: HTMLButtonElement) => button.click())
    await expect(dialog).toBeHidden()
    await expect(accountCloseButton).toBeFocused()
    expect(await page.evaluate(() => document.body.style.overflow)).toBe('hidden')

    await page.keyboard.press('Escape')
    await expect(accountDialog).toBeHidden()
    expect(await page.evaluate(() => document.body.style.overflow)).toBe(overflowBeforeDialog)
  })

  test('renders streaming worker chunks in the transcript', async ({ page }) => {
    await mockCollaborationBootstrap(page)

    await page.addInitScript(() => {
      class MockWebSocket {
        static instances: MockWebSocket[] = []
        url: string
        readyState = 1
        onopen: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onerror: ((event: Event) => void) | null = null
        onclose: ((event: CloseEvent) => void) | null = null

        constructor(url: string) {
          this.url = url
          MockWebSocket.instances.push(this)
          window.setTimeout(() => this.onopen?.(new Event('open')), 0)
        }

        send() {}

        close() {
          this.readyState = 3
        }

        emit(payload: Record<string, unknown>) {
          this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
        }
      }

      ;(window as typeof window & { __mockRoomSockets?: MockWebSocket[] }).__mockRoomSockets = MockWebSocket.instances
      window.WebSocket = MockWebSocket as unknown as typeof WebSocket
    })

    await enableClawRoom(page)

    await page.evaluate(() => {
      const sockets = (window as typeof window & {
        __mockRoomSockets?: Array<{ emit: (payload: Record<string, unknown>) => void }>
      }).__mockRoomSockets
      const socket = sockets?.[sockets.length - 1]
      socket?.emit({
        type: 'message_chunk',
        messageId: 'stream-worker-1',
        chunk: 'Typing ',
        participantType: 'worker',
        participantId: 'worker-a',
      })
      socket?.emit({
        type: 'message_chunk',
        messageId: 'stream-worker-1',
        chunk: 'now',
        participantType: 'worker',
        participantId: 'worker-a',
      })
    })

    await expect(page.locator('[data-room-message-streaming="true"]')).toContainText('Typing now')
  })

  test('renders streaming tool trace cards from websocket events', async ({ page }) => {
    await mockCollaborationBootstrap(page)

    await page.addInitScript(() => {
      class MockWebSocket {
        static instances: MockWebSocket[] = []
        static OPEN = 1
        static CONNECTING = 0
        readyState = MockWebSocket.OPEN
        onopen: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onerror: ((event: Event) => void) | null = null
        onclose: ((event: CloseEvent) => void) | null = null

        constructor() {
          MockWebSocket.instances.push(this)
          window.setTimeout(() => this.onopen?.(new Event('open')), 0)
        }

        send() {}
        close() { this.readyState = 3 }
        emit(payload: Record<string, unknown>) {
          this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
        }
      }

      ;(window as typeof window & { __mockRoomSockets?: MockWebSocket[] }).__mockRoomSockets = MockWebSocket.instances
      window.WebSocket = MockWebSocket as unknown as typeof WebSocket
    })
    await enableClawRoom(page)

    await page.evaluate(() => {
      const sockets = (window as typeof window & {
        __mockRoomSockets?: Array<{ emit: (payload: Record<string, unknown>) => void }>
      }).__mockRoomSockets
      const socket = sockets?.[sockets.length - 1]
      socket?.emit({
        type: 'message_chunk',
        messageId: 'stream-worker-tools',
        chunk: 'Done.',
        participantType: 'worker',
        participantId: 'worker-a',
      })
      socket?.emit({
        type: 'tool_trace_update',
        messageId: 'stream-worker-tools',
        payload: {
          revision: 1,
          steps: [
            { id: 'call_1', name: 'exec', arguments: '{"command":"pwd"}', status: 'running' },
          ],
        },
      })
      socket?.emit({
        type: 'tool_trace_update',
        messageId: 'stream-worker-tools',
        payload: {
          revision: 2,
          steps: [
            {
              id: 'call_1',
              name: 'exec',
              arguments: '{"command":"pwd"}',
              status: 'completed',
              result: '/workspace',
            },
          ],
        },
      })
    })

    await expect(page.getByTestId('claw-room-tool-trace')).toBeVisible()
    await expect(page.getByTestId('claw-room-tool-trace')).toContainText('exec')
    await page.getByRole('button', { name: 'Expand details for exec' }).click()
    await expect(page.getByTestId('claw-room-tool-trace')).toContainText('/workspace')

    await page.evaluate(() => {
      const sockets = (window as typeof window & {
        __mockRoomSockets?: Array<{ emit: (payload: Record<string, unknown>) => void }>
      }).__mockRoomSockets
      const socket = sockets?.[sockets.length - 1]
      socket?.emit({
        type: 'message_updated',
        message: {
          id: 'stream-worker-tools',
          roomId: 'room-alpha',
          teamId: 'team-alpha',
          senderType: 'worker',
          senderName: 'worker-a',
          content: 'Done.',
          createdAt: '2026-05-31T00:00:00Z',
          metadata: {
            toolTrace: JSON.stringify([{
              id: 'call_1',
              name: 'exec',
              arguments: '{"command":"pwd"}',
              status: 'completed',
              result: '/workspace',
            }]),
          },
        },
      })
    })

    await expect(page.getByTestId('claw-room-tool-trace')).toBeVisible()
    await expect(page.locator('[data-room-message-streaming="true"]')).toHaveCount(0)
    await expect(page.locator('[data-room-message-id="stream-worker-tools"]')).toContainText('Done.')
  })

  test('falls back to SSE when websocket disconnects', async ({ page }) => {
    await mockCollaborationBootstrap(page)

    await page.addInitScript(() => {
      class FailingWebSocket {
        readyState = 3
        onopen: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onerror: ((event: Event) => void) | null = null
        onclose: ((event: CloseEvent) => void) | null = null

        constructor() {
          window.setTimeout(() => this.onclose?.({ code: 1006 } as CloseEvent), 0)
        }

        send() {}
        close() {}
      }

      class MockEventSource {
        static latest: MockEventSource | null = null
        url: string
        onopen: ((event: Event) => void) | null = null
        onerror: ((event: Event) => void) | null = null
        messageListener: EventListener | null = null

        constructor(url: string) {
          this.url = url
          MockEventSource.latest = this
          window.setTimeout(() => this.onopen?.(new Event('open')), 0)
        }

        addEventListener(type: string, listener: EventListener) {
          if (type !== 'message') {
            return
          }
          this.messageListener = listener
        }

        close() {}

        emitMessage(payload: Record<string, unknown>) {
          this.messageListener?.({ data: JSON.stringify(payload) } as MessageEvent)
        }
      }

      window.WebSocket = FailingWebSocket as unknown as typeof WebSocket
      window.EventSource = MockEventSource as unknown as typeof EventSource
      ;(window as typeof window & { __mockEventSource?: typeof MockEventSource }).__mockEventSource = MockEventSource
    })

    await enableClawRoom(page, false)
    await expect(page.getByTestId('claw-room-transport-status')).toContainText('SSE')

    await page.evaluate(() => {
      const MockEventSource = (window as typeof window & {
        __mockEventSource?: { latest?: { emitMessage: (payload: Record<string, unknown>) => void } }
      }).__mockEventSource
      MockEventSource?.latest?.emitMessage({
        type: 'message',
        roomId: 'room-alpha',
        message: {
          id: 'room-msg-sse-1',
          roomId: 'room-alpha',
          teamId: 'team-alpha',
          senderType: 'worker',
          senderId: 'worker-a',
          senderName: 'worker-a',
          content: 'SSE fallback delivery',
          createdAt: '2026-05-31T00:01:00Z',
        },
      })
    })

    await expect(page.getByText('SSE fallback delivery')).toBeVisible()
  })

  test('forwards room events to embedded iframe via bridge', async ({ page }) => {
    await mockCollaborationBootstrap(page)

    await page.addInitScript(() => {
      class MockWebSocket {
        static instances: MockWebSocket[] = []
        static OPEN = 1
        static CONNECTING = 0
        sent: string[] = []
        readyState = MockWebSocket.OPEN
        onopen: ((event: Event) => void) | null = null
        onmessage: ((event: MessageEvent) => void) | null = null
        onerror: ((event: Event) => void) | null = null
        onclose: ((event: CloseEvent) => void) | null = null

        constructor(_url: string) {
          MockWebSocket.instances.push(this)
          window.setTimeout(() => this.onopen?.(new Event('open')), 0)
        }

        send(data: string) {
          this.sent.push(data)
        }

        close() {}

        emit(payload: Record<string, unknown>) {
          this.onmessage?.({ data: JSON.stringify(payload) } as MessageEvent)
        }
      }

      ;(window as typeof window & { __mockRoomSockets?: MockWebSocket[] }).__mockRoomSockets = MockWebSocket.instances
      window.WebSocket = MockWebSocket as unknown as typeof WebSocket
    })

    await page.route('**/api/openclaw/status', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
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
          },
        ]),
      })
    })

    await page.route('**/api/openclaw/token?name=*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ token: 'gateway-token' }),
      })
    })

    await page.route('**/embedded/openclaw/**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'text/html',
        body: `<!doctype html><html><body>
          <script>
            window.__bridgeEvents = [];
            window.addEventListener('message', (event) => {
              if (event.data && event.data.source === 'clawos-room-bridge') {
                window.__bridgeEvents.push(event.data);
              }
            });
          </script>
        </body></html>`,
      })
    })

    await page.goto('/clawos')
    await page.getByRole('button', { name: /Claw Dashboard/ }).click()
    await page.getByRole('button', { name: 'Dashboard', exact: true }).click()

    await expect(page.getByTestId('claw-room-bridge-activity')).toBeVisible()

    await page.evaluate(() => {
      const sockets = (window as typeof window & {
        __mockRoomSockets?: Array<{ emit: (payload: Record<string, unknown>) => void }>
      }).__mockRoomSockets
      const socket = sockets?.[sockets.length - 1]
      socket?.emit({
        type: 'new_message',
        message: {
          id: 'room-msg-bridge-1',
          roomId: 'room-alpha',
          teamId: 'team-alpha',
          senderType: 'worker',
          senderId: 'worker-a',
          senderName: 'worker-a',
          content: 'bridge hello',
          createdAt: '2026-05-31T00:02:00Z',
        },
      })
    })

    const iframeEvents = await page.evaluate(async () => {
      const iframe = document.querySelector('iframe[title*="OpenClaw Control UI"]') as HTMLIFrameElement | null
      if (!iframe?.contentWindow) {
        return [] as unknown[]
      }

      for (let attempt = 0; attempt < 20; attempt += 1) {
        const events = (iframe.contentWindow as Window & { __bridgeEvents?: unknown[] }).__bridgeEvents
        if (Array.isArray(events) && events.length > 0) {
          return events
        }
        await new Promise(resolve => window.setTimeout(resolve, 100))
      }
      return [] as unknown[]
    })

    expect(iframeEvents.length).toBeGreaterThan(0)

    await page.evaluate(() => {
      window.dispatchEvent(new MessageEvent('message', {
        data: {
          source: 'clawos-room-bridge',
          type: 'surface_event',
          roomId: 'room-alpha',
          payload: { kind: 'tool_call', name: 'search' },
        },
      }))
    })

    await expect(page.getByTestId('claw-room-bridge-activity')).toContainText('Surface')

    const surfaceWrites = await page.evaluate(() => {
      const sockets = (window as typeof window & {
        __mockRoomSockets?: Array<{ sent: string[] }>
      }).__mockRoomSockets
      return sockets?.flatMap(socket => socket.sent) || []
    })
    expect(surfaceWrites.some(entry => entry.includes('surface_event') && entry.includes('tool_call'))).toBeTruthy()
  })
})
