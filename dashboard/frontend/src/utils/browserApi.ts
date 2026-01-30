/**
 * Browser API - Functions for browser automation (computer-use agent)
 */

// Action types for browser control
export type BrowserActionType =
  | 'navigate'
  | 'click'
  | 'type'
  | 'scroll'
  | 'screenshot'
  | 'wait'
  | 'back'
  | 'forward'
  | 'refresh'
  | 'key'

// Browser action request
export interface BrowserAction {
  type: BrowserActionType
  url?: string       // For navigate
  selector?: string  // For click, type
  text?: string      // For type
  x?: number         // For click (coordinates)
  y?: number         // For click (coordinates)
  delta_x?: number   // For scroll
  delta_y?: number   // For scroll
  duration?: number  // For wait (ms)
  key?: string       // For key press (Enter, Tab, Escape, etc.)
}

// Result from a browser action
export interface BrowserActionResult {
  success: boolean
  screenshot?: string  // Base64 encoded PNG
  url?: string
  title?: string
  error?: string
  width?: number
  height?: number
}

// Session info
export interface BrowserSession {
  session_id: string
  success: boolean
  error?: string
}

const API_BASE = ''

/**
 * Start a new browser session
 */
export async function startBrowserSession(headless = true): Promise<BrowserSession> {
  const response = await fetch(`${API_BASE}/api/browser/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ headless }),
  })

  if (!response.ok) {
    throw new Error(`Failed to start browser: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Stop a browser session
 */
export async function stopBrowserSession(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/browser/${sessionId}`, {
    method: 'DELETE',
  })

  if (!response.ok) {
    throw new Error(`Failed to stop browser: ${response.statusText}`)
  }
}

/**
 * Execute a browser action
 */
export async function executeBrowserAction(
  sessionId: string,
  action: BrowserAction
): Promise<BrowserActionResult> {
  const response = await fetch(`${API_BASE}/api/browser/${sessionId}/action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(action),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Action failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Navigate to a URL
 */
export async function navigateTo(
  sessionId: string,
  url: string
): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'navigate', url })
}

/**
 * Click at coordinates or selector
 */
export async function click(
  sessionId: string,
  options: { selector?: string; x?: number; y?: number }
): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'click', ...options })
}

/**
 * Type text into an element
 */
export async function typeText(
  sessionId: string,
  selector: string,
  text: string
): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'type', selector, text })
}

/**
 * Scroll the page
 */
export async function scroll(
  sessionId: string,
  deltaX: number,
  deltaY: number
): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'scroll', delta_x: deltaX, delta_y: deltaY })
}

/**
 * Take a screenshot
 */
export async function takeScreenshot(sessionId: string): Promise<BrowserActionResult> {
  const response = await fetch(`${API_BASE}/api/browser/${sessionId}/screenshot`, {
    method: 'GET',
  })

  if (!response.ok) {
    throw new Error(`Screenshot failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Wait for a duration
 */
export async function wait(
  sessionId: string,
  durationMs: number
): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'wait', duration: durationMs })
}

/**
 * Go back in history
 */
export async function goBack(sessionId: string): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'back' })
}

/**
 * Go forward in history
 */
export async function goForward(sessionId: string): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'forward' })
}

/**
 * Refresh the page
 */
export async function refresh(sessionId: string): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'refresh' })
}

/**
 * Press a keyboard key
 */
export async function pressKey(
  sessionId: string,
  key: string
): Promise<BrowserActionResult> {
  return executeBrowserAction(sessionId, { type: 'key', key })
}

/**
 * Computer-use tool definition for LLM
 */
export const computerUseToolDefinition = {
  type: 'function' as const,
  function: {
    name: 'computer_use',
    description: `Control a web browser to perform actions. You can navigate to URLs, click elements, type text, press keyboard keys, scroll, and take screenshots. Always start by navigating to a website, then use screenshots to see the page and decide on next actions.`,
    parameters: {
      type: 'object',
      properties: {
        action: {
          type: 'string',
          enum: ['navigate', 'click', 'type', 'key', 'scroll', 'screenshot', 'wait', 'back', 'forward', 'refresh'],
          description: 'The browser action to perform',
        },
        url: {
          type: 'string',
          description: 'URL to navigate to (for navigate action)',
        },
        selector: {
          type: 'string',
          description: 'CSS selector to interact with (for click, type actions)',
        },
        text: {
          type: 'string',
          description: 'Text to type (for type action)',
        },
        key: {
          type: 'string',
          description: 'Key to press (for key action). Examples: Enter, Tab, Escape, Backspace, ArrowUp, ArrowDown',
        },
        x: {
          type: 'number',
          description: 'X coordinate for click (alternative to selector)',
        },
        y: {
          type: 'number',
          description: 'Y coordinate for click (alternative to selector)',
        },
        delta_x: {
          type: 'number',
          description: 'Horizontal scroll amount in pixels',
        },
        delta_y: {
          type: 'number',
          description: 'Vertical scroll amount in pixels (positive = down)',
        },
        duration: {
          type: 'number',
          description: 'Wait duration in milliseconds (for wait action)',
        },
      },
      required: ['action'],
    },
  },
}
