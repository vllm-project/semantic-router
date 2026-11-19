import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ChatComponentErrors from './ChatComponentErrors'
import type { PlaygroundErrorPresentation } from './playgroundErrorPresentation'

const renderErrors = (
  routingModelStatus: 'discovering' | 'ready' | 'error',
  visibleError: PlaygroundErrorPresentation | null,
  overlay = false,
) =>
  renderToStaticMarkup(
    createElement(ChatComponentErrors, {
      onDismissError: vi.fn(),
      onRetryRoutingModelDiscovery: vi.fn(),
      overlay,
      routingModelStatus,
      visibleError,
    }),
  )

describe('ChatComponentErrors', () => {
  it('renders routing discovery failures in the dedicated top error region', () => {
    const markup = renderErrors('error', null)

    expect(markup).toContain('data-testid="playground-error-region"')
    expect(markup).toContain('role="alert"')
    expect(markup).toContain('The automatic routing model is unavailable.')
    expect(markup).toContain('Retry discovery')
    expect(markup).not.toContain('Dismiss error')
  })

  it('renders conversation failures as dismissible alerts in the same region', () => {
    const markup = renderErrors('ready', { message: 'The request could not be completed.' })

    expect(markup).toContain('data-testid="playground-error-region"')
    expect(markup).toContain('role="alert"')
    expect(markup).toContain('The request could not be completed.')
    expect(markup).toContain('aria-label="Dismiss error"')
    expect(markup).not.toContain('Retry discovery')
  })

  it('keeps raw service responses in closed technical details', () => {
    const rawResponse = 'upstream=http://private.internal stack=worker-secret'
    const markup = renderErrors('ready', {
      message: 'The model service is temporarily unavailable. Try again.',
      technicalDetails: rawResponse,
    })

    expect(markup).toContain('The model service is temporarily unavailable. Try again.')
    expect(markup).toContain('data-playground-technical-details="true"')
    expect(markup).toContain('<summary>Technical details</summary>')
    expect(markup).toContain(rawResponse)
    expect(markup).not.toContain('<details open')
    expect(markup.indexOf(rawResponse)).toBeGreaterThan(
      markup.indexOf('<summary>Technical details'),
    )
  })

  it('stays absent when the playground has no error', () => {
    expect(renderErrors('ready', null)).toBe('')
  })

  it('anchors the error region to the top of the chat content', () => {
    const chat = readFileSync(new URL('./ChatComponent.tsx', import.meta.url), 'utf8')
    const regionClass = (markup: string) =>
      markup.match(/<div class="([^"]+)" data-testid="playground-error-region"/)?.[1] ?? ''

    // The region's parent is the chat body below the header, positioned for the overlay.
    expect(chat).toMatch(
      /className="relative flex min-h-0 flex-1 flex-col">\s*<ChatComponentErrors/,
    )
    expect(regionClass(renderErrors('error', null))).toMatch(
      /\brelative\b.*\bw-full\b|\bw-full\b.*\brelative\b/,
    )
    expect(regionClass(renderErrors('error', null, true))).toMatch(/\babsolute\b/)
    expect(regionClass(renderErrors('error', null, true))).toMatch(/\binset-x-0\b/)
    expect(regionClass(renderErrors('error', null, true))).toMatch(/\btop-0\b/)
    expect(renderErrors('error', null)).toContain('max-w-3xl')
  })
})
