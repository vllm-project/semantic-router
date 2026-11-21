import { renderToStaticMarkup } from 'react-dom/server'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'

import { TooltipProvider } from '../ui'
import type { Message } from './ChatComponentTypes'
import PlaygroundInspector from './PlaygroundInspector'
import { PLAYGROUND_LATENCY_MS, PLAYGROUND_TTFT_MS } from './playgroundTimingHeaders'
import { MATCHED_SIGNAL_KEY_PREFIX, traceKey } from './routingMetadataDisplay'

const render = (message: Message | null) =>
  renderToStaticMarkup(
    <MemoryRouter>
      <TooltipProvider>
        <PlaygroundInspector
          open
          overlay={false}
          message={message}
          prompt="Why does my effect run twice?"
          insightsBasePath="/observe/requests"
          onClose={() => {}}
        />
      </TooltipProvider>
    </MemoryRouter>,
  )

const routed: Message = {
  id: 'answer-1',
  role: 'assistant',
  content: 'Routed answer.',
  timestamp: new Date(0),
  headers: {
    'x-vsr-selected-model': 'qwen/qwen3-32b',
    'x-vsr-selected-decision': 'computer_science_route',
    'x-vsr-replay-id': 'replay-123',
    [`${MATCHED_SIGNAL_KEY_PREFIX}domain`]: 'computer science',
    [PLAYGROUND_LATENCY_MS]: '28280',
    [PLAYGROUND_TTFT_MS]: '1320',
    [traceKey('effective_input_tokens')]: '412',
  },
}

describe('PlaygroundInspector', () => {
  it('gives each part of the trace a named section', () => {
    const markup = render(routed)

    for (const title of ['Signals', 'Decision', 'Model selection', 'Timing and usage']) {
      expect(markup).toMatch(
        new RegExp(
          `<section aria-labelledby="([^"]+)"[^>]*>.*?<h3[^>]*id="\\1"[^>]*>${title}</h3>`,
        ),
      )
    }
    expect(markup).toContain('1 matched')
  })

  it('lists timings and usage as terms and right-aligned values in one card', () => {
    const markup = render(routed)

    expect(markup).toMatch(/>Latency<\/dt><dd[^>]*>28\.28 s<\/dd>/)
    expect(markup).toMatch(/>First token<\/dt><dd[^>]*>1\.32 s<\/dd>/)
    expect(markup).toMatch(/>Input tokens<\/dt><dd[^>]*>412<\/dd>/)
    expect(markup).not.toContain('grid grid-cols-2 gap-2')
  })

  it('keeps the Requests link in the pane footer, outside the scrolling trace', () => {
    const markup = render(routed)
    const footer = markup.slice(markup.lastIndexOf('<div class="shrink-0 border-t'))

    expect(footer).toMatch(/href="\/observe\/requests\/replay-123"[^>]*>Open in Requests/)
  })

  it('shows a quiet note rather than a decorated empty state before any answer', () => {
    const markup = render(null)

    expect(markup).toContain('Nothing to inspect yet')
    expect(markup).not.toContain('Open in Requests')
    expect(markup).not.toContain('absolute rounded-full border border-border')
  })
})
