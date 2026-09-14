import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
import ChatComposerCompletionBudget from './ChatComposerCompletionBudget'

describe('ChatComposerCompletionBudget', () => {
  it('presents a selectable budget that explicitly includes reasoning', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComposerCompletionBudget, { value: 8192, onChange: vi.fn() }),
    )
    expect(markup).toContain('Output budget (tokens, including reasoning)')
    expect(markup).toContain('<option value="8192" selected="">8,192</option>')
    expect(markup).toContain('<option value="16384">16,384</option>')
    expect(markup).not.toContain('disabled=""')
  })

  it('displays and locks an explicit probe budget instead of implying an override', () => {
    const markup = renderToStaticMarkup(
      createElement(ChatComposerCompletionBudget, {
        value: 8192,
        exactValue: 512,
        onChange: vi.fn(),
      }),
    )
    expect(markup).toContain('<option value="512" selected="">512</option>')
    expect(markup).toContain('disabled=""')
    expect(markup).toContain('probe preserves its explicit output budget')
  })
})
