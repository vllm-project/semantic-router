import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import {
  SignalConditionsEditor,
  SignalConversationFeatureEditor,
  SignalStructureFeatureEditor,
  SignalSubjectsEditor,
} from './configPageSignalStructuredEditors'

describe('signal structured editors', () => {
  it('renders nested structure features without exposing JSON textareas', () => {
    const markup = renderToStaticMarkup(
      <SignalStructureFeatureEditor
        value={{
          type: 'density',
          source: {
            type: 'keyword_set',
            keywords: ['at least', 'within'],
            case_sensitive: false,
          },
        }}
        onChange={vi.fn()}
        readOnly
      />,
    )

    expect(markup).toContain('density')
    expect(markup).toContain('keyword_set')
    expect(markup).toContain('at least')
    expect(markup).not.toContain('{&quot;type&quot;')
  })

  it('renders condition and subject object arrays as labelled records', () => {
    const conditions = renderToStaticMarkup(
      <SignalConditionsEditor
        value={[{ type: 'domain', name: 'finance' }]}
        onChange={vi.fn()}
        readOnly
      />,
    )
    const subjects = renderToStaticMarkup(
      <SignalSubjectsEditor
        value={[{ kind: 'Group', name: 'admins' }]}
        onChange={vi.fn()}
        readOnly
      />,
    )

    expect(conditions).toContain('domain')
    expect(conditions).toContain('finance')
    expect(subjects).toContain('Group')
    expect(subjects).toContain('admins')
  })

  it('offers every conversation source type, including image_content', () => {
    const markup = renderToStaticMarkup(
      <SignalConversationFeatureEditor
        value={{ type: 'exists', source: { type: 'image_content' } }}
        onChange={vi.fn()}
      />,
    )

    for (const source of [
      'message',
      'tool_definition',
      'assistant_tool_call',
      'assistant_tool_cycle',
      'active_tool_loop',
      'image_content',
    ]) {
      expect(markup).toContain(source)
    }
  })

  it('hides the role select for a non-message source', () => {
    const markup = renderToStaticMarkup(
      <SignalConversationFeatureEditor
        value={{ type: 'exists', source: { type: 'image_content' } }}
        onChange={vi.fn()}
      />,
    )

    expect(markup).not.toContain('Message role')
  })

  it('shows the role select for a message source', () => {
    const markup = renderToStaticMarkup(
      <SignalConversationFeatureEditor
        value={{ type: 'count', source: { type: 'message', role: 'user' } }}
        onChange={vi.fn()}
      />,
    )

    expect(markup).toContain('Message role')
    expect(markup).toContain('non_user')
  })

  it('renders values, not inputs, when readOnly', () => {
    const markup = renderToStaticMarkup(
      <SignalConversationFeatureEditor
        value={{ type: 'count', source: { type: 'message', role: 'user' } }}
        onChange={vi.fn()}
        readOnly
      />,
    )

    expect(markup).not.toContain('<select')
  })
})
