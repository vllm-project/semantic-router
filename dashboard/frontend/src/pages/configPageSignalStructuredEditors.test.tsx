import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import {
  SignalConditionsEditor,
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
})
