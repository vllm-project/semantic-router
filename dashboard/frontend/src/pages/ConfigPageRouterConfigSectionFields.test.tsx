import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { TooltipProvider } from '../ui'
import { PathField } from './ConfigPageRouterConfigSectionFields'
import {
  RouterSettingsEditorContext,
  type RouterSettingsEditor,
} from './configPageRouterSettingsEditor'
import { routerSettingsSections } from './configPageRouterSettingsModel'

const core = routerSettingsSections().find((section) => section.key === 'router_core')
const field = (key: string) => {
  const found = core?.fields.find((entry) => entry.key === key)
  if (!found) throw new Error(`router core has no ${key}`)
  return found
}

function render(key: string, editor: Partial<RouterSettingsEditor>) {
  const value: RouterSettingsEditor = {
    draft: {},
    saved: {},
    effective: null,
    readOnly: false,
    errors: new Map(),
    query: '',
    show: 'all',
    set: vi.fn(),
    unset: vi.fn(),
    restore: vi.fn(),
    applyPatch: vi.fn(),
    reportInvalid: vi.fn(),
    ...editor,
  }
  const setting = field(key)
  return renderToStaticMarkup(
    createElement(
      TooltipProvider,
      null,
      createElement(
        RouterSettingsEditorContext.Provider,
        { value },
        createElement(PathField, {
          schema: setting.schema,
          path: setting.path,
          keyName: setting.key,
        }),
      ),
    ),
  )
}

describe('router settings fields', () => {
  it('shows an inherited default as a muted placeholder', () => {
    const markup = render('drain_timeout_seconds', {
      effective: { router: { drain_timeout_seconds: 25 } },
    })
    expect(markup).toContain('inherited')
    expect(markup).toContain('placeholder="25"')
    expect(markup).toContain('value=""')
    expect(markup).not.toContain('aria-label="Edited"')
  })

  it('marks an edited value and offers undo', () => {
    const markup = render('drain_timeout_seconds', {
      draft: { router: { drain_timeout_seconds: 40 } },
      saved: { router: { drain_timeout_seconds: 25 } },
    })
    expect(markup).toContain('value="40"')
    expect(markup).toContain('aria-label="Edited"')
    expect(markup).toContain('aria-label="Undo Drain timeout"')
    expect(markup).not.toContain('inherited')
  })

  it('offers a reset to the default for a value config.yaml sets', () => {
    const markup = render('drain_timeout_seconds', {
      draft: { router: { drain_timeout_seconds: 25 } },
      saved: { router: { drain_timeout_seconds: 25 } },
    })
    expect(markup).toContain('aria-label="Reset Drain timeout to its default"')
  })

  it('shows the validation error instead of the hint', () => {
    const markup = render('drain_timeout_seconds', {
      draft: { router: { drain_timeout_seconds: '4o' } },
      errors: new Map([
        ['router.drain_timeout_seconds', 'Enter a number, or clear the field to use the default.'],
      ]),
    })
    expect(markup).toContain('aria-invalid="true"')
    expect(markup).toContain('Enter a number, or clear the field to use the default.')
  })

  it('disables every control in read-only mode', () => {
    const markup = render('strategy', {
      readOnly: true,
      effective: { router: { strategy: 'priority' } },
    })
    expect(markup).toContain('disabled=""')
    expect(markup).not.toContain('Reset Default decision strategy')
  })

  it('hides fields the search does not match', () => {
    expect(render('drain_timeout_seconds', { query: 'tokenizer' })).toBe('')
    expect(render('context_window', { query: 'tokenizer' })).toContain('Context window')
  })
})
