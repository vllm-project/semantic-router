import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { KeyValueEditor } from './KeyValueEditor'
import { ObjectListEditor, type ObjectEditorField } from './ObjectListEditor'
import { StringListEditor } from './StringListEditor'
import { normalizeStringList, updateStructuredObjectField } from './structuredFieldEditorSupport'

describe('string field editors', () => {
  it('normalizes legacy delimited values while preserving list order', () => {
    expect(normalizeStringList(' premium, fast\npremium, long-context ')).toEqual([
      'premium',
      'fast',
      'long-context',
    ])
    expect(normalizeStringList(['analysis', '', 'analysis', 'tools'])).toEqual([
      'analysis',
      'tools',
    ])
  })

  it('renders string arrays as read-only chips without mutation controls', () => {
    const markup = renderToStaticMarkup(
      createElement(StringListEditor, {
        value: ['premium', 'long-context'],
        onChange: vi.fn(),
        itemLabel: 'Tag',
        readOnly: true,
      }),
    )

    expect(markup).toContain('premium')
    expect(markup).toContain('long-context')
    expect(markup).not.toContain('Add value')
    expect(markup).not.toContain('Remove')
  })

  it('renders schema-specific string item validation', () => {
    const markup = renderToStaticMarkup(
      createElement(StringListEditor, {
        value: ['not-a-number'],
        onChange: vi.fn(),
        itemLabel: 'Checkpoint',
        validateItem: (value: string) =>
          Number.isFinite(Number(value)) ? null : 'Checkpoint must be numeric.',
      }),
    )

    expect(markup).toContain('Checkpoint must be numeric.')
    expect(markup).toContain('aria-invalid="true"')
  })
})

describe('structured object field editors', () => {
  it('renders key/value entries as labelled controls', () => {
    const markup = renderToStaticMarkup(
      createElement(KeyValueEditor, {
        value: { 'X-Tenant': 'router-demo' },
        onChange: vi.fn(),
        keyLabel: 'Header',
        valueLabel: 'Value',
      }),
    )

    expect(markup).toContain('X-Tenant')
    expect(markup).toContain('router-demo')
    expect(markup).toContain('Add entry')
    expect(markup).toContain('aria-label="Remove header X-Tenant"')
  })

  it('renders object lists with structured fields and validation feedback', () => {
    type Backend = { endpoint?: string; protocol?: string; headers?: Record<string, string> }
    const fields: ObjectEditorField<Backend>[] = [
      { key: 'endpoint', label: 'Endpoint' },
      { key: 'protocol', label: 'Protocol', type: 'select', options: ['http', 'https'] },
      { key: 'headers', label: 'Headers', type: 'key-value', fullWidth: true },
    ]
    const markup = renderToStaticMarkup(
      createElement(ObjectListEditor<Backend>, {
        value: [{ protocol: 'http', headers: { 'X-Tenant': 'demo' } }],
        onChange: vi.fn(),
        fields,
        createItem: () => ({ protocol: 'http' }),
        validateItem: (item) => (item.endpoint ? [] : ['Provide an endpoint.']),
        itemLabel: () => 'Primary backend',
      }),
    )

    expect(markup).toContain('Primary backend')
    expect(markup).toContain('Provide an endpoint.')
    expect(markup).toContain('X-Tenant')
    expect(markup).toContain('Add item')
  })

  it('preserves empty strings only for presence-aware object fields', () => {
    type Backend = { auth_prefix?: string }
    const presenceAware: ObjectEditorField<Backend> = {
      key: 'auth_prefix',
      label: 'Authentication prefix',
      preserveEmpty: true,
    }
    const ordinary: ObjectEditorField<Backend> = {
      key: 'auth_prefix',
      label: 'Authentication prefix',
    }

    const explicitEmpty = updateStructuredObjectField(
      { auth_prefix: 'Bearer' },
      presenceAware.key,
      '',
      presenceAware.preserveEmpty,
    )
    const omitted = updateStructuredObjectField(
      { auth_prefix: 'Bearer' },
      ordinary.key,
      '',
      ordinary.preserveEmpty,
    )

    expect(explicitEmpty).toEqual({ auth_prefix: '' })
    expect(Object.prototype.hasOwnProperty.call(explicitEmpty, 'auth_prefix')).toBe(true)
    expect(omitted).toEqual({})
  })
})
