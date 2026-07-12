import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import {
  getAlgorithmFieldSchema,
  getPluginFieldSchema,
  getSignalFieldSchema,
  type FieldSchema,
} from '@/lib/dslMutations'

import { FieldEditor } from './builderPageFieldControls'

function requireField(schema: FieldSchema[], key: string): FieldSchema {
  const field = schema.find((candidate) => candidate.key === key)
  if (!field) throw new Error(`Missing schema field ${key}`)
  return field
}

describe('Builder structured field controls', () => {
  it('renders workflow roles as nested controls instead of JSON text', () => {
    const roles = requireField(getAlgorithmFieldSchema('workflows'), 'roles')
    const markup = renderToStaticMarkup(
      createElement(FieldEditor, {
        schema: roles,
        value: [
          {
            name: 'worker',
            models: ['qwen-worker'],
            access_list: ['reviewer'],
          },
        ],
        onChange: vi.fn(),
      }),
    )

    expect(markup).toContain('Role: worker')
    expect(markup).toContain('qwen-worker')
    expect(markup).toContain('Accessible Roles')
    expect(markup).toContain('Add role')
    expect(markup).not.toContain('<textarea')
  })

  it('renders fixed nested objects and string matrices with typed inputs', () => {
    const weights = requireField(getAlgorithmFieldSchema('multi_factor'), 'weights')
    const weightsMarkup = renderToStaticMarkup(
      createElement(FieldEditor, {
        schema: weights,
        value: { quality: 0.4, latency: 0.3, cost: 0.2, load: 0.1 },
        onChange: vi.fn(),
      }),
    )
    expect(weightsMarkup).toContain('Quality')
    expect(weightsMarkup).toContain('value="0.4"')
    expect(weightsMarkup).not.toContain('<textarea')

    const structureFeature = requireField(getSignalFieldSchema('structure'), 'feature')
    const featureMarkup = renderToStaticMarkup(
      createElement(FieldEditor, {
        schema: structureFeature,
        value: {
          type: 'sequence',
          source: { type: 'sequence', sequences: [['first', 'then']] },
        },
        onChange: vi.fn(),
      }),
    )
    expect(featureMarkup).toContain('Sequence 1')
    expect(featureMarkup).toContain('first')
    expect(featureMarkup).toContain('then')
    expect(featureMarkup).not.toContain('<textarea')
  })

  it('uses the shared object-list editor for flat header pairs', () => {
    const addHeaders = requireField(getPluginFieldSchema('header_mutation'), 'add')
    const markup = renderToStaticMarkup(
      createElement(FieldEditor, {
        schema: addHeaders,
        value: [{ name: 'X-Tenant', value: 'premium' }],
        onChange: vi.fn(),
      }),
    )

    expect(markup).toContain('Header: X-Tenant')
    expect(markup).toContain('Header Name')
    expect(markup).toContain('premium')
    expect(markup).not.toContain('<textarea')
  })

  it('renders recursive composer rules without falling back to JSON', () => {
    const composer = requireField(getSignalFieldSchema('complexity'), 'composer')
    const markup = renderToStaticMarkup(
      createElement(FieldEditor, {
        schema: composer,
        value: {
          operator: 'OR',
          conditions: [
            { type: 'domain', name: 'math' },
            { type: 'preference', name: 'deep_reasoning' },
          ],
        },
        onChange: vi.fn(),
      }),
    )

    expect(markup).toContain('OR group')
    expect(markup).toContain('Condition 1')
    expect(markup).toContain('value="domain"')
    expect(markup).toContain('value="deep_reasoning"')
    expect(markup).not.toContain('<textarea')
  })

  it('keeps raw JSON textarea handling out of the schema renderer', () => {
    const source = readFileSync(new URL('./builderPageFieldControls.tsx', import.meta.url), 'utf8')
    expect(source).not.toContain('case "json"')
    expect(source).not.toContain('<textarea')
    expect(source).toContain('<StructuredFieldEditor')
  })
})
