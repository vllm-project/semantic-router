import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import {
  ProjectionCalibrationEditor,
  ProjectionInputsEditor,
  ProjectionMembersEditor,
  ProjectionOutputsEditor,
} from './configPageProjectionStructuredEditors'

describe('projection structured editors', () => {
  it('renders partition members as values instead of JSON', () => {
    const markup = renderToStaticMarkup(
      createElement(ProjectionMembersEditor, {
        value: ['technical_support', 'account_management'],
        readOnly: true,
      }),
    )

    expect(markup).toContain('technical_support')
    expect(markup).toContain('account_management')
    expect(markup).not.toContain('JSON')
  })

  it('renders typed inputs, calibration, and output threshold controls', () => {
    const markup = renderToStaticMarkup(
      createElement(
        'div',
        null,
        createElement(ProjectionInputsEditor, {
          value: [{ type: 'embedding', name: 'support', weight: 0.5, value_source: 'confidence' }],
          onChange: vi.fn(),
        }),
        createElement(ProjectionCalibrationEditor, {
          value: { method: 'sigmoid_distance', slope: 6 },
          onChange: vi.fn(),
        }),
        createElement(ProjectionOutputsEditor, {
          value: [{ name: 'fast', lt: 0.25 }],
          onChange: vi.fn(),
        }),
      ),
    )

    expect(markup).toContain('Input type')
    expect(markup).toContain('Value source')
    expect(markup).toContain('Slope')
    expect(markup).toContain('Less than')
    expect(markup).not.toContain('textarea')
  })
})
