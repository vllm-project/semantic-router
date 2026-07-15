import { readFileSync } from 'node:fs'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import {
  BatchSizeRangesEditor,
  MetricBucketsEditor,
  TracingExporterEditor,
  TracingResourceEditor,
  TracingSamplingEditor,
} from './configPageToolsObservabilityStructuredEditors'
import {
  createTracingEditorValue,
  normalizeBatchMetrics,
  normalizeTracingConfig,
} from './configPageToolsObservabilitySupport'

describe('tools observability structured fields', () => {
  it('normalizes tracing objects and retains their wire shape', () => {
    expect(
      normalizeTracingConfig({
        enabled: true,
        provider: 'otlp',
        exporter: { type: 'otlp', endpoint: 'http://localhost:4318', insecure: true },
        sampling: { type: 'probabilistic', rate: 0.1 },
        resource: {
          service_name: 'semantic-router',
          service_version: '1.0.0',
          deployment_environment: 'production',
        },
      }),
    ).toMatchObject({
      exporter: { type: 'otlp', endpoint: 'http://localhost:4318', insecure: true },
      sampling: { type: 'probabilistic', rate: 0.1 },
      resource: { service_name: 'semantic-router' },
    })

    expect(
      createTracingEditorValue({ enabled: false } as unknown as Parameters<
        typeof createTracingEditorValue
      >[0]),
    ).toMatchObject({
      provider: 'otlp',
      exporter: { type: 'otlp' },
      sampling: { type: 'probabilistic', rate: 0.1 },
      resource: { service_name: 'semantic-router' },
    })
  })

  it('converts bucket editor strings back to numeric arrays and validates schemas', () => {
    expect(
      normalizeBatchMetrics({
        batch_size_ranges: [{ min: 1, max: 8, label: '1-8' }],
        duration_buckets: ['0.01', '0.1'] as unknown as number[],
        size_buckets: ['1', '8', '16'] as unknown as number[],
      }),
    ).toEqual({
      batch_size_ranges: [{ min: 1, max: 8, label: '1-8' }],
      duration_buckets: [0.01, 0.1],
      size_buckets: [1, 8, 16],
    })

    expect(() =>
      normalizeBatchMetrics({ duration_buckets: [0.1, 0.05] }),
    ).toThrow(/strictly increasing/i)
    expect(() =>
      normalizeBatchMetrics({ size_buckets: [1, 1.5] }),
    ).toThrow(/integer/i)
    expect(() =>
      normalizeTracingConfig({
        enabled: true,
        provider: 'otlp',
        exporter: { type: 'otlp' },
        sampling: { type: 'probabilistic', rate: 2 },
        resource: {
          service_name: 'semantic-router',
          service_version: '1.0.0',
          deployment_environment: 'production',
        },
      }),
    ).toThrow(/between 0 and 1/i)
  })

  it('renders typed object and list controls without JSON textareas', () => {
    const markup = renderToStaticMarkup(
      createElement(
        'div',
        null,
        createElement(TracingExporterEditor, { value: { type: 'otlp' }, onChange: vi.fn() }),
        createElement(TracingSamplingEditor, {
          value: { type: 'probabilistic', rate: 0.1 },
          onChange: vi.fn(),
        }),
        createElement(TracingResourceEditor, {
          value: {
            service_name: 'semantic-router',
            service_version: '1.0.0',
            deployment_environment: 'production',
          },
          onChange: vi.fn(),
        }),
        createElement(BatchSizeRangesEditor, {
          value: [{ min: 1, max: 8, label: '1-8' }],
          onChange: vi.fn(),
        }),
        createElement(MetricBucketsEditor, {
          value: [0.01, 0.1],
          onChange: vi.fn(),
          label: 'Duration buckets',
          placeholder: '0.1',
        }),
      ),
    )

    expect(markup).toContain('Exporter type')
    expect(markup).toContain('Sampling type')
    expect(markup).toContain('Service name')
    expect(markup).toContain('Minimum batch size')
    expect(markup).not.toContain('textarea')
    expect(markup).not.toContain('(JSON)')
  })

  it('wires every observability structure into typed modal controls', () => {
    const source = readFileSync(
      new URL('./ConfigPageToolsObservabilitySection.tsx', import.meta.url),
      'utf8',
    )
    expect(source).toContain('<TracingExporterEditor')
    expect(source).toContain('<TracingSamplingEditor')
    expect(source).toContain('<TracingResourceEditor')
    expect(source).toContain('<BatchSizeRangesEditor')
    expect(source).toContain('<MetricBucketsEditor')
    expect(source).not.toMatch(/Configuration \(JSON\)|Buckets \(JSON\)|Ranges \(JSON\)/)
  })
})
