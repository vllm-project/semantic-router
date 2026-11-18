import { describe, expect, it } from 'vitest'

import {
  DEFAULT_TRACE_LIMIT,
  DEFAULT_TRACE_LOOKBACK,
  formatAgo,
  formatMicros,
  jaegerOperationsUrl,
  jaegerServicesUrl,
  jaegerTraceUrl,
  jaegerTracesUrl,
  normalizeTraceDuration,
  parentSpanID,
  readJaegerNames,
  readJaegerTraces,
  readTraceLimit,
  readTraceLookback,
  serviceColor,
  summarizeTrace,
  type TraceSpan,
} from './jaegerTraces'

const jaegerTrace = {
  traceID: 'f00dfeed',
  spans: [
    {
      traceID: 'f00dfeed',
      spanID: 'root',
      operationName: 'semantic_router.request',
      references: [],
      startTime: 1_000_000,
      duration: 900_000,
      tags: [
        { key: 'span.kind', type: 'string', value: 'server' },
        { key: 'http.status_code', type: 'int64', value: 200 },
      ],
      logs: [
        {
          timestamp: 1_100_000,
          fields: [
            { key: 'event', type: 'string', value: 'routing.decision' },
            { key: 'decision', type: 'string', value: 'coding' },
          ],
        },
      ],
      processID: 'p1',
    },
    {
      traceID: 'f00dfeed',
      spanID: 'upstream',
      operationName: 'semantic_router.upstream.request',
      references: [{ refType: 'CHILD_OF', traceID: 'f00dfeed', spanID: 'root' }],
      startTime: 1_300_000,
      duration: 500_000,
      tags: [
        { key: 'error', type: 'bool', value: true },
        { key: 'attempt', type: 'string', value: { nested: 1 } },
      ],
      logs: [],
      processID: 'p1',
    },
    {
      traceID: 'f00dfeed',
      spanID: 'model',
      operationName: 'POST /v1/chat/completions',
      references: [{ refType: 'CHILD_OF', traceID: 'f00dfeed', spanID: 'upstream' }],
      startTime: 1_350_000,
      duration: 400_000,
      tags: [{ key: 'otel.status_code', type: 'string', value: 'ERROR' }],
      processID: 'p2',
    },
    { spanID: 'malformed', operationName: 'no timing', processID: 'p1' },
  ],
  processes: {
    p1: { serviceName: 'vllm-sr', tags: [] },
    p2: { serviceName: 'qwen3-backend', tags: [] },
  },
}

describe('Jaeger search', () => {
  it('builds the query API addresses with only the filters that are set', () => {
    expect(jaegerServicesUrl()).toBe('/embedded/jaeger/api/services')
    expect(jaegerOperationsUrl('vllm sr/1')).toBe(
      '/embedded/jaeger/api/services/vllm%20sr%2F1/operations',
    )
    expect(jaegerTraceUrl('f00d')).toBe('/embedded/jaeger/api/traces/f00d')
    const all = new URL(
      jaegerTracesUrl({
        service: 'vllm-sr',
        operation: '',
        lookback: '1h',
        limit: 20,
        minDuration: '',
      }),
      'http://x',
    )
    expect(all.pathname).toBe('/embedded/jaeger/api/traces')
    expect([...all.searchParams]).toEqual([
      ['service', 'vllm-sr'],
      ['lookback', '1h'],
      ['limit', '20'],
    ])
    const filtered = new URL(
      jaegerTracesUrl({
        service: 'vllm-sr',
        operation: 'semantic_router.request',
        lookback: '168h',
        limit: 50,
        minDuration: '250ms',
      }),
      'http://x',
    )
    expect(Object.fromEntries(filtered.searchParams)).toEqual({
      service: 'vllm-sr',
      operation: 'semantic_router.request',
      lookback: '168h',
      limit: '50',
      minDuration: '250ms',
    })
  })

  it('reads search filters from the URL with Go durations Jaeger accepts', () => {
    expect(readTraceLookback('6h')).toBe('6h')
    expect(readTraceLookback('7d')).toBe(DEFAULT_TRACE_LOOKBACK)
    expect(readTraceLimit('100')).toBe(100)
    expect(readTraceLimit('37')).toBe(DEFAULT_TRACE_LIMIT)
    expect(normalizeTraceDuration(' 250 ms ')).toBe('250ms')
    expect(normalizeTraceDuration('1.5s')).toBe('1.5s')
    expect(normalizeTraceDuration('1m30s')).toBe('1m30s')
    expect(normalizeTraceDuration('500µs')).toBe('500µs')
    expect(normalizeTraceDuration('')).toBe('')
    expect(normalizeTraceDuration('fast')).toBeNull()
    expect(normalizeTraceDuration('2d')).toBeNull()
    expect(normalizeTraceDuration('100')).toBeNull()
  })

  it('reads service and operation names, sorted and without duplicates', () => {
    expect(readJaegerNames({ data: ['vllm-sr', 'envoy', 'vllm-sr', ''], total: 3 })).toEqual([
      'envoy',
      'vllm-sr',
    ])
    expect(
      readJaegerNames({ data: [{ name: 'POST', spanKind: 'server' }, { spanKind: 'client' }] }),
    ).toEqual(['POST'])
    expect(readJaegerNames({ data: null, errors: null })).toEqual([])
    expect(() => readJaegerNames({ data: 'vllm-sr' })).toThrow('Jaeger did not return a list.')
  })
})

describe('Jaeger traces', () => {
  it('reads spans with their services, string values and failures', () => {
    const [trace] = readJaegerTraces({ data: [jaegerTrace], total: 0, errors: null })
    expect(trace.traceID).toBe('f00dfeed')
    expect(trace.spans.map((span) => span.spanID)).toEqual(['root', 'upstream', 'model'])
    const [root, upstream, model] = trace.spans
    expect(root).toMatchObject({
      serviceName: 'vllm-sr',
      kind: 'server',
      error: false,
      references: [],
    })
    expect(root.tags).toContainEqual({ key: 'http.status_code', value: '200' })
    expect(root.logs[0].fields[1]).toEqual({ key: 'decision', value: 'coding' })
    expect(upstream).toMatchObject({
      error: true,
      references: [{ refType: 'CHILD_OF', spanID: 'root' }],
    })
    expect(upstream.tags).toContainEqual({ key: 'attempt', value: '{"nested":1}' })
    expect(model).toMatchObject({ serviceName: 'qwen3-backend', error: true, logs: [] })
  })

  it('keeps a trace returned twice once, with the most spans', () => {
    const partial = { ...jaegerTrace, spans: jaegerTrace.spans.slice(0, 1) }
    const traces = readJaegerTraces({ data: [partial, jaegerTrace, partial] })
    expect(traces).toHaveLength(1)
    expect(traces[0].spans).toHaveLength(3)
  })

  it('drops traces without readable spans and accepts an empty answer', () => {
    expect(readJaegerTraces({ data: null })).toEqual([])
    expect(
      readJaegerTraces({
        data: [{ traceID: 'empty', spans: [{ spanID: 'x' }] }, { spans: [] }, null],
      }),
    ).toEqual([])
    expect(() => readJaegerTraces({ data: {} })).toThrow('Jaeger did not return traces.')
  })

  it('summarizes a trace from its root span', () => {
    const [trace] = readJaegerTraces({ data: [jaegerTrace] })
    expect(summarizeTrace(trace)).toMatchObject({
      traceID: 'f00dfeed',
      root: { spanID: 'root', operationName: 'semantic_router.request' },
      services: [
        { name: 'vllm-sr', spans: 2 },
        { name: 'qwen3-backend', spans: 1 },
      ],
      spanCount: 3,
      errorCount: 2,
      start: 1_000_000,
      duration: 900_000,
    })
  })

  it('takes a span whose parent the trace lacks as a root', () => {
    const span = (spanID: string, references: TraceSpan['references']): TraceSpan => ({
      spanID,
      references,
      operationName: spanID,
      serviceName: 'svc',
      startTime: 0,
      duration: 1,
      tags: [],
      logs: [],
      error: false,
      kind: null,
    })
    const ids = new Set(['a', 'b'])
    expect(parentSpanID(span('b', [{ refType: 'CHILD_OF', spanID: 'gone' }]), ids)).toBeNull()
    expect(parentSpanID(span('b', [{ refType: 'CHILD_OF', spanID: 'b' }]), ids)).toBeNull()
    expect(
      parentSpanID(
        span('b', [
          { refType: 'FOLLOWS_FROM', spanID: 'gone' },
          { refType: 'FOLLOWS_FROM', spanID: 'a' },
        ]),
        ids,
      ),
    ).toBe('a')
    expect(
      parentSpanID(
        span('b', [
          { refType: 'FOLLOWS_FROM', spanID: 'c' },
          { refType: 'CHILD_OF', spanID: 'a' },
        ]),
        new Set(['a', 'b', 'c']),
      ),
    ).toBe('a')
  })
})

describe('trace formatting', () => {
  it('formats microsecond durations and ages', () => {
    expect(formatMicros(420)).toBe('420 µs')
    expect(formatMicros(4_200)).toBe('4.20 ms')
    expect(formatMicros(42_000)).toBe('42.0 ms')
    expect(formatMicros(420_000)).toBe('420 ms')
    expect(formatMicros(4_200_000)).toBe('4.20 s')
    expect(formatMicros(125_000_000)).toBe('2m 5s')
    expect(formatMicros(-1)).toBe('—')
    const now = 10_000_000
    expect(formatAgo(now - 20_000, now)).toBe('Just now')
    expect(formatAgo(now - 5 * 60_000, now)).toBe('5 min ago')
    expect(formatAgo(now - 3 * 3_600_000, now)).toBe('3 h ago')
  })

  it('keeps one color per service and leaves rose for errors', () => {
    const services = ['envoy', 'qwen3-backend', 'vllm-sr']
    expect(serviceColor(services, 'envoy')).toBe('var(--chart-1)')
    expect(serviceColor(services, 'vllm-sr')).toBe(serviceColor([...services], 'vllm-sr'))
    expect(serviceColor(services, 'missing')).toBe('var(--chart-8)')
    const many = Array.from({ length: 20 }, (_, index) => `service-${index}`)
    expect(many.map((name) => serviceColor(many, name))).not.toContain('var(--chart-5)')
  })
})
