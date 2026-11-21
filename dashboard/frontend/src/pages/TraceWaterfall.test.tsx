import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { TraceWaterfall } from './TraceWaterfall'
import type { TraceSpan } from './jaegerTraces'
import { layoutWaterfall } from './traceWaterfall'

const span = (spanID: string, startTime: number, duration: number, parent?: string): TraceSpan => ({
  spanID,
  references: parent ? [{ refType: 'CHILD_OF', spanID: parent }] : [],
  operationName: `${spanID} operation`,
  serviceName: spanID === 'model' ? 'qwen3-backend' : 'vllm-sr',
  startTime,
  duration,
  tags: [],
  logs: [],
  error: spanID === 'model',
  kind: null,
})

describe('trace waterfall view', () => {
  it('renders the span tree with levels, the selection and timing labels', () => {
    const waterfall = layoutWaterfall({
      traceID: 'f00d',
      spans: [
        span('root', 0, 2_000),
        span('route', 100, 400, 'root'),
        span('model', 600, 1_200, 'root'),
      ],
    })
    const markup = renderToStaticMarkup(
      <TraceWaterfall
        waterfall={waterfall}
        palette={['qwen3-backend', 'vllm-sr']}
        selected={waterfall.rows[1].span}
        onSelect={() => undefined}
      />,
    )
    expect(markup).toContain('role="tree"')
    expect(markup.match(/role="treeitem"/g)).toHaveLength(3)
    expect(markup.match(/aria-level="1"/g)).toHaveLength(1)
    expect(markup.match(/aria-level="2"/g)).toHaveLength(2)
    expect(markup.match(/aria-selected="true"/g)).toHaveLength(1)
    expect(markup).toContain('aria-expanded="true"')
    expect(markup).toContain('route operation')
    expect(markup).toContain('400 µs')
    expect(markup).toContain('1.20 ms')
    expect(markup).toContain('aria-label="Error"')
    expect(markup).toContain('background:var(--destructive)')
    expect(markup).toContain('background:var(--chart-3)')
  })
})
