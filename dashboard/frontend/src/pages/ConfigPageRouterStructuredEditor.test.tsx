import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import ConfigPageRouterStructuredEditor from './ConfigPageRouterStructuredEditor'
import { ROUTER_STRUCTURED_FIELDS } from './configPageRouterStructuredSchema'

describe('ConfigPageRouterStructuredEditor', () => {
  it('renders nested object arrays and maps without a JSON textarea', () => {
    const schema = ROUTER_STRUCTURED_FIELDS.authz?.providers.schema
    expect(schema).toBeDefined()
    const markup = renderToStaticMarkup(
      <ConfigPageRouterStructuredEditor
        schema={schema!}
        value={[{ type: 'header', headers: { 'X-Plan': 'premium' } }]}
        onChange={vi.fn()}
        readOnly
      />,
    )

    expect(markup).toContain('header')
    expect(markup).toContain('X-Plan')
    expect(markup).toContain('premium')
    expect(markup).not.toContain('<textarea')
  })

  it('renders string arrays as addable typed rows', () => {
    const schema = ROUTER_STRUCTURED_FIELDS.router_core?.auto_model_names.schema
    expect(schema).toBeDefined()
    const markup = renderToStaticMarkup(
      <ConfigPageRouterStructuredEditor
        schema={schema!}
        value={['vllm-sr/auto', 'MoM']}
        onChange={vi.fn()}
      />,
    )

    expect(markup).toContain('vllm-sr/auto')
    expect(markup).toContain('MoM')
    expect(markup).toContain('Add alias')
  })

  it('surfaces preserved future fields in read-only object views', () => {
    const schema = ROUTER_STRUCTURED_FIELDS.router_core?.streamed_body.schema
    expect(schema).toBeDefined()
    const markup = renderToStaticMarkup(
      <ConfigPageRouterStructuredEditor
        schema={schema!}
        value={{ enabled: true, future_limit: 3 }}
        onChange={vi.fn()}
        readOnly
      />,
    )

    expect(markup).toContain('1 additional advanced field')
    expect(markup).toContain('Enabled')
  })
})
