import { describe, expect, it } from 'vitest'

import { parseStructuredDictionary, parseStructuredList } from './vsrTrace'

describe('parseStructuredDictionary', () => {
  it('reads items, inner lists, parameters and true booleans', () => {
    const trace = parseStructuredDictionary(
      'algorithm=static, signals=("keyword:a b" "pii:%C3%BC"), prompt_cache=skipped;inserted=1, system_prompt, retention;drop;ttl_turns=0',
    )

    expect(trace.get('algorithm')).toEqual({ value: 'static', items: [], params: {} })
    expect(trace.get('signals')?.items.map((item) => item.value)).toEqual(['keyword:a b', 'pii:ü'])
    expect(trace.get('prompt_cache')).toEqual({
      value: 'skipped',
      items: [],
      params: { inserted: '1' },
    })
    expect(trace.get('system_prompt')?.value).toBe('?1')
    expect(trace.get('retention')?.params).toEqual({ drop: '?1', ttl_turns: '0' })
  })

  it('unescapes quotes and backslashes in strings', () => {
    expect(
      parseStructuredDictionary('category="say \\"hi\\" \\\\ now"').get('category')?.value,
    ).toBe('say "hi" \\ now')
  })

  it('stops at a malformed member', () => {
    expect([...parseStructuredDictionary('algorithm=static, =broken').keys()]).toEqual([
      'algorithm',
    ])
  })
})

describe('parseStructuredList', () => {
  it('reads warning codes with their parameters', () => {
    expect(
      parseStructuredList('hallucination, protocol;action=dropped;field="tools[0].x"').map(
        (member) => [member.value, member.params],
      ),
    ).toEqual([
      ['hallucination', {}],
      ['protocol', { action: 'dropped', field: 'tools[0].x' }],
    ])
  })
})
