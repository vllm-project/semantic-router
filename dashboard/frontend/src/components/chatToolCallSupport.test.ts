import { describe, expect, it } from 'vitest'

import {
  extractTextToolCalls,
  mergeToolCallArgumentChunk,
  normalizeToolCallArguments,
} from './chatToolCallSupport'

describe('extractTextToolCalls', () => {
  it('normalizes Qwen tagged tool calls embedded after prose', () => {
    const result = extractTextToolCalls(`
Let me search for more specific technical details.

<tool_call>
<function=search_web>
<parameter=query>vLLM Semantic Router architecture technical details</parameter>
<parameter=max_results>10</parameter>
<parameter=output_format>json</parameter>
</function>
</tool_call>
`)

    expect(result.content).toBe('Let me search for more specific technical details.')
    expect(result.toolCalls).toEqual([
      {
        index: 0,
        functionName: 'search_web',
        functionArguments: JSON.stringify({
          query: 'vLLM Semantic Router architecture technical details',
          max_results: 10,
          output_format: 'json',
        }),
      },
    ])
  })

  it('normalizes JSON tool call envelopes', () => {
    const result = extractTextToolCalls(
      '<tool_call>{"name":"search_web","arguments":{"query":"vllm-sr"}}</tool_call>',
    )

    expect(result.content).toBe('')
    expect(result.toolCalls[0]).toMatchObject({
      functionName: 'search_web',
      functionArguments: '{"query":"vllm-sr"}',
    })
  })
})

describe('mergeToolCallArgumentChunk', () => {
  it('does not duplicate cumulative or repeated streaming arguments', () => {
    expect(mergeToolCallArgumentChunk('{"query":"vllm', '{"query":"vllm-sr"}')).toBe(
      '{"query":"vllm-sr"}',
    )
    expect(mergeToolCallArgumentChunk('{"query":"vllm-sr"}', '{"query":"vllm-sr"}')).toBe(
      '{"query":"vllm-sr"}',
    )
  })

  it('still appends true delta chunks', () => {
    expect(mergeToolCallArgumentChunk('{"query":"', 'vllm-sr"}')).toBe('{"query":"vllm-sr"}')
  })
})

describe('normalizeToolCallArguments', () => {
  it('repairs legacy concatenated JSON arguments before replay', () => {
    expect(
      normalizeToolCallArguments(
        '{"query":"vllm-sr","max_results":10}{"query":"vllm-sr","max_results":10}',
      ),
    ).toBe('{"query":"vllm-sr","max_results":10}')
  })

  it('uses an empty object when legacy arguments cannot be recovered', () => {
    expect(normalizeToolCallArguments('not-json')).toBe('{}')
  })
})
