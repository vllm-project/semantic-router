import { describe, expect, it } from 'vitest'

import { DECISION_SIGNAL_TYPES } from '../generated/routerConfigContract'
import { getPluginFieldSchema, getSignalFieldSchema, SIGNAL_TYPES, type FieldSchema } from './dslSchemas'

function field(schema: FieldSchema[], key: string): FieldSchema | undefined {
  return schema.find((candidate) => candidate.key === key)
}

describe('agent routing in the Builder', () => {
  it('offers the agent signal as a route condition with role, harness and types', () => {
    expect(SIGNAL_TYPES).toContain('agent')
    expect(DECISION_SIGNAL_TYPES).toContain('agent')
    const schema = getSignalFieldSchema('agent')
    expect(field(schema, 'role')).toMatchObject({ label: 'Role', type: 'select', options: ['main', 'subagent'] })
    expect(field(schema, 'harness')).toMatchObject({
      label: 'Harness',
      type: 'select',
      options: ['claude_code', 'codex', 'opencode', 'deepseek_harness'],
    })
    expect(field(schema, 'types')).toMatchObject({ label: 'Subagent Types', type: 'string[]' })
    expect(field(schema, 'request_classes')).toMatchObject({ label: 'Request Classes', type: 'string[]' })
  })

  it('lets a route block what it matches with a fast_response', () => {
    expect(field(getPluginFieldSchema('fast_response'), 'block')).toMatchObject({
      label: 'Block the request',
      type: 'boolean',
    })
  })

  it('names the status a block answers with, as the Router sends it', () => {
    expect(field(getPluginFieldSchema('fast_response'), 'block')?.description).toContain('HTTP 400')
    const rules = field(getPluginFieldSchema('response_guard'), 'rules')
    expect(field(rules?.fields ?? [], 'action')?.description).toContain('HTTP 400')
  })
})
