import { describe, expect, it } from 'vitest'

import { serializeFields, type FieldSchema } from '@/lib/dslMutations'

import {
  normalizeStructuredObject,
  normalizeStructuredObjectList,
  normalizeStructuredStringMatrix,
  removeStructuredObjectListItem,
  requiredStructuredFieldErrors,
  structuredItemLabel,
  updateStructuredObjectField,
  updateStructuredObjectListItem,
} from './builderPageStructuredFieldSupport'

describe('Builder structured field interactions', () => {
  it('adopts valid legacy JSON values without preserving raw text state', () => {
    expect(normalizeStructuredObject('{"quality":0.4,"cost":0.6}')).toEqual({
      quality: 0.4,
      cost: 0.6,
    })
    expect(normalizeStructuredObjectList('[{"name":"worker","models":["qwen"]}]')).toEqual([
      { name: 'worker', models: ['qwen'] },
    ])
    expect(normalizeStructuredStringMatrix('[ ["first", "then"], ["先", "再"] ]')).toEqual([
      ['first', 'then'],
      ['先', '再'],
    ])
  })

  it('updates and clears nested object fields immutably', () => {
    const original = { quality: 0.4, cost: 0.2 }
    const updated = updateStructuredObjectField(original, 'quality', 0.6)
    const cleared = updateStructuredObjectField(updated, 'cost', undefined)

    expect(original).toEqual({ quality: 0.4, cost: 0.2 })
    expect(updated).toEqual({ quality: 0.6, cost: 0.2 })
    expect(cleared).toEqual({ quality: 0.6 })
  })

  it('updates and removes one object-list entry without disturbing siblings', () => {
    const items = [{ name: 'planner' }, { name: 'worker' }]
    const updated = updateStructuredObjectListItem(items, 1, {
      name: 'reviewer',
      models: ['qwen-32b'],
    })

    expect(updated).toEqual([
      { name: 'planner' },
      { name: 'reviewer', models: ['qwen-32b'] },
    ])
    expect(removeStructuredObjectListItem(updated, 0)).toEqual([
      { name: 'reviewer', models: ['qwen-32b'] },
    ])
  })

  it('reports required nested values and derives stable item labels', () => {
    const fields: FieldSchema[] = [
      { key: 'name', label: 'Role Name', type: 'string', required: true },
      { key: 'models', label: 'Models', type: 'string[]', required: true },
    ]
    const schema: FieldSchema = {
      key: 'roles',
      label: 'Roles',
      type: 'object[]',
      itemLabel: 'Role',
      itemLabelKey: 'name',
      fields,
    }

    expect(requiredStructuredFieldErrors(fields, { name: '', models: [] })).toEqual([
      'Role Name is required.',
      'Models is required.',
    ])
    expect(structuredItemLabel(schema, { name: 'reviewer' }, 0)).toBe('Role: reviewer')
    expect(structuredItemLabel(schema, {}, 1)).toBe('Role 2')
  })

  it('serializes structured form values through the existing DSL value path', () => {
    const source = serializeFields({
      planner: { model: 'qwen-coordinator', max_completion_tokens: 1024 },
      roles: [{ name: 'worker', models: ['qwen-worker'] }],
      weights: { quality: 0.4, cost: 0.6 },
    })

    expect(source).toContain('planner: { model: "qwen-coordinator", max_completion_tokens: 1024 }')
    expect(source).toContain('name: "worker"')
    expect(source).toContain('models: ["qwen-worker"]')
    expect(source).toContain('weights: { quality: 0.4, cost: 0.6 }')
  })
})
