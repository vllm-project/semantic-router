import type { FieldSchema } from './dslSchemas'

export function getPolicySignalFieldSchema(signalType: string): FieldSchema[] | null {
  switch (signalType) {
    case 'metadata':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        { key: 'key', label: 'Metadata Key', type: 'string', required: true },
        {
          key: 'predicate',
          label: 'Predicate',
          type: 'object',
          required: true,
          description: 'Set exactly one of equals, in, or exists.',
          fields: [
            { key: 'equals', label: 'Equals', type: 'string' },
            { key: 'in', label: 'In', type: 'string[]' },
            { key: 'exists', label: 'Exists', type: 'boolean' },
          ],
        },
      ]
    case 'classifier':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'type',
          label: 'Backend Type',
          type: 'select',
          options: ['local', 'llm'],
          required: true,
        },
        { key: 'model', label: 'External Model', type: 'string' },
        { key: 'model_path', label: 'Local Model Path', type: 'string' },
        {
          key: 'labels',
          label: 'Labels',
          type: 'string[]',
          required: true,
        },
        { key: 'instructions', label: 'Instructions', type: 'string' },
        { key: 'use_cpu', label: 'Use CPU', type: 'boolean' },
      ]
    default:
      return null
  }
}
