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
          options: ['local', 'llm', 'sequence_classifier'],
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
        {
          key: 'disable_rationale',
          label: 'Disable Rationale',
          type: 'boolean',
          description: 'Only LLM classifiers support this option.',
        },
        { key: 'use_cpu', label: 'Use CPU', type: 'boolean' },
      ]
    case 'input_modality':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'modality',
          label: 'Modality',
          type: 'select',
          options: ['text', 'image', 'audio', 'video'],
          required: true,
          description: 'Input modality whose structural presence this signal matches.',
        },
      ]
    case 'topic_continuity':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'include_assistant',
          label: 'Include Assistant Text',
          type: 'boolean',
          description: 'Read assistant text of prior turns as evidence (default true).',
        },
        {
          key: 'thresholds',
          label: 'Thresholds',
          type: 'object',
          description: 'Require 0 <= change < continuation < 1. Defaults: continuation 0.35, change 0.08.',
          fields: [
            { key: 'continuation', label: 'Continuation', type: 'number' },
            { key: 'change', label: 'Change', type: 'number' },
          ],
        },
        {
          key: 'limits',
          label: 'Evidence Limits',
          type: 'object',
          description:
            'Defaults: 8 prior turns, 16384 bytes per turn, and (prior turns + 1) x turn bytes in total.',
          fields: [
            { key: 'max_prior_turns', label: 'Max Prior Turns', type: 'number' },
            { key: 'max_turn_bytes', label: 'Max Turn Bytes', type: 'number' },
            { key: 'max_input_bytes', label: 'Max Input Bytes', type: 'number' },
          ],
        },
      ]
    default:
      return null
  }
}
