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
    case 'agent':
      return [
        { key: 'description', label: 'Description', type: 'string' },
        {
          key: 'role',
          label: 'Role',
          type: 'select',
          options: ['main', 'subagent'],
          description: 'The main agent or a subagent, as protection reads the agent identity.',
        },
        {
          key: 'harness',
          label: 'Harness',
          type: 'select',
          options: ['claude_code', 'codex', 'opencode', 'deepseek_harness', 'cursor', 'openclaw'],
          description: 'The harness whose identity headers the request carries.',
        },
        {
          key: 'types',
          label: 'Subagent Types',
          type: 'string[]',
          placeholder: 'Explore',
          description:
            'Any of these matches, ignoring case: a Claude Code agent type (Explore, Plan, general-purpose, custom, teammate, fork) or a Codex child kind (collab_spawn, review).',
        },
        {
          key: 'request_classes',
          label: 'Request Classes',
          type: 'string[]',
          placeholder: 'compaction',
          description:
            'Any of these request classes matches: main, subagent, workflow, compaction or auxiliary. Claude Code sends the class; OpenCode session titles are auxiliary and its summaries compaction.',
        },
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
    default:
      return null
  }
}
