import type { AccessEditor } from './AccessControlPageSupport'

export const ACCESS_EDITOR_TITLES: Record<
  AccessEditor['kind'],
  { eyebrow: string; create: string; edit: string; description: string }
> = {
  user: {
    eyebrow: 'Identity',
    create: 'Add user',
    edit: 'Edit user',
    description: 'A user can own API keys, join teams, and optionally receive Dashboard access.',
  },
  team: {
    eyebrow: 'Identity',
    create: 'Create team',
    edit: 'Edit team',
    description: 'Group users under shared model grants and quota.',
  },
  key: {
    eyebrow: 'Credential',
    create: 'Create API key',
    edit: 'API key',
    description: 'Choose an owner, model visibility, and an optional key-specific limit.',
  },
  group: {
    eyebrow: 'Model policy',
    create: 'Create access group',
    edit: 'Edit access group',
    description: 'Compose reusable model grants and assign them to identities or keys.',
  },
  budget: {
    eyebrow: 'Rate limit',
    create: 'Create budget',
    edit: 'Edit budget',
    description: 'Enforce RPM, TPM, and daily tokens at any scope.',
  },
}

export function toLocalDateTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60_000)
  return local.toISOString().slice(0, 16)
}
