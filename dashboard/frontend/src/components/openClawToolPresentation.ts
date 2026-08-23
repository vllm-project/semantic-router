interface HighlightField {
  label: string
  value: string
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  return value as Record<string, unknown>
}

function fieldString(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return ''
}

export function truncateHighlight(value: string, maxLength = 120): string {
  const text = value.trim()
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength - 3).trim()}...`
}

function extractRawArgument(rawArguments: string, key: string): string {
  if (!rawArguments) return ''
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return rawArguments.match(new RegExp(`"${escapedKey}"\\s*:\\s*"([^"]*)`))?.[1]?.trim() ?? ''
}

function firstValue(
  source: Record<string, unknown> | null,
  keys: string[],
  rawArguments = '',
): string {
  for (const key of keys) {
    const value = fieldString(source?.[key])
    if (value) return value
  }
  for (const key of keys) {
    const value = extractRawArgument(rawArguments, key)
    if (value) return value
  }
  return ''
}

function highlights(entries: Array<[string, string]>): HighlightField[] {
  return entries
    .filter(([, value]) => Boolean(value))
    .map(([label, value]) => ({ label, value: truncateHighlight(value) }))
}

export function buildClawRequestHighlights(
  toolName: string,
  argumentsObject: Record<string, unknown> | null,
  rawArguments: string,
): HighlightField[] {
  if (toolName === 'claw_create_team') {
    return highlights([
      ['name', firstValue(argumentsObject, ['name'], rawArguments)],
      ['vibe', firstValue(argumentsObject, ['vibe'], rawArguments)],
      ['role', firstValue(argumentsObject, ['role'], rawArguments)],
      ['principal', firstValue(argumentsObject, ['principal'], rawArguments)],
    ])
  }
  if (toolName === 'claw_create_worker') {
    return highlights([
      ['name', firstValue(argumentsObject, ['name'], rawArguments)],
      ['vibe', firstValue(argumentsObject, ['vibe'], rawArguments)],
      ['role', firstValue(argumentsObject, ['role'], rawArguments)],
      ['team', firstValue(argumentsObject, ['team_id', 'teamId'], rawArguments)],
    ])
  }
  return []
}

export function buildClawResultHighlights(
  toolName: string,
  resultContent: unknown,
  argumentsObject: Record<string, unknown> | null,
  rawArguments: string,
): HighlightField[] {
  const result = asRecord(resultContent)
  if (toolName === 'claw_create_team') {
    return highlights([
      ['name', firstValue(result, ['name']) || firstValue(argumentsObject, ['name'], rawArguments)],
      ['vibe', firstValue(result, ['vibe']) || firstValue(argumentsObject, ['vibe'], rawArguments)],
      ['role', firstValue(result, ['role']) || firstValue(argumentsObject, ['role'], rawArguments)],
      ['team_id', firstValue(result, ['id'])],
    ])
  }
  if (toolName === 'claw_create_worker') {
    const identity = asRecord(result?.identity)
    return highlights([
      [
        'name',
        firstValue(identity, ['name']) ||
          firstValue(result, ['agentName', 'name']) ||
          firstValue(argumentsObject, ['name'], rawArguments),
      ],
      [
        'vibe',
        firstValue(identity, ['vibe']) ||
          firstValue(result, ['agentVibe']) ||
          firstValue(argumentsObject, ['vibe'], rawArguments),
      ],
      [
        'role',
        firstValue(identity, ['role']) ||
          firstValue(result, ['agentRole']) ||
          firstValue(argumentsObject, ['role'], rawArguments),
      ],
      [
        'team',
        firstValue(result, ['teamName', 'teamId']) ||
          firstValue(argumentsObject, ['team_id', 'teamId'], rawArguments),
      ],
      ['container', firstValue(result, ['containerName'])],
      ['message', firstValue(result, ['message'])],
    ])
  }
  return []
}
