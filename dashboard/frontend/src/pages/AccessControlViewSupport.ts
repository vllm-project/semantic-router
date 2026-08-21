import type { AccessAPIKey, AccessBudget, AccessGroup } from '../utils/inferenceAccessApi'
import type {
  AccessControlPageState as PageState,
  AccessControlViewProps as Props,
} from './AccessControlViewTypes'

export const number = (value?: number) => new Intl.NumberFormat('en-US').format(value || 0)

export const date = (value?: string | number) => {
  if (!value) return 'Never'
  const timestamp = typeof value === 'number' ? value * 1000 : Date.parse(value)
  if (Number.isNaN(timestamp)) return 'Never'
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(timestamp)
}

export const percent = (part: number, total: number) =>
  total ? `${((part / total) * 100).toFixed(1)}%` : '—'

export const initials = (value: string) =>
  value
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('') || 'U'

export function slicePage<T>(items: T[], state: PageState) {
  const start = (state.page - 1) * state.pageSize
  return items.slice(start, start + state.pageSize)
}

export function assignmentSummary(group: AccessGroup) {
  const counts = group.bindings.reduce<Record<string, number>>(
    (result, binding) => ({
      ...result,
      [binding.subjectType]: (result[binding.subjectType] || 0) + 1,
    }),
    {},
  )
  return (
    [
      counts.user ? `${counts.user} users` : '',
      counts.team ? `${counts.team} teams` : '',
      counts.key ? `${counts.key} keys` : '',
    ]
      .filter(Boolean)
      .join(' · ') || 'Not assigned'
  )
}

export function keyPolicy(key: AccessAPIKey, groups: AccessGroup[]) {
  if (key.modelPatterns?.length) {
    return {
      direct: key.accessGroupIds.length > 0,
      patterns: [...new Set(key.modelPatterns)],
    }
  }
  const direct = groups.filter((group) =>
    group.bindings.some((binding) => binding.subjectType === 'key' && binding.subjectId === key.id),
  )
  const user = groups.filter((group) =>
    group.bindings.some(
      (binding) => binding.subjectType === 'user' && binding.subjectId === key.userId,
    ),
  )
  const team = groups.filter((group) =>
    group.bindings.some(
      (binding) =>
        binding.subjectType === 'team' && binding.subjectId === (key.effectiveTeamId || key.teamId),
    ),
  )
  const effective = direct.length ? direct : user.length ? user : team
  return {
    direct: direct.length > 0,
    patterns: [...new Set(effective.flatMap((group) => group.modelPatterns))],
  }
}

export function scopeName(budget: AccessBudget, props: Props) {
  if (budget.scopeType === 'global') return 'All traffic'
  if (budget.scopeType === 'user')
    return props.users.find((item) => item.id === budget.scopeId)?.name || budget.scopeId
  if (budget.scopeType === 'team')
    return props.teams.find((item) => item.id === budget.scopeId)?.name || budget.scopeId
  return props.keys.find((item) => item.id === budget.scopeId)?.name || budget.scopeId
}

export function friendlyAction(action: string) {
  return action
    .split(/[._]/)
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(' ')
}
