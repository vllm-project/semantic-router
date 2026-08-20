import type {
  AccessAPIKey,
  AccessGroup,
  AccessTeam,
  AccessUser,
  UsageSummary,
} from '../utils/inferenceAccessApi'

export const EMPTY_USAGE: UsageSummary = {
  requests: 0,
  successful: 0,
  failed: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  activeKeys: 0,
  averageLatencyMs: 0,
  p95LatencyMs: 0,
  averageTtftMs: 0,
  p95TtftMs: 0,
  series: [],
  byModel: [],
  byUser: [],
  byTeam: [],
  byKey: [],
}

export const formatNumber = (value?: number) => new Intl.NumberFormat('en-US').format(value || 0)
export const formatDate = (value?: string) =>
  value
    ? new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeStyle: 'short' }).format(
        new Date(value),
      )
    : 'Never'

export function effectivePatterns(key: AccessAPIKey, groups: AccessGroup[]) {
  if (key.modelPatterns?.length) return key.modelPatterns
  const direct = groups.filter((group) =>
    group.bindings.some((binding) => binding.subjectType === 'key' && binding.subjectId === key.id),
  )
  const inherited = groups.filter((group) =>
    group.bindings.some(
      (binding) =>
        (binding.subjectType === 'user' && binding.subjectId === key.userId) ||
        (binding.subjectType === 'team' && binding.subjectId === key.teamId),
    ),
  )
  return [...new Set((direct.length ? direct : inherited).flatMap((group) => group.modelPatterns))]
}

export function ownerLabel(key: AccessAPIKey, users: AccessUser[], teams: AccessTeam[]) {
  if (key.userId) return users.find((user) => user.id === key.userId)?.name || key.userId
  return teams.find((team) => team.id === key.teamId)?.name || key.teamId || 'Unassigned'
}
