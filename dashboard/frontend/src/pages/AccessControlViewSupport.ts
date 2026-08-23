import type { AccessAPIKey, AccessGroup } from '../utils/inferenceAccessApi'
import type { AccessControlPageState as PageState } from './AccessControlViewTypes'

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

export function keyPolicy(key: AccessAPIKey, groups: AccessGroup[]) {
  if (key.effectiveAccess?.length) {
    return {
      direct: key.accessGroupIds.length > 0,
      resources: key.effectiveAccess,
    }
  }
  const effective = groups.filter((group) => key.accessGroupIds.includes(group.id))
  return {
    direct: key.accessGroupIds.length > 0,
    resources: [
      ...new Map(
        effective
          .flatMap((group) => group.resources)
          .map((resource) => [`${resource.resourceType}:${resource.resourceId}`, resource]),
      ).values(),
    ],
  }
}

export function friendlyAction(action: string) {
  return action
    .split(/[._]/)
    .map((word) => `${word.charAt(0).toUpperCase()}${word.slice(1)}`)
    .join(' ')
}
