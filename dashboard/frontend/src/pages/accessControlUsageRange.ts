export type UsageRangePreset = 'today' | '7d' | '30d' | 'mtd' | 'ytd' | 'custom'
export type UsageGranularity = 'auto' | 'minute' | 'hour' | 'day'

export interface UsageScope {
  type: 'global' | 'user' | 'team' | 'key'
  id: string
  model: string
  range: UsageRangePreset
  granularity: UsageGranularity
  customFrom: string
  customTo: string
}

const startOfDay = (value: Date) => {
  const result = new Date(value)
  result.setHours(0, 0, 0, 0)
  return result
}

export const dateInputValue = (value: Date) => {
  const year = value.getFullYear()
  const month = String(value.getMonth() + 1).padStart(2, '0')
  const day = String(value.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}

const parseLocalDate = (value: string) => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (!match) return null
  const parsed = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]))
  return Number.isNaN(parsed.getTime()) ? null : parsed
}

export function usageRangeBounds(scope: UsageScope, now = new Date()) {
  let from = startOfDay(now)
  let to = now
  switch (scope.range) {
    case '7d':
      from.setDate(from.getDate() - 6)
      break
    case '30d':
      from.setDate(from.getDate() - 29)
      break
    case 'mtd':
      from.setDate(1)
      break
    case 'ytd':
      from.setMonth(0, 1)
      break
    case 'custom': {
      const customFrom = parseLocalDate(scope.customFrom)
      const customTo = parseLocalDate(scope.customTo)
      if (customFrom) from = startOfDay(customFrom)
      if (customTo) {
        to = startOfDay(customTo)
        to.setDate(to.getDate() + 1)
        to.setMilliseconds(-1)
      }
      break
    }
    case 'today':
      break
  }
  if (to > now) to = now
  if (from > to) from = startOfDay(to)
  return { from: from.toISOString(), to: to.toISOString() }
}

export function usageRangeDays(scope: UsageScope, now = new Date()) {
  const bounds = usageRangeBounds(scope, now)
  return Math.max(1, (Date.parse(bounds.to) - Date.parse(bounds.from)) / 86_400_000)
}

export function rangeLabel(range: UsageRangePreset) {
  return (
    {
      today: 'Today',
      '7d': '7D',
      '30d': '30D',
      mtd: 'MTD',
      ytd: 'YTD',
      custom: 'Custom',
    } as const
  )[range]
}
