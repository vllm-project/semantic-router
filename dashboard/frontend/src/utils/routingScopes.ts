export interface RoutingProfileLike {
  signals?: Record<string, unknown>
  projections?: Record<string, unknown>
  decisions?: unknown[]
  strategy?: unknown
  [key: string]: unknown
}

export interface RoutingScope {
  id: string
  label: string
  description?: string
  entrypointModelNames: string[]
  document: RoutingProfileLike
}

const arrayCount = (value: unknown): number => (Array.isArray(value) ? value.length : 0)

export function countSignalsInProfile(profile: RoutingProfileLike | undefined): {
  total: number
  byType: Record<string, number>
} {
  const byType: Record<string, number> = {}
  let total = 0
  for (const [type, value] of Object.entries(profile?.signals ?? {})) {
    const count = arrayCount(value)
    if (count === 0) continue
    byType[type] = count
    total += count
  }
  return { total, byType }
}

export function countProjectionsInProfile(profile: RoutingProfileLike | undefined): number {
  return Object.values(profile?.projections ?? {}).reduce<number>(
    (total, value) => total + (Array.isArray(value) ? value.length : value ? 1 : 0),
    0,
  )
}

export function hasRoutingProfileContent(profile: RoutingProfileLike | undefined): boolean {
  return (
    countSignalsInProfile(profile).total > 0 ||
    countProjectionsInProfile(profile) > 0 ||
    arrayCount(profile?.decisions) > 0
  )
}
