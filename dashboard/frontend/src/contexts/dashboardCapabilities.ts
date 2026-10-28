export const CAPABILITY_IDS = ['setup', 'config', 'evaluation', 'mlPipeline', 'mcp'] as const
export type CapabilityId = (typeof CAPABILITY_IDS)[number]

const CAPABILITY_STATES = [
  'disabled',
  'unconfigured',
  'unauthorized',
  'unreachable',
  'degraded',
  'available',
] as const
export type CapabilityState = (typeof CAPABILITY_STATES)[number]

/** One Dashboard feature's state for the caller, from GET /api/v1/capabilities. */
export interface DashboardCapability {
  id: CapabilityId
  enabled: boolean
  configured: boolean
  reachable: boolean
  authorized: boolean
  degraded: boolean
  state: CapabilityState
  reason: string
  remediation: string
  checkedAt: string | null
}

export type DashboardCapabilities = Record<CapabilityId, DashboardCapability>

/** The server has not finished its first health check for this capability. */
export const PROBE_PENDING = 'probe_pending'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const optionalString = (value: unknown, field: string): string => {
  if (value === undefined) return ''
  if (typeof value !== 'string') throw new Error(`Capability field ${field} must be string`)
  return value
}

const decodeCapability = (value: unknown): DashboardCapability | null => {
  if (!isRecord(value)) throw new Error('Capability must be an object')
  // A newer server may report capabilities this client does not use yet.
  if (!CAPABILITY_IDS.includes(value.id as CapabilityId)) return null
  for (const field of ['enabled', 'configured', 'reachable', 'authorized', 'degraded'] as const) {
    if (typeof value[field] !== 'boolean') {
      throw new Error(`Capability ${String(value.id)} field ${field} must be boolean`)
    }
  }
  if (!CAPABILITY_STATES.includes(value.state as CapabilityState)) {
    throw new Error(`Capability ${String(value.id)} has unknown state ${String(value.state)}`)
  }
  return {
    id: value.id as CapabilityId,
    enabled: value.enabled as boolean,
    configured: value.configured as boolean,
    reachable: value.reachable as boolean,
    authorized: value.authorized as boolean,
    degraded: value.degraded as boolean,
    state: value.state as CapabilityState,
    reason: optionalString(value.reason, 'reason'),
    remediation: optionalString(value.remediation, 'remediation'),
    checkedAt: value.checkedAt === undefined ? null : optionalString(value.checkedAt, 'checkedAt'),
  }
}

/** Decode the same-version backend contract; every capability this client uses must be present. */
export const decodeDashboardCapabilities = (value: unknown): DashboardCapabilities => {
  if (!isRecord(value) || !Array.isArray(value.capabilities)) {
    throw new Error('Dashboard capabilities must be an object with a capabilities array')
  }
  const byID: Partial<DashboardCapabilities> = {}
  for (const entry of value.capabilities) {
    const capability = decodeCapability(entry)
    if (capability) byID[capability.id] = capability
  }
  for (const id of CAPABILITY_IDS) {
    if (!byID[id]) throw new Error(`Dashboard capabilities omit ${id}`)
  }
  return byID as DashboardCapabilities
}

/** A degraded capability still works, with a caveat its remediation explains. */
export const isCapabilityUsable = (capability: DashboardCapability | undefined): boolean =>
  capability?.state === 'available' || capability?.state === 'degraded'
