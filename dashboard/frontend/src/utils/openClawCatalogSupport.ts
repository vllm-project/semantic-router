export const OPENCLAW_TEAMS_PAGE_SIZE = 12
export const OPENCLAW_WORKERS_PAGE_SIZE = 12
export const OPENCLAW_CONTAINERS_PAGE_SIZE = 15
export const OPENCLAW_DASHBOARD_TEAMS_PAGE_SIZE = 6
export const OPENCLAW_DASHBOARD_ROSTER_PAGE_SIZE = 12
export const OPENCLAW_ROOMS_PAGE_SIZE = 8

export interface OpenClawTeamLike {
  id: string
  name: string
  role?: string
  vibe?: string
  principal?: string
  description?: string
  createdAt?: string
  updatedAt?: string
}

export interface OpenClawWorkerLike {
  containerName: string
  running: boolean
  healthy: boolean
  port?: number
  error?: string
  teamId?: string
  teamName?: string
  agentName?: string
  agentRole?: string
  agentVibe?: string
  agentPrinciples?: string
  createdAt?: string
}

export interface OpenClawRoomLike {
  id: string
  name: string
}

export interface TeamRuntimeStats {
  total: number
  running: number
  healthy: number
}

export type TeamCatalogFilter = 'all' | 'with-workers' | 'empty'
export type TeamCatalogSort = 'name-asc' | 'workers-desc' | 'running-desc' | 'updated-desc'
export type WorkerHealthFilter = 'all' | 'healthy' | 'starting' | 'stopped'
export type WorkerCatalogSort = 'name-asc' | 'team-asc' | 'status' | 'created-desc'
export type RoomCatalogFilter = 'all' | 'selected' | 'other'
export type RoomCatalogSort = 'name-asc' | 'id-asc' | 'selected-first'

export function buildTeamRuntimeStats<T extends OpenClawWorkerLike>(
  workers: readonly T[],
): Map<string, TeamRuntimeStats> {
  const stats = new Map<string, TeamRuntimeStats>()
  for (const worker of workers) {
    const key = worker.teamId?.trim() || '__unassigned__'
    const current = stats.get(key) || { total: 0, running: 0, healthy: 0 }
    current.total += 1
    if (worker.running) current.running += 1
    if (worker.healthy) current.healthy += 1
    stats.set(key, current)
  }
  return stats
}

export function filterAndSortOpenClawTeams<T extends OpenClawTeamLike>(
  teams: readonly T[],
  stats: ReadonlyMap<string, TeamRuntimeStats>,
  searchValue: string,
  filter: TeamCatalogFilter,
  sort: TeamCatalogSort,
): T[] {
  const search = searchValue.trim().toLowerCase()
  return teams
    .filter((team) => {
      const workerCount = stats.get(team.id)?.total || 0
      if (filter === 'with-workers') return workerCount > 0
      if (filter === 'empty') return workerCount === 0
      return true
    })
    .filter((team) => {
      if (!search) return true
      return [
        team.id,
        team.name,
        team.role || '',
        team.vibe || '',
        team.principal || '',
        team.description || '',
      ].some((value) => value.toLowerCase().includes(search))
    })
    .sort((left, right) => {
      const leftStats = stats.get(left.id) || { total: 0, running: 0, healthy: 0 }
      const rightStats = stats.get(right.id) || { total: 0, running: 0, healthy: 0 }
      if (sort === 'workers-desc') {
        return rightStats.total - leftStats.total || compareName(left.name, right.name)
      }
      if (sort === 'running-desc') {
        return rightStats.running - leftStats.running || compareName(left.name, right.name)
      }
      if (sort === 'updated-desc') {
        return (
          toTimestamp(right.updatedAt || right.createdAt) -
            toTimestamp(left.updatedAt || left.createdAt) || compareName(left.name, right.name)
        )
      }
      return compareName(left.name, right.name)
    })
}

export function getOpenClawWorkerHealth(
  worker: OpenClawWorkerLike,
): Exclude<WorkerHealthFilter, 'all'> {
  if (worker.healthy) return 'healthy'
  if (worker.running) return 'starting'
  return 'stopped'
}

export function filterAndSortOpenClawWorkers<T extends OpenClawWorkerLike>(
  workers: readonly T[],
  searchValue: string,
  healthFilter: WorkerHealthFilter,
  teamFilter: string,
  sort: WorkerCatalogSort,
): T[] {
  const search = searchValue.trim().toLowerCase()
  return workers
    .filter((worker) => healthFilter === 'all' || getOpenClawWorkerHealth(worker) === healthFilter)
    .filter((worker) => {
      if (teamFilter === 'all') return true
      if (teamFilter === 'unassigned') return !worker.teamId?.trim()
      return worker.teamId === teamFilter
    })
    .filter((worker) => {
      if (!search) return true
      return [
        worker.containerName,
        worker.agentName || '',
        worker.teamName || '',
        worker.teamId || '',
        worker.agentRole || '',
        worker.agentVibe || '',
        worker.agentPrinciples || '',
        worker.error || '',
        String(worker.port || ''),
      ].some((value) => value.toLowerCase().includes(search))
    })
    .sort((left, right) => {
      if (sort === 'team-asc') {
        return (
          compareName(left.teamName || 'Unassigned', right.teamName || 'Unassigned') ||
          compareWorkerName(left, right)
        )
      }
      if (sort === 'status') {
        const rank = { healthy: 0, starting: 1, stopped: 2 }
        return (
          rank[getOpenClawWorkerHealth(left)] - rank[getOpenClawWorkerHealth(right)] ||
          compareWorkerName(left, right)
        )
      }
      if (sort === 'created-desc') {
        return (
          toTimestamp(right.createdAt) - toTimestamp(left.createdAt) ||
          compareWorkerName(left, right)
        )
      }
      return compareWorkerName(left, right)
    })
}

export function filterAndSortOpenClawRooms<T extends OpenClawRoomLike>(
  rooms: readonly T[],
  selectedRoomId: string,
  searchValue: string,
  filter: RoomCatalogFilter,
  sort: RoomCatalogSort,
): T[] {
  const search = searchValue.trim().toLowerCase()
  return rooms
    .filter((room) => {
      if (filter === 'selected') return room.id === selectedRoomId
      if (filter === 'other') return room.id !== selectedRoomId
      return true
    })
    .filter(
      (room) =>
        !search ||
        room.name.toLowerCase().includes(search) ||
        room.id.toLowerCase().includes(search),
    )
    .sort((left, right) => {
      if (sort === 'selected-first') {
        const selectedRank =
          Number(right.id === selectedRoomId) - Number(left.id === selectedRoomId)
        return selectedRank || compareName(left.name, right.name)
      }
      if (sort === 'id-asc') return compareName(left.id, right.id)
      return compareName(left.name, right.name)
    })
}

export interface OpenClawTeamCompositionRow<
  TTeam extends OpenClawTeamLike,
  TWorker extends OpenClawWorkerLike,
> {
  key: string
  team: TTeam | null
  workers: TWorker[]
}

export function buildOpenClawTeamComposition<
  TTeam extends OpenClawTeamLike,
  TWorker extends OpenClawWorkerLike,
>(
  teams: readonly TTeam[],
  workers: readonly TWorker[],
): OpenClawTeamCompositionRow<TTeam, TWorker>[] {
  const rows = new Map<string, OpenClawTeamCompositionRow<TTeam, TWorker>>()
  for (const team of teams) rows.set(team.id, { key: team.id, team, workers: [] })
  for (const worker of workers) {
    const key = worker.teamId?.trim() || '__unassigned__'
    if (!rows.has(key)) rows.set(key, { key, team: null, workers: [] })
    rows.get(key)?.workers.push(worker)
  }
  return [...rows.values()]
    .filter((row) => row.team !== null || row.workers.length > 0)
    .sort(
      (left, right) =>
        right.workers.length - left.workers.length ||
        compareName(left.team?.name || 'Unassigned', right.team?.name || 'Unassigned'),
    )
}

export function paginateOpenClawItems<T>(items: readonly T[], page: number, pageSize: number): T[] {
  const safePage = Math.max(1, page)
  return items.slice((safePage - 1) * pageSize, safePage * pageSize)
}

export function getOpenClawPageCount(itemCount: number, pageSize: number): number {
  return Math.max(1, Math.ceil(itemCount / pageSize))
}

export function getOpenClawVisibleRange(itemCount: number, page: number, pageSize: number) {
  if (itemCount === 0) return { start: 0, end: 0 }
  const start = (Math.max(page, 1) - 1) * pageSize + 1
  return { start, end: Math.min(itemCount, start + pageSize - 1) }
}

function compareName(left: string, right: string): number {
  return left.localeCompare(right, undefined, { sensitivity: 'base' })
}

function compareWorkerName(left: OpenClawWorkerLike, right: OpenClawWorkerLike): number {
  return compareName(left.agentName || left.containerName, right.agentName || right.containerName)
}

function toTimestamp(value?: string): number {
  const timestamp = value ? Date.parse(value) : 0
  return Number.isFinite(timestamp) ? timestamp : 0
}
