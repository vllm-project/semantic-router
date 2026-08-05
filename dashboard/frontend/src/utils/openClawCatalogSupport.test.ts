import { describe, expect, it } from 'vitest'

import {
  buildTeamRuntimeStats,
  filterAndSortOpenClawRooms,
  filterAndSortOpenClawTeams,
  filterAndSortOpenClawWorkers,
  getOpenClawPageCount,
  getOpenClawVisibleRange,
  paginateOpenClawItems,
} from './openClawCatalogSupport'

describe('OpenClaw catalog support', () => {
  const workers = Array.from({ length: 31 }, (_, index) => ({
    containerName: `worker-${String(index + 1).padStart(2, '0')}`,
    agentName: index === 20 ? 'Atlas Search' : `Agent ${index + 1}`,
    teamId: index % 3 === 0 ? 'alpha' : index % 3 === 1 ? 'beta' : undefined,
    teamName: index % 3 === 0 ? 'Alpha' : index % 3 === 1 ? 'Beta' : undefined,
    running: index % 4 !== 0,
    healthy: index % 4 === 1,
    createdAt: `2026-07-${String((index % 28) + 1).padStart(2, '0')}T00:00:00Z`,
  }))

  it('searches and filters large worker collections before bounded paging', () => {
    const healthyBeta = filterAndSortOpenClawWorkers(workers, '', 'healthy', 'beta', 'name-asc')
    expect(healthyBeta.length).toBeGreaterThan(1)
    expect(healthyBeta.every((worker) => worker.healthy && worker.teamId === 'beta')).toBe(true)

    const searchResult = filterAndSortOpenClawWorkers(
      workers,
      'atlas search',
      'all',
      'all',
      'name-asc',
    )
    expect(searchResult.map((worker) => worker.containerName)).toEqual(['worker-21'])

    expect(paginateOpenClawItems(workers, 2, 12)).toHaveLength(12)
    expect(paginateOpenClawItems(workers, 3, 12)).toHaveLength(7)
    expect(getOpenClawPageCount(workers.length, 12)).toBe(3)
    expect(getOpenClawVisibleRange(workers.length, 3, 12)).toEqual({ start: 25, end: 31 })
  })

  it('uses runtime counts for team filtering and deterministic sorting', () => {
    const teams = [
      { id: 'empty', name: 'Empty Team' },
      { id: 'beta', name: 'Beta Team' },
      { id: 'alpha', name: 'Alpha Team' },
    ]
    const stats = buildTeamRuntimeStats(workers)

    expect(filterAndSortOpenClawTeams(teams, stats, '', 'empty', 'name-asc')).toEqual([teams[0]])
    expect(
      filterAndSortOpenClawTeams(teams, stats, '', 'with-workers', 'workers-desc').map(
        (team) => team.id,
      ),
    ).toEqual(['alpha', 'beta'])
  })

  it('supports selected-first room sorting without mutating the source', () => {
    const rooms = [
      { id: 'room-c', name: 'Charlie' },
      { id: 'room-a', name: 'Alpha' },
      { id: 'room-b', name: 'Beta' },
    ]
    const result = filterAndSortOpenClawRooms(rooms, 'room-b', '', 'all', 'selected-first')

    expect(result.map((room) => room.id)).toEqual(['room-b', 'room-a', 'room-c'])
    expect(rooms.map((room) => room.id)).toEqual(['room-c', 'room-a', 'room-b'])
    expect(filterAndSortOpenClawRooms(rooms, 'room-b', 'char', 'other', 'name-asc')).toEqual([
      rooms[0],
    ])
  })
})
