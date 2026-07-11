import React, { useDeferredValue, useEffect, useMemo, useState } from 'react'

import {
  buildOpenClawTeamComposition,
  filterAndSortOpenClawWorkers,
  getOpenClawPageCount,
  getOpenClawWorkerHealth,
  OPENCLAW_DASHBOARD_ROSTER_PAGE_SIZE,
  OPENCLAW_DASHBOARD_TEAMS_PAGE_SIZE,
  paginateOpenClawItems,
  type WorkerCatalogSort,
  type WorkerHealthFilter,
} from '../utils/openClawCatalogSupport'
import { OpenClawCatalogControls, OpenClawPagination } from './OpenClawCatalogControls'
import styles from './OpenClawPage.module.css'
import { truncateText, type OpenClawStatus, type TeamProfile } from './OpenClawPageSupport'
import { OpenClawRequestNotice } from './OpenClawRequestNotice'

interface DashboardTabProps {
  containers: OpenClawStatus[]
  error?: string | null
  loading?: boolean
  readOnly: boolean
  teams: TeamProfile[]
  onRetry?: () => void
  onSwitchToStatus: () => void
}

export const DashboardTab: React.FC<DashboardTabProps> = ({
  containers,
  teams,
  loading = false,
  error,
  onRetry,
  onSwitchToStatus,
  readOnly,
}) => {
  const [teamPage, setTeamPage] = useState(1)
  const [rosterPage, setRosterPage] = useState(1)
  const [search, setSearch] = useState('')
  const deferredSearch = useDeferredValue(search)
  const [healthFilter, setHealthFilter] = useState<WorkerHealthFilter>('all')
  const [sort, setSort] = useState<WorkerCatalogSort>('name-asc')
  const [teamFilter, setTeamFilter] = useState('all')

  const totalAgents = containers.length
  const healthyAgents = containers.filter((container) => container.healthy).length
  const runningAgents = containers.filter((container) => container.running).length
  const startingAgents = containers.filter(
    (container) => container.running && !container.healthy,
  ).length
  const stoppedAgents = containers.filter((container) => !container.running).length
  const teamCompositionRows = useMemo(
    () => buildOpenClawTeamComposition(teams, containers),
    [containers, teams],
  )
  const teamPageCount = getOpenClawPageCount(
    teamCompositionRows.length,
    OPENCLAW_DASHBOARD_TEAMS_PAGE_SIZE,
  )
  const safeTeamPage = Math.min(teamPage, teamPageCount)
  const visibleTeamRows = paginateOpenClawItems(
    teamCompositionRows,
    safeTeamPage,
    OPENCLAW_DASHBOARD_TEAMS_PAGE_SIZE,
  )
  const filteredRoster = useMemo(
    () => filterAndSortOpenClawWorkers(containers, deferredSearch, healthFilter, teamFilter, sort),
    [containers, deferredSearch, healthFilter, sort, teamFilter],
  )
  const rosterPageCount = getOpenClawPageCount(
    filteredRoster.length,
    OPENCLAW_DASHBOARD_ROSTER_PAGE_SIZE,
  )
  const safeRosterPage = Math.min(rosterPage, rosterPageCount)
  const visibleRoster = paginateOpenClawItems(
    filteredRoster,
    safeRosterPage,
    OPENCLAW_DASHBOARD_ROSTER_PAGE_SIZE,
  )

  const roleRows = useMemo(
    () => buildDistribution(containers, (container) => container.agentRole),
    [containers],
  )
  const teamRows = useMemo(
    () => buildDistribution(containers, (container) => container.teamName),
    [containers],
  )
  const roleMax = Math.max(...roleRows.map(([, value]) => value), 1)
  const teamMax = Math.max(...teamRows.map(([, value]) => value), 1)

  useEffect(() => setRosterPage(1), [deferredSearch, healthFilter, sort, teamFilter])
  useEffect(() => {
    if (teamPage > teamPageCount) setTeamPage(teamPageCount)
  }, [teamPage, teamPageCount])
  useEffect(() => {
    if (rosterPage > rosterPageCount) setRosterPage(rosterPageCount)
  }, [rosterPage, rosterPageCount])

  if (loading && containers.length === 0 && teams.length === 0) {
    return (
      <div className={styles.loading} role="status">
        <div className={styles.spinner} />
        <p>Loading the OpenClaw control plane…</p>
      </div>
    )
  }

  return (
    <div className={styles.teamDashboard}>
      {error ? (
        <OpenClawRequestNotice
          title="OpenClaw dashboard data is incomplete"
          message={error}
          onRetry={onRetry}
        />
      ) : null}

      <section className={styles.teamCompositionSection}>
        <div className={styles.teamCompositionHeader}>
          <h3 className={styles.teamCompositionTitle}>OpenClaw Team Composition</h3>
          <span className={styles.teamPanelSubtitle}>
            Client view · {OPENCLAW_DASHBOARD_TEAMS_PAGE_SIZE} per page
          </span>
        </div>
        <div className={styles.teamCompositionSummary}>
          <span>{teamCompositionRows.length} teams</span>
          <span>{totalAgents} workers</span>
          <span>{healthyAgents} healthy</span>
          <span>{runningAgents} running</span>
        </div>
        {visibleTeamRows.length === 0 ? (
          <div className={styles.teamCompositionEmpty}>
            No OpenClaw team data yet. Provision a team and workers to populate this view.
          </div>
        ) : (
          <div className={styles.teamCompositionGrid}>
            {visibleTeamRows.map((row) => {
              const teamName = row.team?.name || 'Unassigned'
              const previewWorkers = row.workers.slice(0, 5)
              return (
                <article key={row.key} className={styles.teamCompositionCard}>
                  <div className={styles.teamCompositionCardHeader}>
                    <div>
                      <h4 className={styles.teamCompositionCardTitle}>{teamName}</h4>
                      <div className={styles.teamCompositionCardMeta}>
                        {row.team?.role || 'No role'} · {row.team?.vibe || 'No vibe'}
                      </div>
                    </div>
                    <span className={styles.teamCompositionCardCount}>
                      {row.workers.length} worker{row.workers.length === 1 ? '' : 's'}
                    </span>
                  </div>
                  {row.team?.principal ? (
                    <p className={styles.teamCompositionPrincipal}>{row.team.principal}</p>
                  ) : null}
                  <div className={styles.teamCompositionAgentList}>
                    {previewWorkers.length === 0 ? (
                      <div className={styles.teamCompositionAgentEmpty}>No workers assigned.</div>
                    ) : (
                      previewWorkers.map((worker) => (
                        <div key={worker.containerName} className={styles.teamCompositionAgentItem}>
                          <span className={styles.teamCompositionAgentName}>
                            {worker.agentName || worker.containerName}
                          </span>
                          <HealthBadge worker={worker} />
                        </div>
                      ))
                    )}
                    {row.workers.length > previewWorkers.length ? (
                      <div className={styles.teamCompositionAgentEmpty}>
                        +{row.workers.length - previewWorkers.length} more workers
                      </div>
                    ) : null}
                  </div>
                </article>
              )
            })}
          </div>
        )}
        <OpenClawPagination
          itemCount={teamCompositionRows.length}
          itemLabel="teams"
          page={safeTeamPage}
          pageSize={OPENCLAW_DASHBOARD_TEAMS_PAGE_SIZE}
          onPageChange={setTeamPage}
        />
      </section>

      <div className={styles.teamStatsGrid}>
        <StatCard label="Total Workers" value={totalAgents} />
        <StatCard label="Healthy" value={healthyAgents} semantic />
        <StatCard label="Running" value={runningAgents} />
        <StatCard label="Total Teams" value={teams.length} />
      </div>

      <div className={styles.teamChartsGrid}>
        <DistributionPanel
          title="Health Distribution"
          rows={[
            ['Healthy', healthyAgents, 'semantic'],
            ['Starting', startingAgents, 'neutral'],
            ['Stopped', stoppedAgents, 'danger'],
          ]}
          max={Math.max(totalAgents, 1)}
          showPercentage
        />
        <DistributionPanel title="Role Distribution" rows={roleRows} max={roleMax} />
      </div>

      <div className={styles.teamPanelsRow}>
        <DistributionPanel title="Team Distribution" rows={teamRows} max={teamMax} />
        <div className={styles.teamPanel}>
          <div className={styles.teamPanelHeader}>
            <h3 className={styles.teamPanelTitle}>Quick Action</h3>
            <span className={styles.teamPanelSubtitle}>Control Plane</span>
          </div>
          <p className={styles.panelText}>
            {readOnly
              ? 'Review worker health and runtime status without lifecycle controls.'
              : 'Use Claw Dashboard for lifecycle actions, logs, and embedded control UI.'}
          </p>
          <button type="button" className={styles.btnPrimary} onClick={onSwitchToStatus}>
            {readOnly ? 'View Claw Status' : 'Open Claw Dashboard'}
          </button>
        </div>
      </div>

      <section className={styles.teamPanel}>
        <div className={styles.teamPanelHeader}>
          <h3 className={styles.teamPanelTitle}>Team Roster</h3>
          <span className={styles.teamPanelSubtitle}>{totalAgents} workers</span>
        </div>
        <OpenClawCatalogControls
          searchLabel="Search roster"
          searchValue={search}
          filterLabel="Health"
          filterValue={healthFilter}
          filterOptions={[
            { value: 'all', label: 'All health states' },
            { value: 'healthy', label: 'Healthy' },
            { value: 'starting', label: 'Starting' },
            { value: 'stopped', label: 'Stopped' },
          ]}
          sortValue={sort}
          sortOptions={[
            { value: 'name-asc', label: 'Name A–Z' },
            { value: 'team-asc', label: 'Team A–Z' },
            { value: 'status', label: 'Health status' },
            { value: 'created-desc', label: 'Recently created' },
          ]}
          itemCount={filteredRoster.length}
          totalCount={containers.length}
          itemLabel="workers"
          page={safeRosterPage}
          pageSize={OPENCLAW_DASHBOARD_ROSTER_PAGE_SIZE}
          onSearchChange={setSearch}
          onFilterChange={(value) => setHealthFilter(value as WorkerHealthFilter)}
          onSortChange={(value) => setSort(value as WorkerCatalogSort)}
          onPageChange={setRosterPage}
        />
        <label className={styles.enterpriseTeamFilter}>
          <span>Team</span>
          <select value={teamFilter} onChange={(event) => setTeamFilter(event.target.value)}>
            <option value="all">All teams</option>
            <option value="unassigned">Unassigned</option>
            {teams.map((team) => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
        </label>
        {visibleRoster.length === 0 ? (
          <div className={styles.emptyState}>
            <div className={styles.emptyStateText}>
              {containers.length === 0
                ? 'No workers are provisioned.'
                : 'No workers match the current roster filters.'}
            </div>
          </div>
        ) : (
          <div className={styles.agentGrid}>
            {visibleRoster.map((worker) => (
              <article key={worker.containerName} className={styles.agentCard}>
                <div className={styles.agentCardHeader}>
                  <div className={styles.agentAvatar}>{worker.agentEmoji?.trim() || '🧠'}</div>
                  <div className={styles.agentHeaderMeta}>
                    <div className={styles.agentName}>
                      {worker.agentName?.trim() || worker.containerName}
                    </div>
                    <div className={styles.agentContainerRef}>{worker.containerName}</div>
                    <div className={styles.teamTag}>{worker.teamName?.trim() || 'Unassigned'}</div>
                  </div>
                  <HealthBadge worker={worker} />
                </div>
                <div className={styles.agentBody}>
                  <RosterFact label="Role" value={worker.agentRole?.trim() || 'Not set'} />
                  <RosterFact label="Vibe" value={worker.agentVibe?.trim() || 'Not set'} />
                  <RosterFact
                    label="Principal"
                    value={truncateText(worker.agentPrinciples, 160) || 'Not set'}
                  />
                </div>
                <div className={styles.agentFooter}>
                  <button type="button" className={styles.btnSmall} onClick={onSwitchToStatus}>
                    {readOnly ? 'View status' : 'Manage'}
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

function StatCard({
  label,
  value,
  semantic = false,
}: {
  label: string
  value: number
  semantic?: boolean
}) {
  return (
    <div className={styles.teamStatCard}>
      <div className={`${styles.teamStatValue} ${semantic ? styles.semanticStatusText : ''}`}>
        {value}
      </div>
      <div className={styles.teamStatLabel}>{label}</div>
    </div>
  )
}

function HealthBadge({ worker }: { worker: OpenClawStatus }) {
  const health = getOpenClawWorkerHealth(worker)
  return (
    <span className={`${styles.healthBadge} ${styles[`healthBadge_${health}`]}`}>
      {health === 'healthy' ? 'Healthy' : health === 'starting' ? 'Starting' : 'Stopped'}
    </span>
  )
}

function RosterFact({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.agentMetaRow}>
      <span className={styles.agentMetaLabel}>{label}</span>
      <span className={styles.agentMetaValue}>{value}</span>
    </div>
  )
}

type DistributionRow = [string, number, string?]

function DistributionPanel({
  title,
  rows,
  max,
  showPercentage = false,
}: {
  title: string
  rows: DistributionRow[]
  max: number
  showPercentage?: boolean
}) {
  return (
    <div className={styles.teamPanel}>
      <div className={styles.teamPanelHeader}>
        <h3 className={styles.teamPanelTitle}>{title}</h3>
        <span className={styles.teamPanelSubtitle}>Top {Math.min(rows.length, 5)}</span>
      </div>
      <div className={styles.breakdownList}>
        {rows.length === 0 ? (
          <div className={styles.teamCompositionAgentEmpty}>No data available.</div>
        ) : (
          rows.map(([label, value, tone]) => {
            const percentage = max > 0 ? Math.round((value / max) * 100) : 0
            return (
              <div key={label} className={styles.breakdownRow}>
                <div className={styles.breakdownLabel}>{label}</div>
                <div className={styles.breakdownTrack}>
                  <div
                    className={`${styles.breakdownBar} ${tone ? styles[`breakdownBar_${tone}`] : ''}`}
                    style={{ width: `${Math.max(value > 0 ? 8 : 0, percentage)}%` }}
                  />
                </div>
                <div className={styles.breakdownValue}>
                  {value}
                  {showPercentage ? ` (${percentage}%)` : ''}
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

function buildDistribution(
  containers: OpenClawStatus[],
  getValue: (container: OpenClawStatus) => string | undefined,
): Array<[string, number]> {
  const counts = new Map<string, number>()
  for (const container of containers) {
    const value = getValue(container)?.trim() || 'Unassigned'
    counts.set(value, (counts.get(value) || 0) + 1)
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1]).slice(0, 5)
}
