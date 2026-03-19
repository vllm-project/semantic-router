import React, { useEffect, useMemo, useRef, useState } from 'react'
import styles from './OpenClawPage.module.css'
import {
  truncateText,
  type OpenClawStatus,
  type TeamProfile,
} from './OpenClawPageSupport'

export { ArchitectureTab } from './OpenClawArchitectureTab'
export { TeamTab } from './OpenClawTeamTab'
export { WorkerTab } from './OpenClawWorkerTab'

export const DashboardTab: React.FC<{
  containers: OpenClawStatus[]
  teams: TeamProfile[]
  onSwitchToStatus: () => void
}> = ({ containers, teams, onSwitchToStatus }) => {
  const totalAgents = containers.length
  const totalTeams = teams.length
  const healthyAgents = containers.filter(c => c.healthy).length
  const runningAgents = containers.filter(c => c.running).length
  const startingAgents = containers.filter(c => c.running && !c.healthy).length
  const stoppedAgents = containers.filter(c => !c.running).length

  const roleRows = useMemo(() => {
    const counts = new Map<string, number>()
    for (const c of containers) {
      const role = (c.agentRole || '').trim() || 'Unspecified'
      counts.set(role, (counts.get(role) || 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
  }, [containers])

  const roleMax = Math.max(...roleRows.map(([, value]) => value), 1)
  const teamRows = useMemo(() => {
    const counts = new Map<string, number>()
    for (const c of containers) {
      const team = (c.teamName || '').trim() || 'Unassigned'
      counts.set(team, (counts.get(team) || 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
  }, [containers])
  const teamMax = Math.max(...teamRows.map(([, value]) => value), 1)
  const teamCompositionRows = useMemo(() => {
    const rows = new Map<string, { team: TeamProfile | null; agents: OpenClawStatus[] }>()
    for (const team of teams) {
      rows.set(team.id, { team, agents: [] })
    }
    for (const agent of containers) {
      const key = (agent.teamId || '').trim() || '__unassigned__'
      if (!rows.has(key)) {
        rows.set(key, {
          team: key === '__unassigned__'
            ? null
            : {
                id: key,
                name: (agent.teamName || key).trim() || key,
                vibe: '',
                role: '',
                principal: '',
              },
          agents: [],
        })
      }
      rows.get(key)?.agents.push(agent)
    }
    return Array.from(rows.values())
      .filter(row => row.team !== null || row.agents.length > 0)
      .sort((a, b) => b.agents.length - a.agents.length)
  }, [teams, containers])

  const teamComposition = (
    <section className={styles.teamCompositionSection}>
      <div className={styles.teamCompositionHeader}>
        <h3 className={styles.teamCompositionTitle}>OpenClaw Team Composition</h3>
      </div>
      <div className={styles.teamCompositionSummary}>
        <span>{teamCompositionRows.length} teams</span>
        <span>{totalAgents} agents</span>
        <span>{healthyAgents} healthy</span>
        <span>{runningAgents} running</span>
      </div>
      {teamCompositionRows.length === 0 ? (
        <div className={styles.teamCompositionEmpty}>
          No OpenClaw team data yet. Provision a team and agents to populate this view.
        </div>
      ) : (
        <div className={styles.teamCompositionGrid}>
          {teamCompositionRows.map((row, index) => {
            const team = row.team
            const teamName = team?.name || 'Unassigned'
            return (
              <article key={`${team?.id || 'unassigned'}-${index}`} className={styles.teamCompositionCard}>
                <div className={styles.teamCompositionCardHeader}>
                  <div>
                    <h4 className={styles.teamCompositionCardTitle}>{teamName}</h4>
                    <div className={styles.teamCompositionCardMeta}>
                      {team?.role || 'No role'} • {team?.vibe || 'No vibe'}
                    </div>
                  </div>
                  <span className={styles.teamCompositionCardCount}>
                    {row.agents.length} agent{row.agents.length !== 1 ? 's' : ''}
                  </span>
                </div>
                {team?.principal && (
                  <p className={styles.teamCompositionPrincipal}>{team.principal}</p>
                )}
                <div className={styles.teamCompositionAgentList}>
                  {row.agents.length === 0 ? (
                    <div className={styles.teamCompositionAgentEmpty}>No agents assigned.</div>
                  ) : row.agents.map(agent => (
                    <div key={agent.containerName} className={styles.teamCompositionAgentItem}>
                      <span className={styles.teamCompositionAgentName}>{agent.agentName || agent.containerName}</span>
                      <span className={`${styles.healthBadge} ${
                        agent.healthy
                          ? styles.healthBadgeHealthy
                          : agent.running
                            ? styles.healthBadgeRunning
                            : styles.healthBadgeStopped
                      }`}>
                        {agent.healthy ? 'Healthy' : agent.running ? 'Starting' : 'Stopped'}
                      </span>
                    </div>
                  ))}
                </div>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )

  if (containers.length === 0) {
    return <div className={styles.teamDashboard}>{teamComposition}</div>
  }

  return (
    <div className={styles.teamDashboard}>
      {teamComposition}
      <div className={styles.teamStatsGrid}>
        <div className={styles.teamStatCard}>
          <div className={styles.teamStatValue}>{totalAgents}</div>
          <div className={styles.teamStatLabel}>Total Agents</div>
        </div>
        <div className={styles.teamStatCard}>
          <div className={styles.teamStatValue}>{healthyAgents}</div>
          <div className={styles.teamStatLabel}>Healthy</div>
        </div>
        <div className={styles.teamStatCard}>
          <div className={styles.teamStatValue}>{runningAgents}</div>
          <div className={styles.teamStatLabel}>Running</div>
        </div>
        <div className={styles.teamStatCard}>
          <div className={styles.teamStatValue}>{totalTeams}</div>
          <div className={styles.teamStatLabel}>Total Teams</div>
        </div>
      </div>

      <div className={styles.teamChartsGrid}>
        <div className={styles.teamPanel}>
          <div className={styles.teamPanelHeader}>
            <h3 className={styles.teamPanelTitle}>Health Distribution</h3>
            <span className={styles.teamPanelSubtitle}>Realtime</span>
          </div>
          <div className={styles.breakdownList}>
            {[
              ['Healthy', healthyAgents, '#22c55e'],
              ['Starting', startingAgents, '#eab308'],
              ['Stopped', stoppedAgents, '#ef4444'],
            ].map(([label, value, color]) => {
              const numericValue = Number(value)
              const pct = totalAgents > 0 ? Math.round((numericValue / totalAgents) * 100) : 0
              return (
                <div key={String(label)} className={styles.breakdownRow}>
                  <div className={styles.breakdownLabel}>{label}</div>
                  <div className={styles.breakdownTrack}>
                    <div
                      className={styles.breakdownBar}
                      style={{ width: `${Math.max(8, pct)}%`, backgroundColor: String(color) }}
                    />
                  </div>
                  <div className={styles.breakdownValue}>{numericValue} ({pct}%)</div>
                </div>
              )
            })}
          </div>
        </div>
        <div className={styles.teamPanel}>
          <div className={styles.teamPanelHeader}>
            <h3 className={styles.teamPanelTitle}>Role Distribution</h3>
            <span className={styles.teamPanelSubtitle}>Top 5</span>
          </div>
          <div className={styles.breakdownList}>
            {roleRows.map(([role, value]) => (
              <div key={role} className={styles.breakdownRow}>
                <div className={styles.breakdownLabel}>{role}</div>
                <div className={styles.breakdownTrack}>
                  <div className={styles.breakdownBar} style={{ width: `${Math.max(10, Math.round((value / roleMax) * 100))}%` }} />
                </div>
                <div className={styles.breakdownValue}>{value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.teamPanelsRow}>
        <div className={styles.teamPanel}>
          <div className={styles.teamPanelHeader}>
            <h3 className={styles.teamPanelTitle}>Team Distribution</h3>
            <span className={styles.teamPanelSubtitle}>Top 5</span>
          </div>
          <div className={styles.breakdownList}>
            {teamRows.map(([team, value]) => (
              <div key={team} className={styles.breakdownRow}>
                <div className={styles.breakdownLabel}>{team}</div>
                <div className={styles.breakdownTrack}>
                  <div className={styles.breakdownBar} style={{ width: `${Math.max(10, Math.round((value / teamMax) * 100))}%` }} />
                </div>
                <div className={styles.breakdownValue}>{value}</div>
              </div>
            ))}
          </div>
        </div>
        <div className={styles.teamPanel}>
          <div className={styles.teamPanelHeader}>
            <h3 className={styles.teamPanelTitle}>Quick Action</h3>
            <span className={styles.teamPanelSubtitle}>Control Plane</span>
          </div>
          <p className={styles.panelText}>
            Use Claw Dashboard for lifecycle actions, logs, and embedded control UI.
          </p>
          <button className={styles.btnPrimary} onClick={onSwitchToStatus}>
            Open Claw Dashboard
          </button>
        </div>
      </div>

      <div className={styles.teamPanel}>
        <div className={styles.teamPanelHeader}>
          <h3 className={styles.teamPanelTitle}>Team Roster</h3>
          <span className={styles.teamPanelSubtitle}>{totalAgents} agents</span>
        </div>
        <div className={styles.agentGrid}>
          {containers.map((agent) => {
            const name = agent.agentName?.trim() || agent.containerName
            const emoji = agent.agentEmoji?.trim() || '\u{1F9E0}'
            const role = agent.agentRole?.trim() || 'Not set'
            const vibe = agent.agentVibe?.trim() || 'Not set'
            const principles = truncateText(agent.agentPrinciples, 160) || 'Not set'

            return (
              <div key={agent.containerName} className={styles.agentCard}>
                <div className={styles.agentCardHeader}>
                  <div className={styles.agentAvatar}>{emoji}</div>
                  <div className={styles.agentHeaderMeta}>
                    <div className={styles.agentName}>{name}</div>
                    <div className={styles.agentContainerRef}>{agent.containerName}</div>
                    <div className={styles.teamTag}>{agent.teamName?.trim() || 'Unassigned'}</div>
                  </div>
                  <span
                    className={`${styles.healthBadge} ${
                      agent.healthy
                        ? styles.healthBadgeHealthy
                        : agent.running
                          ? styles.healthBadgeRunning
                          : styles.healthBadgeStopped
                    }`}
                  >
                    {agent.healthy ? 'Healthy' : agent.running ? 'Starting' : 'Stopped'}
                  </span>
                </div>

                <div className={styles.agentBody}>
                  <div className={styles.agentMetaRow}>
                    <span className={styles.agentMetaLabel}>Role</span>
                    <span className={styles.agentMetaValue}>{role}</span>
                  </div>
                  <div className={styles.agentMetaRow}>
                    <span className={styles.agentMetaLabel}>Vibe</span>
                    <span className={styles.agentMetaValue}>{vibe}</span>
                  </div>
                  <div className={styles.agentMetaRow}>
                    <span className={styles.agentMetaLabel}>Team</span>
                    <span className={styles.agentMetaValue}>{agent.teamName?.trim() || 'Unassigned'}</span>
                  </div>
                  <div className={styles.agentMetaRow}>
                    <span className={styles.agentMetaLabel}>Principal</span>
                    <span className={styles.agentMetaValue}>{principles}</span>
                  </div>
                </div>

                <div className={styles.agentFooter}>
                  <button className={styles.btnSmall} onClick={onSwitchToStatus}>
                    Manage
                  </button>
                  {agent.createdAt && (
                    <span className={styles.agentTimestamp}>
                      Created {new Date(agent.createdAt).toLocaleString()}
                    </span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export const StatusTab: React.FC<{
  containers: OpenClawStatus[]
  statusLoading: boolean
  onRefresh: () => void
  readOnly: boolean
}> = ({ containers, statusLoading, onRefresh, readOnly }) => {
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [actionError, setActionError] = useState('')
  const [selectedContainer, setSelectedContainer] = useState<string | null>(null)
  const [gatewayToken, setGatewayToken] = useState('')
  const [isFullscreen, setIsFullscreen] = useState(false)
  const iframeContainerRef = useRef<HTMLDivElement | null>(null)

  const selected = containers.find(c => c.containerName === selectedContainer)

  useEffect(() => {
    if (selected?.healthy && selectedContainer) {
      setGatewayToken('')
      fetch(`/api/openclaw/token?name=${encodeURIComponent(selectedContainer)}`)
        .then(r => r.json())
        .then(d => { if (d.token) setGatewayToken(d.token) })
        .catch(() => {})
    }
  }, [selected?.healthy, selectedContainer])

  useEffect(() => {
    const onFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement))
    }
    document.addEventListener('fullscreenchange', onFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', onFullscreenChange)
  }, [])

  useEffect(() => {
    if (readOnly && selectedContainer) {
      setSelectedContainer(null)
    }
  }, [readOnly, selectedContainer])

  const handleAction = async (action: 'start' | 'stop', name: string) => {
    if (readOnly) return
    setActionLoading(name)
    setActionError('')
    try {
      const res = await fetch(`/api/openclaw/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ containerName: name }),
      })
      if (!res.ok) {
        const data = await res.json()
        setActionError(data.error || `Failed to ${action}`)
      }
      setTimeout(onRefresh, 2000)
    } catch (e) {
      setActionError(String(e))
    } finally {
      setActionLoading(null)
    }
  }

  const handleDelete = async (name: string) => {
    if (readOnly) return
    if (!confirm(`Remove container "${name}"? This will stop and remove the Docker container.`)) return
    setActionLoading(name)
    setActionError('')
    try {
      const res = await fetch(`/api/openclaw/containers/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (!res.ok) {
        const data = await res.json()
        setActionError(data.error || 'Failed to remove')
      }
      if (selectedContainer === name) setSelectedContainer(null)
      setTimeout(onRefresh, 1000)
    } catch (e) {
      setActionError(String(e))
    } finally {
      setActionLoading(null)
    }
  }

  if (statusLoading) {
    return (
      <div className={styles.loading}>
        <div className={styles.spinner} />
        <p>Checking OpenClaw containers...</p>
      </div>
    )
  }

  if (!readOnly && selectedContainer && selected?.healthy) {
    if (!gatewayToken) {
      return (
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Connecting to gateway...</p>
        </div>
      )
    }
    const proxyBase = `/embedded/openclaw/${encodeURIComponent(selectedContainer)}/`
    const gatewayUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}${proxyBase}`
    const iframeSrc = `${proxyBase}#gatewayUrl=${encodeURIComponent(gatewayUrl)}&token=${encodeURIComponent(gatewayToken)}`
    const openInNewTab = () => {
      window.open(iframeSrc, '_blank', 'noopener,noreferrer')
    }
    const toggleFullscreen = async () => {
      try {
        if (!document.fullscreenElement) {
          if (iframeContainerRef.current?.requestFullscreen) {
            await iframeContainerRef.current.requestFullscreen()
            return
          }
        } else {
          await document.exitFullscreen()
          return
        }
      } catch {
        openInNewTab()
        return
      }
      openInNewTab()
    }

    return (
      <div>
        <div className={styles.embeddedHeader}>
          <div className={styles.embeddedHeaderLeft}>
            <button className={styles.btnSecondary} onClick={() => setSelectedContainer(null)} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 18 9 12 15 6" />
              </svg>
              Back to Claw Dashboard
            </button>
            <span style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)' }}>
              {selected.containerName} &mdash; port {selected.port}
            </span>
          </div>
          <button className={styles.btnSecondary} onClick={toggleFullscreen} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M8 3H5a2 2 0 0 0-2 2v3" />
              <path d="M16 3h3a2 2 0 0 1 2 2v3" />
              <path d="M8 21H5a2 2 0 0 1-2-2v-3" />
              <path d="M16 21h3a2 2 0 0 0 2-2v-3" />
            </svg>
            {isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          </button>
        </div>
        <div ref={iframeContainerRef} className={styles.iframeContainer}>
          <iframe
            key={gatewayToken}
            className={styles.iframe}
            src={iframeSrc}
            title={`OpenClaw Control UI — ${selectedContainer}`}
            sandbox="allow-same-origin allow-scripts allow-forms allow-popups"
          />
        </div>
      </div>
    )
  }

  return (
    <div>
      {actionError && (
        <div className={styles.errorAlert}>
          <span>{actionError}</span>
          <button onClick={() => setActionError('')} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '1rem' }}>
            &times;
          </button>
        </div>
      )}

      {containers.length === 0 ? (
        <>
          <div className={styles.emptyState}>
            <div className={styles.emptyStateIcon}>{'\u{1F433}'}</div>
            <div className={styles.emptyStateText}>
              No OpenClaw containers provisioned yet.<br />
              Use the <strong>Claw Provision</strong> tab to create one.
            </div>
          </div>
          <div className={styles.statusActionsCentered}>
            <button className={styles.btnSecondary} onClick={onRefresh}>
              Refresh Status
            </button>
          </div>
        </>
      ) : (
        <>
          <table className={styles.containerTable}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Team</th>
                <th>Port</th>
                <th>Health</th>
                <th>Error</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {containers.map(c => (
                <tr key={c.containerName}>
                  <td className={styles.containerTableName}>{c.containerName}</td>
                  <td>{c.teamName?.trim() || 'Unassigned'}</td>
                  <td className={styles.containerTablePort}>{c.port}</td>
                  <td>
                    <span className={`${styles.healthBadge} ${
                      c.healthy ? styles.healthBadgeHealthy :
                      c.running ? styles.healthBadgeRunning :
                      styles.healthBadgeStopped
                    }`}>
                      {c.healthy ? 'Healthy' : c.running ? 'Starting' : 'Stopped'}
                    </span>
                  </td>
                  <td style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', maxWidth: '240px' }}>
                    {c.error || '\u2014'}
                  </td>
                  <td>
                    <div className={styles.containerActions}>
                      {c.healthy && (
                        <button
                          className={`${styles.btnSmall} ${styles.btnSmallPrimary}`}
                          onClick={() => setSelectedContainer(c.containerName)}
                          disabled={readOnly}
                        >
                          Dashboard
                        </button>
                      )}
                      {c.running ? (
                        <button
                          className={styles.btnSmall}
                          onClick={() => handleAction('stop', c.containerName)}
                          disabled={readOnly || actionLoading === c.containerName}
                        >
                          Stop
                        </button>
                      ) : (
                        <button
                          className={styles.btnSmall}
                          onClick={() => handleAction('start', c.containerName)}
                          disabled={readOnly || actionLoading === c.containerName}
                        >
                          Start
                        </button>
                      )}
                      <button
                        className={`${styles.btnSmall} ${styles.btnSmallDanger}`}
                        onClick={() => handleDelete(c.containerName)}
                        disabled={readOnly || actionLoading === c.containerName}
                      >
                        Remove
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className={styles.statusActions}>
            <button className={styles.btnSecondary} onClick={onRefresh}>
              Refresh Status
            </button>
          </div>
        </>
      )}
    </div>
  )
}
