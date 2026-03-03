import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import styles from './OpenClawPage.module.css'

// --- Types ---

interface SkillTemplate {
  id: string
  name: string
  description: string
  emoji: string
  category: string
  builtin: boolean
}

interface IdentityConfig {
  name: string
  emoji: string
  role: string
  vibe: string
  principles: string
  boundaries: string
}

interface ContainerConfig {
  containerName: string
  gatewayPort: number
  authToken: string
  modelBaseUrl: string
  modelName: string
  memoryBackend: string
  memoryBaseUrl: string
  vectorStore: string
  browserEnabled: boolean
  baseImage: string
  networkMode: string
}

interface OpenClawStatus {
  running: boolean
  containerName: string
  gatewayUrl: string
  port: number
  healthy: boolean
  error: string
  image?: string
  createdAt?: string
  teamId?: string
  teamName?: string
  agentName?: string
  agentEmoji?: string
  agentRole?: string
  agentVibe?: string
  agentPrinciples?: string
}

interface TeamProfile {
  id: string
  name: string
  vibe?: string
  role?: string
  principal?: string
  description?: string
  createdAt?: string
  updatedAt?: string
}

interface ProvisionResponse {
  success: boolean
  message: string
  workspaceDir: string
  configPath: string
  containerId: string
  dockerCmd: string
  composeYaml: string
}

// --- Provision Steps ---

const PROVISION_STEPS = [
  { key: 'identity', label: 'Identity & Team' },
  { key: 'skills', label: 'Skills' },
  { key: 'config', label: 'Configuration' },
  { key: 'deploy', label: 'Deploy' },
]

const FALLBACK_MODEL_BASE_URL = 'http://127.0.0.1:8801/v1'

const getDynamicModelBaseUrl = (): string => {
  if (typeof window === 'undefined' || !window.location?.origin) {
    return FALLBACK_MODEL_BASE_URL
  }
  return `${window.location.origin.replace(/\/+$/, '')}/api/router/v1`
}

// --- Component ---

const OpenClawPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'team' | 'provision' | 'status'>('dashboard')
  const [containers, setContainers] = useState<OpenClawStatus[]>([])
  const [teams, setTeams] = useState<TeamProfile[]>([])
  const [statusLoading, setStatusLoading] = useState(true)
  const [teamsLoading, setTeamsLoading] = useState(true)

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/openclaw/status')
      if (res.ok) {
        const data = await res.json()
        setContainers(Array.isArray(data) ? data : [])
      }
    } catch {
      // ignore
    } finally {
      setStatusLoading(false)
    }
  }, [])

  const fetchTeams = useCallback(async () => {
    try {
      const res = await fetch('/api/openclaw/teams')
      if (res.ok) {
        const data = await res.json()
        setTeams(Array.isArray(data) ? data : [])
      } else {
        setTeams([])
      }
    } catch {
      setTeams([])
    } finally {
      setTeamsLoading(false)
    }
  }, [])

  const refreshAll = useCallback(() => {
    void Promise.all([fetchStatus(), fetchTeams()])
  }, [fetchStatus, fetchTeams])

  useEffect(() => {
    refreshAll()
    const interval = setInterval(refreshAll, 15000)
    return () => clearInterval(interval)
  }, [refreshAll])

  const runningCount = containers.filter(c => c.running).length
  const teamCount = teams.length

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <img className={styles.logo} src="/openclaw.png" alt="OpenClaw logo" />
        <div className={styles.titleRow}>
          <h1 className={styles.title}>OpenClaw Team</h1>
          {runningCount > 0 && (
            <span className={`${styles.titleBadge} ${styles.badgeRunning}`}>
              {runningCount} Running
            </span>
          )}
        </div>
        <p className={styles.subtitle}>
          Build, compose, and operate your OpenClaw team with unified identity, routing, and runtime control.
        </p>
        <div className={styles.headerActions}>
          <button className={styles.btnSecondary} onClick={refreshAll}>
            Refresh Team
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className={styles.tabs}>
        <button
          className={`${styles.tab} ${activeTab === 'dashboard' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          <span className={styles.tabIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
              <polyline points="7.5 4.21 12 6.81 16.5 4.21" />
              <polyline points="7.5 19.79 7.5 14.6 3 12" />
              <polyline points="21 12 16.5 14.6 16.5 19.79" />
              <polyline points="12 22.08 12 16.89 16.5 14.3" />
              <polyline points="12 16.89 7.5 14.3" />
              <polyline points="12 6.81 12 12" />
            </svg>
          </span>
          Claw Dashboard ({containers.length})
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'team' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('team')}
        >
          <span className={styles.tabIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
          </span>
          Team ({teamCount})
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'provision' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('provision')}
        >
          <span className={styles.tabIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" /><path d="M2 17l10 5 10-5" /><path d="M2 12l10 5 10-5" />
            </svg>
          </span>
          Claw Provision
        </button>
        <button
          className={`${styles.tab} ${activeTab === 'status' ? styles.tabActive : ''}`}
          onClick={() => setActiveTab('status')}
        >
          <span className={styles.tabIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </span>
          Claw Status ({containers.length})
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'dashboard' && (
        <ClawDashboardTab
          containers={containers}
          teams={teams}
          onSwitchToStatus={() => setActiveTab('status')}
        />
      )}
      {activeTab === 'team' && (
        <TeamTab
          teams={teams}
          teamsLoading={teamsLoading}
          containers={containers}
          onTeamsUpdated={fetchTeams}
          onSwitchToProvision={() => setActiveTab('provision')}
        />
      )}
      {activeTab === 'provision' && (
        <ProvisionTab
          containers={containers}
          teams={teams}
          onProvisioned={refreshAll}
          onSwitchToTeam={() => setActiveTab('team')}
          onSwitchToStatus={() => setActiveTab('status')}
        />
      )}
      {activeTab === 'status' && (
        <StatusTab
          containers={containers}
          statusLoading={statusLoading}
          onRefresh={refreshAll}
        />
      )}
    </div>
  )
}

const truncateText = (value?: string, maxLength = 180): string => {
  const text = (value || '').trim()
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength).trim()}...`
}

const ClawDashboardTab: React.FC<{
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

  if (containers.length === 0) {
    return (
      <div className={styles.emptyState}>
        <div className={styles.emptyStateIcon}>{'\u{1F4DD}'}</div>
        <div className={styles.emptyStateText}>
          No agent profile available yet.<br />
          Create one in <strong>Claw Provision</strong> and it will appear here.
        </div>
      </div>
    )
  }

  return (
    <div className={styles.teamDashboard}>
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
            Use Claw Status for lifecycle actions, logs, and embedded control UI.
          </p>
          <button className={styles.btnPrimary} onClick={onSwitchToStatus}>
            Open Claw Status
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
                    Manage in Claw Status
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

const TeamTab: React.FC<{
  teams: TeamProfile[]
  teamsLoading: boolean
  containers: OpenClawStatus[]
  onTeamsUpdated: () => void
  onSwitchToProvision: () => void
}> = ({ teams, teamsLoading, containers, onTeamsUpdated, onSwitchToProvision }) => {
  const [editingTeamId, setEditingTeamId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [form, setForm] = useState({
    id: '',
    name: '',
    vibe: '',
    role: '',
    principal: '',
    description: '',
  })

  const teamStats = useMemo(() => {
    const counts = new Map<string, { total: number; running: number }>()
    for (const container of containers) {
      const key = (container.teamId || '').trim() || '__unassigned__'
      const prev = counts.get(key) || { total: 0, running: 0 }
      prev.total += 1
      if (container.running) prev.running += 1
      counts.set(key, prev)
    }
    return counts
  }, [containers])

  const updateForm = (field: keyof typeof form, value: string) =>
    setForm(prev => ({ ...prev, [field]: value }))

  const resetForm = () => {
    setEditingTeamId(null)
    setForm({
      id: '',
      name: '',
      vibe: '',
      role: '',
      principal: '',
      description: '',
    })
  }

  const handleSave = async () => {
    const name = form.name.trim()
    if (!name) {
      setError('Team name is required')
      return
    }
    setSaving(true)
    setError('')
    const payload = {
      id: form.id.trim(),
      name,
      vibe: form.vibe.trim(),
      role: form.role.trim(),
      principal: form.principal.trim(),
      description: form.description.trim(),
    }
    try {
      const endpoint = editingTeamId
        ? `/api/openclaw/teams/${encodeURIComponent(editingTeamId)}`
        : '/api/openclaw/teams'
      const method = editingTeamId ? 'PUT' : 'POST'
      const res = await fetch(endpoint, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(data.error || 'Failed to save team')
      } else {
        resetForm()
        onTeamsUpdated()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const handleEdit = (team: TeamProfile) => {
    setEditingTeamId(team.id)
    setForm({
      id: team.id,
      name: team.name || '',
      vibe: team.vibe || '',
      role: team.role || '',
      principal: team.principal || '',
      description: team.description || '',
    })
    setError('')
  }

  const handleDelete = async (team: TeamProfile) => {
    if (!confirm(`Delete team "${team.name}"? Assigned agents must be removed or reassigned first.`)) return
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/openclaw/teams/${encodeURIComponent(team.id)}`, { method: 'DELETE' })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(data.error || 'Failed to delete team')
      } else {
        if (editingTeamId === team.id) resetForm()
        onTeamsUpdated()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className={styles.teamManager}>
      <div className={styles.teamPanel}>
        <div className={styles.teamPanelHeader}>
          <h3 className={styles.teamPanelTitle}>{editingTeamId ? 'Edit Team' : 'Create Team'}</h3>
          <span className={styles.teamPanelSubtitle}>{teams.length} teams</span>
        </div>
        {error && <div className={styles.errorAlert}><span>{error}</span></div>}
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Team Name</label>
            <input className={styles.textInput} value={form.name} onChange={e => updateForm('name', e.target.value)} placeholder="Routing Core" />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Team ID (Optional)</label>
            <input className={styles.textInput} value={form.id} onChange={e => updateForm('id', e.target.value)} placeholder="routing-core" disabled={Boolean(editingTeamId)} />
          </div>
        </div>
        <div className={styles.formRowThree}>
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Vibe</label>
            <input className={styles.textInput} value={form.vibe} onChange={e => updateForm('vibe', e.target.value)} placeholder="Calm, decisive" />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Role</label>
            <input className={styles.textInput} value={form.role} onChange={e => updateForm('role', e.target.value)} placeholder="Research pod" />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel}>Principal</label>
            <input className={styles.textInput} value={form.principal} onChange={e => updateForm('principal', e.target.value)} placeholder="Safety first" />
          </div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Description</label>
          <textarea className={styles.textArea} value={form.description} onChange={e => updateForm('description', e.target.value)} rows={3} placeholder="What this team is responsible for..." />
        </div>
        <div className={styles.actions}>
          <div className={styles.actionsLeft}>
            {editingTeamId && <button className={styles.btnSecondary} onClick={resetForm}>Cancel Edit</button>}
          </div>
          <div className={styles.actionsRight}>
            <button className={styles.btnPrimary} onClick={handleSave} disabled={saving}>
              {saving ? 'Saving...' : editingTeamId ? 'Update Team' : 'Create Team'}
            </button>
            <button className={styles.btnSecondary} onClick={onSwitchToProvision}>
              Create Agent
            </button>
          </div>
        </div>
      </div>

      {teamsLoading ? (
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Loading teams...</p>
        </div>
      ) : teams.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyStateIcon}>{'\u{1F465}'}</div>
          <div className={styles.emptyStateText}>
            No teams yet. Create one above before provisioning OpenClaw agents.
          </div>
        </div>
      ) : (
        <div className={styles.teamCardGrid}>
          {teams.map(team => {
            const stats = teamStats.get(team.id) || { total: 0, running: 0 }
            return (
              <article key={team.id} className={styles.teamEntityCard}>
                <div className={styles.teamEntityHeader}>
                  <div>
                    <h3 className={styles.teamEntityName}>{team.name}</h3>
                    <div className={styles.teamEntityId}>{team.id}</div>
                  </div>
                  <div className={styles.teamEntityActions}>
                    <button className={styles.btnSmall} onClick={() => handleEdit(team)}>Edit</button>
                    <button className={`${styles.btnSmall} ${styles.btnSmallDanger}`} onClick={() => handleDelete(team)} disabled={saving}>Delete</button>
                  </div>
                </div>
                <div className={styles.teamEntityMeta}>
                  <span><strong>Role:</strong> {team.role || 'Not set'}</span>
                  <span><strong>Vibe:</strong> {team.vibe || 'Not set'}</span>
                  <span><strong>Principal:</strong> {team.principal || 'Not set'}</span>
                </div>
                {team.description && <p className={styles.teamEntityDesc}>{truncateText(team.description, 180)}</p>}
                <div className={styles.teamEntityStats}>
                  <span>{stats.total} agent{stats.total !== 1 ? 's' : ''}</span>
                  <span>{stats.running} running</span>
                </div>
              </article>
            )
          })}
        </div>
      )}
    </div>
  )
}

// =============================================================
//  Status Tab — Multi-container list + embedded Gateway UI
// =============================================================

const StatusTab: React.FC<{
  containers: OpenClawStatus[]
  statusLoading: boolean
  onRefresh: () => void
}> = ({ containers, statusLoading, onRefresh }) => {
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

  const handleAction = async (action: 'start' | 'stop', name: string) => {
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

  // Gateway UI sub-view
  if (selectedContainer && selected?.healthy) {
    if (!gatewayToken) {
      return (
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Connecting to gateway...</p>
        </div>
      )
    }
    const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const proxyBase = `/embedded/openclaw/${encodeURIComponent(selectedContainer)}/`
    const iframeSrc = `${proxyBase}#token=${encodeURIComponent(gatewayToken)}&gatewayUrl=${encodeURIComponent(`${wsProto}://${window.location.host}${proxyBase}`)}`
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
        // fall through to new tab
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
              Back to Claw Status
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
        <div className={styles.emptyState}>
          <div className={styles.emptyStateIcon}>{'\u{1F433}'}</div>
          <div className={styles.emptyStateText}>
            No OpenClaw containers provisioned yet.<br />
            Use the <strong>Claw Provision</strong> tab to create one.
          </div>
        </div>
      ) : (
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
                      >
                        Dashboard
                      </button>
                    )}
                    {c.running ? (
                      <button
                        className={styles.btnSmall}
                        onClick={() => handleAction('stop', c.containerName)}
                        disabled={actionLoading === c.containerName}
                      >
                        Stop
                      </button>
                    ) : (
                      <button
                        className={styles.btnSmall}
                        onClick={() => handleAction('start', c.containerName)}
                        disabled={actionLoading === c.containerName}
                      >
                        Start
                      </button>
                    )}
                    <button
                      className={`${styles.btnSmall} ${styles.btnSmallDanger}`}
                      onClick={() => handleDelete(c.containerName)}
                      disabled={actionLoading === c.containerName}
                    >
                      Remove
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div style={{ display: 'flex', gap: '0.75rem' }}>
        <button className={styles.btnSecondary} onClick={onRefresh}>
          Refresh Status
        </button>
      </div>
    </div>
  )
}

// =============================================================
//  Provision Tab — 4-Step Wizard
// =============================================================

const ProvisionTab: React.FC<{
  containers: OpenClawStatus[]
  teams: TeamProfile[]
  onProvisioned: () => void
  onSwitchToTeam: () => void
  onSwitchToStatus: () => void
}> = ({ containers, teams, onProvisioned, onSwitchToTeam, onSwitchToStatus }) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [skills, setSkills] = useState<SkillTemplate[]>([])
  const [selectedSkills, setSelectedSkills] = useState<string[]>([])
  const [selectedTeamId, setSelectedTeamId] = useState('')
  const [identity, setIdentity] = useState<IdentityConfig>({
    name: '',
    emoji: '',
    role: '',
    vibe: '',
    principles: '',
    boundaries: '',
  })
  const [container, setContainer] = useState<ContainerConfig>({
    containerName: '',
    gatewayPort: 0,
    authToken: '',
    modelBaseUrl: getDynamicModelBaseUrl(),
    modelName: 'auto',
    memoryBackend: 'local',
    memoryBaseUrl: '',
    vectorStore: 'openclaw-demo',
    browserEnabled: false,
    baseImage: 'ghcr.io/openclaw/openclaw:latest',
    networkMode: 'host',
  })
  const [provisionResult, setProvisionResult] = useState<ProvisionResponse | null>(null)
  const [provisionLoading, setProvisionLoading] = useState(false)
  const [provisionError, setProvisionError] = useState('')

  // Fetch available skills and next port on mount
  useEffect(() => {
    fetch('/api/openclaw/skills')
      .then(r => r.json())
      .then(data => setSkills(data))
      .catch(() => {})
    fetch('/api/openclaw/next-port')
      .then(r => r.json())
      .then(d => {
        if (d.port) setContainer(prev => prev.gatewayPort === 0 ? { ...prev, gatewayPort: d.port } : prev)
      })
      .catch(() => {})
  }, [])

  const nameCollision = container.containerName !== '' && containers.some(c => c.containerName === container.containerName)
  const selectedTeam = teams.find(team => team.id === selectedTeamId) || null

  useEffect(() => {
    if (!selectedTeamId && teams.length > 0) {
      setSelectedTeamId(teams[0].id)
    }
  }, [teams, selectedTeamId])

  useEffect(() => {
    if (selectedTeamId && !teams.some(team => team.id === selectedTeamId)) {
      setSelectedTeamId(teams[0]?.id || '')
    }
  }, [teams, selectedTeamId])

  const toggleSkill = (id: string) => {
    setSelectedSkills(prev =>
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    )
  }

  const handleProvision = async () => {
    if (!selectedTeamId || !selectedTeam) {
      setProvisionError('Team selection is required before provisioning.')
      return
    }
    setProvisionLoading(true)
    setProvisionError('')
    setProvisionResult(null)
    try {
      const res = await fetch('/api/openclaw/provision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teamId: selectedTeamId, identity, skills: selectedSkills, container }),
      })
      const data = await res.json()
      if (!res.ok) {
        setProvisionError(data.error || 'Provisioning failed')
      } else {
        setProvisionResult(data)
        onProvisioned()
      }
    } catch (e) {
      setProvisionError(String(e))
    } finally {
      setProvisionLoading(false)
    }
  }

  const goToStep = (step: number) => {
    if (step >= 0 && step <= 3) setCurrentStep(step)
  }

  return (
    <div>
      {/* Stepper */}
      <div className={styles.stepper}>
        {PROVISION_STEPS.map((step, idx) => (
          <React.Fragment key={step.key}>
            {idx > 0 && (
              <div className={`${styles.stepConnector} ${idx <= currentStep ? styles.stepConnectorActive : ''}`} />
            )}
            <button
              className={`${styles.stepItem} ${idx === currentStep ? styles.stepActive : ''} ${idx < currentStep ? styles.stepCompleted : ''}`}
              onClick={() => goToStep(idx)}
            >
              <div className={styles.stepCircle}>
                {idx < currentStep ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  idx + 1
                )}
              </div>
              <span className={styles.stepLabel}>{step.label}</span>
            </button>
          </React.Fragment>
        ))}
      </div>

      {provisionError && (
        <div className={styles.errorAlert}>
          <span>{provisionError}</span>
          <button onClick={() => setProvisionError('')} style={{ background: 'none', border: 'none', color: '#ef4444', cursor: 'pointer', fontSize: '1rem' }}>
            &times;
          </button>
        </div>
      )}

      {teams.length === 0 && (
        <div className={styles.errorAlert} style={{ background: 'rgba(234, 179, 8, 0.1)', borderColor: 'rgba(234, 179, 8, 0.35)', color: '#eab308' }}>
          <span>No team available. Create a team first, then come back to provision.</span>
          <button className={styles.btnSmall} onClick={onSwitchToTeam} type="button">
            Open Team Tab
          </button>
        </div>
      )}

      {/* Step Content */}
      {currentStep === 0 && (
        <IdentityStep
          identity={identity}
          setIdentity={setIdentity}
          teams={teams}
          selectedTeamId={selectedTeamId}
          setSelectedTeamId={setSelectedTeamId}
          onSwitchToTeam={onSwitchToTeam}
        />
      )}
      {currentStep === 1 && <SkillsStep skills={skills} selectedSkills={selectedSkills} toggleSkill={toggleSkill} />}
      {currentStep === 2 && (
        <ConfigStep
          container={container}
          setContainer={setContainer}
          nameCollision={nameCollision}
        />
      )}
      {currentStep === 3 && (
        <DeployStep
          identity={identity}
          selectedSkills={selectedSkills}
          skills={skills}
          container={container}
          selectedTeam={selectedTeam}
          teamMissing={!selectedTeamId}
          nameCollision={nameCollision}
          onProvision={handleProvision}
          provisionLoading={provisionLoading}
          provisionResult={provisionResult}
          onSwitchToStatus={onSwitchToStatus}
        />
      )}

      {/* Navigation */}
      <div className={styles.actions}>
        <div className={styles.actionsLeft}>
          {currentStep > 0 && (
            <button className={styles.btnSecondary} onClick={() => goToStep(currentStep - 1)}>
              Back
            </button>
          )}
        </div>
        <div className={styles.actionsRight}>
          {currentStep < 3 && (
            <button
              className={styles.btnPrimary}
              onClick={() => goToStep(currentStep + 1)}
              disabled={currentStep === 0 && !selectedTeamId}
            >
              Next Step
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// =============================================================
//  Step 1: Identity
// =============================================================

const IdentityStep: React.FC<{
  identity: IdentityConfig
  setIdentity: React.Dispatch<React.SetStateAction<IdentityConfig>>
  teams: TeamProfile[]
  selectedTeamId: string
  setSelectedTeamId: React.Dispatch<React.SetStateAction<string>>
  onSwitchToTeam: () => void
}> = ({ identity, setIdentity, teams, selectedTeamId, setSelectedTeamId, onSwitchToTeam }) => {
  const update = (field: keyof IdentityConfig, value: string) =>
    setIdentity(prev => ({ ...prev, [field]: value }))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 1: Agent Identity</h2>
      <p className={styles.stepDescription}>
        Define who your OpenClaw agent is — its name, personality, principles, and boundaries.
        These files form the agent's core identity (SOUL.md, IDENTITY.md).
      </p>

      <div className={styles.sectionTitle}>Team Selection (Required)</div>
      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Target Team</label>
          <select
            className={styles.selectInput}
            value={selectedTeamId}
            onChange={e => setSelectedTeamId(e.target.value)}
          >
            <option value="">Select a team...</option>
            {teams.map(team => (
              <option key={team.id} value={team.id}>
                {team.name}
              </option>
            ))}
          </select>
          <div className={styles.formHint}>Every agent must belong to one team.</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Need a new team?</label>
          <button className={styles.btnSecondary} onClick={onSwitchToTeam} type="button">
            Go to Team Tab
          </button>
        </div>
      </div>

      <div className={styles.formRowThree}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Agent Name</label>
          <input className={styles.textInput} value={identity.name} onChange={e => update('name', e.target.value)} placeholder="Atlas" />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Emoji</label>
          <input className={styles.textInput} value={identity.emoji} onChange={e => update('emoji', e.target.value)} placeholder={'\u{1F531}'} />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Vibe</label>
          <input className={styles.textInput} value={identity.vibe} onChange={e => update('vibe', e.target.value)} placeholder="Calm, precise, opinionated" />
        </div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Role / Creature</label>
        <input className={styles.textInput} value={identity.role} onChange={e => update('role', e.target.value)} placeholder="AI operations engineer" />
        <div className={styles.formHint}>What kind of creature is your agent? SRE, architect, assistant, mentor...</div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Core Principles (SOUL.md)</label>
        <textarea className={styles.textArea} value={identity.principles} onChange={e => update('principles', e.target.value)} rows={6} placeholder="Your agent's core truths and principles..." />
        <div className={styles.formHint}>Markdown supported. Define the agent's operating principles and values.</div>
      </div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Boundaries</label>
        <textarea className={styles.textArea} value={identity.boundaries} onChange={e => update('boundaries', e.target.value)} rows={4} placeholder="- Don't run destructive commands without approval..." />
        <div className={styles.formHint}>What should the agent never do? Safety guardrails and limits.</div>
      </div>
    </div>
  )
}

// =============================================================
//  Step 2: Skills
// =============================================================

const SkillsStep: React.FC<{
  skills: SkillTemplate[]
  selectedSkills: string[]
  toggleSkill: (id: string) => void
}> = ({ skills, selectedSkills, toggleSkill }) => (
  <div className={styles.stepContent}>
    <h2 className={styles.stepTitle}>Step 2: Select Skills</h2>
    <p className={styles.stepDescription}>
      Skills give your agent specialized abilities. Each skill is a SKILL.md file that defines
      a structured workflow. Selected skills are auto-discovered at startup.
    </p>

    <div className={styles.skillGrid}>
      {skills.map(skill => (
        <div
          key={skill.id}
          className={`${styles.skillCard} ${selectedSkills.includes(skill.id) ? styles.skillCardSelected : ''}`}
          onClick={() => toggleSkill(skill.id)}
        >
          <div className={styles.skillCardHeader}>
            <span className={styles.skillCardEmoji}>{skill.emoji}</span>
            <span className={styles.skillCardName}>{skill.name}</span>
            <span className={styles.skillCardCategory}>{skill.category}</span>
          </div>
          <div className={styles.skillCardDesc}>{skill.description}</div>
        </div>
      ))}
    </div>

    <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--color-text-secondary)' }}>
      {selectedSkills.length} skill{selectedSkills.length !== 1 ? 's' : ''} selected
    </div>
  </div>
)

// =============================================================
//  Step 3: Configuration
// =============================================================

const ConfigStep: React.FC<{
  container: ContainerConfig
  setContainer: React.Dispatch<React.SetStateAction<ContainerConfig>>
  nameCollision: boolean
}> = ({ container, setContainer, nameCollision }) => {
  const update = (field: keyof ContainerConfig, value: string | number | boolean) =>
    setContainer(prev => ({ ...prev, [field]: value }))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 3: Container & Model Configuration</h2>
      <p className={styles.stepDescription}>
        Configure how OpenClaw connects to Semantic Router for model routing and memory,
        and set container parameters.
      </p>

      <div className={styles.sectionTitle}>Container</div>

      <div className={styles.formRowThree}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Container Name</label>
          <input className={styles.textInput} value={container.containerName} onChange={e => update('containerName', e.target.value)} placeholder="my-agent" />
          {nameCollision && (
            <div className={styles.nameWarning}>
              A container with this name already exists and will be replaced.
            </div>
          )}
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Gateway Port</label>
          <input className={styles.numberInput} type="number" value={container.gatewayPort} onChange={e => update('gatewayPort', parseInt(e.target.value) || 0)} />
          <div className={styles.formHint}>Auto-assigned if 0 or conflicting</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Base Image</label>
          <input className={styles.textInput} value={container.baseImage} onChange={e => update('baseImage', e.target.value)} />
        </div>
      </div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Auth Token</label>
          <input className={styles.textInput} value={container.authToken} onChange={e => update('authToken', e.target.value)} placeholder="Auto-generated if empty" />
          <div className={styles.formHint}>Leave blank to auto-generate</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Network Mode</label>
          <select className={styles.selectInput} value={container.networkMode} onChange={e => update('networkMode', e.target.value)}>
            <option value="host">host (recommended)</option>
            <option value="bridge">bridge</option>
          </select>
        </div>
      </div>

      <div className={styles.sectionTitle}>Model Provider (via Semantic Router)</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Base URL</label>
          <input className={styles.textInput} value={container.modelBaseUrl} onChange={e => update('modelBaseUrl', e.target.value)} />
          <div className={styles.formHint}>Auto-discovered from playground routing; editable if needed</div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Model Name</label>
          <input className={styles.textInput} value={container.modelName} onChange={e => update('modelName', e.target.value)} />
          <div className={styles.formHint}>&quot;auto&quot; for SR confidence routing</div>
        </div>
      </div>

      <div className={styles.sectionTitle}>Memory Mode</div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Memory Backend</label>
          <div className={styles.toggle}>
            <button className={`${styles.toggleOption} ${container.memoryBackend === 'remote' ? styles.toggleOptionSelected : ''}`} onClick={() => update('memoryBackend', 'remote')}>
              Remote Embeddings
            </button>
            <button className={`${styles.toggleOption} ${container.memoryBackend === 'local' ? styles.toggleOptionSelected : ''}`} onClick={() => update('memoryBackend', 'local')}>
              Built-in (Recommended)
            </button>
          </div>
        </div>
      </div>

      {container.memoryBackend === 'remote' && (
        <div className={styles.formGroup}>
          <label className={styles.formLabel}>Embedding Base URL (Optional)</label>
          <input className={styles.textInput} value={container.memoryBaseUrl} onChange={e => update('memoryBaseUrl', e.target.value)} placeholder="https://your-openai-compatible-endpoint/v1" />
          <div className={styles.formHint}>Used for agents.defaults.memorySearch.remote.baseUrl</div>
        </div>
      )}

      <div className={styles.sectionTitle}>Features</div>

      <div className={styles.formGroup}>
        <label className={styles.formLabel}>Browser (Playwright)</label>
        <div className={styles.toggle}>
          <button className={`${styles.toggleOption} ${container.browserEnabled ? styles.toggleOptionSelected : ''}`} onClick={() => update('browserEnabled', true)}>
            Enabled
          </button>
          <button className={`${styles.toggleOption} ${!container.browserEnabled ? styles.toggleOptionSelected : ''}`} onClick={() => update('browserEnabled', false)}>
            Disabled
          </button>
        </div>
        <div className={styles.formHint}>Enable headless browser for web browsing and CUA tasks</div>
      </div>
    </div>
  )
}

// =============================================================
//  Step 4: Deploy
// =============================================================

const DeployStep: React.FC<{
  identity: IdentityConfig
  selectedSkills: string[]
  skills: SkillTemplate[]
  container: ContainerConfig
  selectedTeam: TeamProfile | null
  teamMissing: boolean
  nameCollision: boolean
  onProvision: () => void
  provisionLoading: boolean
  provisionResult: ProvisionResponse | null
  onSwitchToStatus: () => void
}> = ({ identity, selectedSkills, skills, container, selectedTeam, teamMissing, nameCollision, onProvision, provisionLoading, provisionResult, onSwitchToStatus }) => {
  const [copied, setCopied] = useState('')
  const [showCommands, setShowCommands] = useState(false)

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(label)
      setTimeout(() => setCopied(''), 2000)
    })
  }

  const selectedSkillNames = skills.filter(s => selectedSkills.includes(s.id))

  return (
    <div className={styles.stepContent}>
      <h2 className={styles.stepTitle}>Step 4: Review & Deploy</h2>
      <p className={styles.stepDescription}>
        Review your configuration, then provision and start the OpenClaw container.
      </p>

      {nameCollision && (
        <div className={styles.errorAlert} style={{ background: 'rgba(234, 179, 8, 0.1)', borderColor: 'rgba(234, 179, 8, 0.3)', color: '#eab308' }}>
          <span>Container &quot;{container.containerName}&quot; already exists and will be replaced upon provisioning.</span>
        </div>
      )}
      {teamMissing && (
        <div className={styles.errorAlert} style={{ background: 'rgba(239, 68, 68, 0.1)', borderColor: 'rgba(239, 68, 68, 0.35)', color: '#ef4444' }}>
          <span>Team selection is required before deployment.</span>
        </div>
      )}

      {/* Summary */}
      <div className={styles.summaryGrid}>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Identity</div>
          <div className={styles.summaryCardContent}>
            <strong>{identity.emoji} {identity.name || '(unnamed)'}</strong><br />
            {identity.role || '(no role)'}<br />
            <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>{identity.vibe}</span>
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Team</div>
          <div className={styles.summaryCardContent}>
            <strong>{selectedTeam?.name || '(not selected)'}</strong><br />
            {(selectedTeam?.role || 'No role set')}<br />
            <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>{selectedTeam?.vibe || 'No vibe set'}</span>
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Skills ({selectedSkills.length})</div>
          <div className={styles.summarySkillList}>
            {selectedSkillNames.map(s => (
              <span key={s.id} className={styles.summarySkillBadge}>{s.emoji} {s.name}</span>
            ))}
            {selectedSkills.length === 0 && <span style={{ color: 'var(--color-text-secondary)', fontSize: '0.8rem' }}>No skills selected</span>}
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Container</div>
          <div className={styles.summaryCardContent}>
            <strong>{container.containerName || '(auto)'}</strong> :{container.gatewayPort || 'auto'}<br />
            Image: {container.baseImage}<br />
            Network: {container.networkMode}
          </div>
        </div>
        <div className={styles.summaryCard}>
          <div className={styles.summaryCardTitle}>Model & Memory</div>
          <div className={styles.summaryCardContent}>
            Model: {container.modelName} via SR<br />
            Memory: {container.memoryBackend === 'remote' ? `Remote embeddings${container.memoryBaseUrl ? ` (${container.memoryBaseUrl})` : ''}` : 'Built-in'}<br />
            Browser: {container.browserEnabled ? 'Enabled' : 'Disabled'}
          </div>
        </div>
      </div>

      {/* Provision & Start Button */}
      {!provisionResult && (
        <button className={styles.btnSuccess} onClick={onProvision} disabled={provisionLoading || teamMissing}>
          {provisionLoading ? 'Provisioning & starting container...' : 'Provision & Start'}
        </button>
      )}

      {/* Result */}
      {provisionResult?.success && (
        <div className={styles.successCard}>
          <div className={styles.successIcon}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><polyline points="16 8 10 16 7 13" />
            </svg>
          </div>
          <div className={styles.successTitle}>Container Started</div>
          <div className={styles.successMessage}>
            {provisionResult.message}
            {provisionResult.containerId && (
              <><br /><code style={{ fontSize: '0.75rem' }}>{provisionResult.containerId.slice(0, 12)}</code></>
            )}
          </div>

          <button className={styles.btnPrimary} onClick={onSwitchToStatus} style={{ marginBottom: '1rem' }}>
            Go to Claw Status
          </button>

          {/* Collapsible reference commands */}
          <div style={{ textAlign: 'left' }}>
            <button
              onClick={() => setShowCommands(!showCommands)}
              style={{
                background: 'none', border: 'none', color: 'var(--color-text-secondary)',
                fontSize: '0.8rem', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem',
              }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                style={{ transform: showCommands ? 'rotate(90deg)' : 'rotate(0)', transition: 'transform 0.2s' }}>
                <polyline points="9 18 15 12 9 6" />
              </svg>
              Docker commands reference
            </button>

            {showCommands && (
              <>
                {provisionResult.dockerCmd && (
                  <div className={styles.codePreview}>
                    <div className={styles.codePreviewHeader}>
                      <span className={styles.codePreviewLabel}>Docker Run Command</span>
                      <button className={styles.codePreviewCopy} onClick={() => copyToClipboard(provisionResult.dockerCmd, 'docker')}>
                        {copied === 'docker' ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                    <pre className={styles.codePreviewContent}>{provisionResult.dockerCmd}</pre>
                  </div>
                )}

                {provisionResult.composeYaml && (
                  <div className={styles.codePreview}>
                    <div className={styles.codePreviewHeader}>
                      <span className={styles.codePreviewLabel}>Docker Compose YAML</span>
                      <button className={styles.codePreviewCopy} onClick={() => copyToClipboard(provisionResult.composeYaml, 'compose')}>
                        {copied === 'compose' ? 'Copied!' : 'Copy'}
                      </button>
                    </div>
                    <pre className={styles.codePreviewContent}>{provisionResult.composeYaml}</pre>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default OpenClawPage
