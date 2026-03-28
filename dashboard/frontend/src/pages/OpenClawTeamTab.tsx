import React, { useEffect, useMemo, useState } from 'react'
import styles from './OpenClawPage.module.css'
import {
  truncateText,
  type OpenClawStatus,
  type TeamProfile,
} from './OpenClawPageSupport'

export const TeamTab: React.FC<{
  teams: TeamProfile[]
  teamsLoading: boolean
  containers: OpenClawStatus[]
  onTeamsUpdated: () => void
  readOnly: boolean
}> = ({ teams, teamsLoading, containers, onTeamsUpdated, readOnly }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [isModalOpen, setIsModalOpen] = useState(false)
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

  const openCreateModal = () => {
    if (readOnly) return
    resetForm()
    setError('')
    setIsModalOpen(true)
  }

  const handleSave = async () => {
    if (readOnly) return
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
        setIsModalOpen(false)
        onTeamsUpdated()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const handleEdit = (team: TeamProfile) => {
    if (readOnly) return
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
    setIsModalOpen(true)
  }

  const handleDelete = async (team: TeamProfile) => {
    if (readOnly) return
    if (!confirm(`Delete team "${team.name}"? Assigned agents must be removed or reassigned first.`)) return
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/openclaw/teams/${encodeURIComponent(team.id)}`, { method: 'DELETE' })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(data.error || 'Failed to delete team')
      } else {
        if (editingTeamId === team.id) {
          resetForm()
          setIsModalOpen(false)
        }
        onTeamsUpdated()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const filteredTeams = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    if (!query) {
      return teams
    }
    return teams.filter(team => {
      return [
        team.id,
        team.name,
        team.role || '',
        team.vibe || '',
        team.principal || '',
        team.description || '',
      ].some(value => value.toLowerCase().includes(query))
    })
  }, [teams, searchQuery])

  useEffect(() => {
    if (!readOnly) {
      return
    }
    setIsModalOpen(false)
    setSaving(false)
  }, [readOnly])

  return (
    <div className={styles.teamManager}>
      <div className={styles.entityToolbar}>
        <div className={styles.entitySearch}>
          <input
            className={styles.entitySearchInput}
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search team by name, id, role, vibe..."
          />
        </div>
        <div className={styles.entityToolbarActions}>
          <button className={styles.btnPrimary} onClick={openCreateModal} disabled={readOnly}>
            New Team
          </button>
        </div>
      </div>
      {error && <div className={styles.errorAlert}><span>{error}</span></div>}

      {teamsLoading ? (
        <div className={styles.loading}>
          <div className={styles.spinner} />
          <p>Loading teams...</p>
        </div>
      ) : filteredTeams.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyStateText}>
            {teams.length === 0 ? 'No teams yet.' : 'No teams match your search.'}
          </div>
        </div>
      ) : (
        <div className={styles.teamCardGrid}>
          {filteredTeams.map(team => {
            const stats = teamStats.get(team.id) || { total: 0, running: 0 }
            return (
              <article key={team.id} className={styles.teamEntityCard}>
                <div className={styles.teamEntityHeader}>
                  <div>
                    <h3 className={styles.teamEntityName}>{team.name}</h3>
                    <div className={styles.teamEntityId}>{team.id}</div>
                  </div>
                  <div className={styles.teamEntityActions}>
                    <button className={styles.btnSmall} onClick={() => handleEdit(team)} disabled={readOnly}>Edit</button>
                    <button className={`${styles.btnSmall} ${styles.btnSmallDanger}`} onClick={() => handleDelete(team)} disabled={readOnly || saving}>Delete</button>
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

      {isModalOpen && (
        <div className={styles.ocModalOverlay} onClick={() => !saving && setIsModalOpen(false)}>
          <div className={styles.ocModal} onClick={e => e.stopPropagation()}>
            <div className={styles.ocModalHeader}>
              <h3 className={styles.ocModalTitle}>{editingTeamId ? 'Edit Team' : 'New Team'}</h3>
              <button
                className={styles.ocModalClose}
                onClick={() => !saving && setIsModalOpen(false)}
                disabled={saving}
                aria-label="Close"
              >
                ×
              </button>
            </div>
            <div className={styles.ocModalBody}>
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
                <textarea className={styles.textArea} value={form.description} onChange={e => updateForm('description', e.target.value)} rows={4} placeholder="What this team is responsible for..." />
              </div>
            </div>
            <div className={styles.ocModalFooter}>
              <button
                className={styles.btnSecondary}
                onClick={() => setIsModalOpen(false)}
                disabled={saving}
              >
                Cancel
              </button>
              <button className={styles.btnPrimary} onClick={handleSave} disabled={saving}>
                {saving ? 'Saving...' : editingTeamId ? 'Update Team' : 'Create Team'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
