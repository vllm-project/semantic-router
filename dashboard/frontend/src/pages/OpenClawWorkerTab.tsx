import React, { useEffect, useMemo, useState } from 'react'
import styles from './OpenClawPage.module.css'
import {
  truncateText,
  type OpenClawStatus,
  type TeamProfile,
} from './OpenClawPageSupport'
import { WorkerProvisionWizard } from './OpenClawWorkerProvisionWizard'

export const WorkerTab: React.FC<{
  containers: OpenClawStatus[]
  teams: TeamProfile[]
  onProvisioned: () => void
  onSwitchToTeam: () => void
  onSwitchToStatus: () => void
  readOnly: boolean
}> = ({ containers, teams, onProvisioned, onSwitchToTeam, onSwitchToStatus, readOnly }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [isEditModalOpen, setIsEditModalOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [editingWorker, setEditingWorker] = useState<OpenClawStatus | null>(null)
  const [editForm, setEditForm] = useState({
    teamId: '',
    name: '',
    emoji: '',
    role: '',
    vibe: '',
    principles: '',
  })

  const filteredWorkers = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    const sorted = [...containers].sort((a, b) => {
      const left = (a.agentName || a.containerName || '').toLowerCase()
      const right = (b.agentName || b.containerName || '').toLowerCase()
      return left.localeCompare(right)
    })
    if (!query) return sorted
    return sorted.filter(worker => (
      [
        worker.containerName,
        worker.agentName || '',
        worker.teamName || '',
        worker.teamId || '',
        worker.agentRole || '',
        worker.agentVibe || '',
        worker.agentPrinciples || '',
      ].some(value => value.toLowerCase().includes(query))
    ))
  }, [containers, searchQuery])

  const openEditModal = (worker: OpenClawStatus) => {
    if (readOnly) return
    setEditingWorker(worker)
    setEditForm({
      teamId: worker.teamId || '',
      name: worker.agentName || '',
      emoji: worker.agentEmoji || '',
      role: worker.agentRole || '',
      vibe: worker.agentVibe || '',
      principles: worker.agentPrinciples || '',
    })
    setError('')
    setIsEditModalOpen(true)
  }

  const updateEditForm = (field: keyof typeof editForm, value: string) => {
    setEditForm(prev => ({ ...prev, [field]: value }))
  }

  const handleUpdateWorker = async () => {
    if (readOnly) return
    if (!editingWorker) return
    if (!editForm.teamId.trim()) {
      setError('Worker team is required')
      return
    }
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/openclaw/workers/${encodeURIComponent(editingWorker.containerName)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          teamId: editForm.teamId.trim(),
          identity: {
            name: editForm.name.trim(),
            emoji: editForm.emoji.trim(),
            role: editForm.role.trim(),
            vibe: editForm.vibe.trim(),
            principles: editForm.principles.trim(),
          },
        }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(data.error || 'Failed to update worker')
      } else {
        setIsEditModalOpen(false)
        setEditingWorker(null)
        onProvisioned()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const handleDeleteWorker = async (worker: OpenClawStatus) => {
    if (readOnly) return
    if (!confirm(`Delete worker "${worker.agentName || worker.containerName}"?`)) return
    setSaving(true)
    setError('')
    try {
      const res = await fetch(`/api/openclaw/workers/${encodeURIComponent(worker.containerName)}`, { method: 'DELETE' })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        setError(data.error || 'Failed to delete worker')
      } else {
        onProvisioned()
      }
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  useEffect(() => {
    if (!readOnly) {
      return
    }
    setIsCreateModalOpen(false)
    setIsEditModalOpen(false)
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
            placeholder="Search worker by name, team, role, vibe..."
          />
        </div>
        <div className={styles.entityToolbarActions}>
          <button className={styles.btnPrimary} onClick={() => setIsCreateModalOpen(true)} disabled={readOnly}>
            New Worker
          </button>
        </div>
      </div>
      {error && <div className={styles.errorAlert}><span>{error}</span></div>}

      {filteredWorkers.length === 0 ? (
        <div className={styles.emptyState}>
          <div className={styles.emptyStateText}>
            {containers.length === 0 ? 'No workers created yet.' : 'No workers match your search.'}
          </div>
        </div>
      ) : (
        <div className={styles.agentGrid}>
          {filteredWorkers.map(worker => {
            const name = worker.agentName?.trim() || worker.containerName
            const emoji = worker.agentEmoji?.trim() || '\u{1F916}'
            const role = worker.agentRole?.trim() || 'Not set'
            const vibe = worker.agentVibe?.trim() || 'Not set'
            const principles = truncateText(worker.agentPrinciples, 160) || 'Not set'

            return (
              <article key={worker.containerName} className={styles.agentCard}>
                <div className={styles.agentCardHeader}>
                  <div className={styles.agentAvatar}>{emoji}</div>
                  <div className={styles.agentHeaderMeta}>
                    <div className={styles.agentName}>{name}</div>
                    <div className={styles.agentContainerRef}>{worker.containerName}</div>
                    <div className={styles.teamTag}>{worker.teamName?.trim() || 'Unassigned'}</div>
                  </div>
                  <span className={`${styles.healthBadge} ${
                    worker.healthy
                      ? styles.healthBadgeHealthy
                      : worker.running
                        ? styles.healthBadgeRunning
                        : styles.healthBadgeStopped
                  }`}>
                    {worker.healthy ? 'Healthy' : worker.running ? 'Starting' : 'Stopped'}
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
                    <span className={styles.agentMetaValue}>{worker.teamName?.trim() || 'Unassigned'}</span>
                  </div>
                  <div className={styles.agentMetaRow}>
                    <span className={styles.agentMetaLabel}>Principal</span>
                    <span className={styles.agentMetaValue}>{principles}</span>
                  </div>
                </div>

                <div className={styles.agentFooter}>
                  <div className={styles.entityRowActions}>
                    <button className={styles.btnSmall} onClick={() => openEditModal(worker)} disabled={readOnly}>Edit</button>
                    <button className={styles.btnSmall} onClick={onSwitchToStatus}>Status</button>
                    <button className={`${styles.btnSmall} ${styles.btnSmallDanger}`} onClick={() => handleDeleteWorker(worker)} disabled={readOnly || saving}>
                      Delete
                    </button>
                  </div>
                  {worker.createdAt && (
                    <span className={styles.agentTimestamp}>
                      Created {new Date(worker.createdAt).toLocaleString()}
                    </span>
                  )}
                </div>
              </article>
            )
          })}
        </div>
      )}

      {isCreateModalOpen && (
        <div className={styles.ocModalOverlay} onClick={() => setIsCreateModalOpen(false)}>
          <div className={`${styles.ocModal} ${styles.ocModalWide}`} onClick={e => e.stopPropagation()}>
            <div className={styles.ocModalHeader}>
              <div className={styles.ocModalTitleRow}>
                <img className={styles.ocModalLogo} src="/openclaw.svg" alt="" aria-hidden="true" />
                <h3 className={styles.ocModalTitle}>New Worker</h3>
              </div>
              <button className={styles.ocModalClose} onClick={() => setIsCreateModalOpen(false)} aria-label="Close">
                ×
              </button>
            </div>
            <div className={styles.ocModalBody}>
              <WorkerProvisionWizard
                teams={teams}
                onProvisioned={onProvisioned}
                onSwitchToTeam={onSwitchToTeam}
                onSwitchToStatus={onSwitchToStatus}
                onCreated={() => setIsCreateModalOpen(false)}
              />
            </div>
          </div>
        </div>
      )}

      {isEditModalOpen && editingWorker && (
        <div className={styles.ocModalOverlay} onClick={() => !saving && setIsEditModalOpen(false)}>
          <div className={styles.ocModal} onClick={e => e.stopPropagation()}>
            <div className={styles.ocModalHeader}>
              <h3 className={styles.ocModalTitle}>Edit Worker</h3>
              <button className={styles.ocModalClose} onClick={() => !saving && setIsEditModalOpen(false)} disabled={saving} aria-label="Close">
                ×
              </button>
            </div>
            <div className={styles.ocModalBody}>
              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Team</label>
                  <select
                    className={styles.selectInput}
                    value={editForm.teamId}
                    onChange={e => updateEditForm('teamId', e.target.value)}
                  >
                    <option value="">Select a team...</option>
                    {teams.map(team => (
                      <option key={team.id} value={team.id}>{team.name}</option>
                    ))}
                  </select>
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Emoji</label>
                  <input
                    className={styles.textInput}
                    value={editForm.emoji}
                    onChange={e => updateEditForm('emoji', e.target.value)}
                    placeholder={'\u{1F916}'}
                  />
                </div>
              </div>
              <div className={styles.formRow}>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Worker Name</label>
                  <input className={styles.textInput} value={editForm.name} onChange={e => updateEditForm('name', e.target.value)} placeholder="Atlas" />
                </div>
                <div className={styles.formGroup}>
                  <label className={styles.formLabel}>Role</label>
                  <input className={styles.textInput} value={editForm.role} onChange={e => updateEditForm('role', e.target.value)} placeholder="AI operations engineer" />
                </div>
              </div>
              <div className={styles.formGroup}>
                <label className={styles.formLabel}>Vibe</label>
                <input className={styles.textInput} value={editForm.vibe} onChange={e => updateEditForm('vibe', e.target.value)} placeholder="Calm, precise, opinionated" />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.formLabel}>Principal</label>
                <textarea className={styles.textArea} value={editForm.principles} onChange={e => updateEditForm('principles', e.target.value)} rows={5} placeholder="Core truths and operating principles..." />
              </div>
            </div>
            <div className={styles.ocModalFooter}>
              <button className={styles.btnSecondary} onClick={() => setIsEditModalOpen(false)} disabled={saving}>
                Cancel
              </button>
              <button className={styles.btnPrimary} onClick={handleUpdateWorker} disabled={saving}>
                {saving ? 'Saving...' : 'Update Worker'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
