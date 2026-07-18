import React, { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import {
  buildTeamRuntimeStats,
  filterAndSortOpenClawTeams,
  getOpenClawPageCount,
  OPENCLAW_TEAMS_PAGE_SIZE,
  paginateOpenClawItems,
  type TeamCatalogFilter,
  type TeamCatalogSort,
} from '../utils/openClawCatalogSupport'
import {
  createLatestOpenClawRequest,
  fetchOpenClawJSON,
  getOpenClawErrorMessage,
  type LatestOpenClawRequest,
} from '../utils/openClawRequestSupport'
import { OpenClawCatalogControls } from './OpenClawCatalogControls'
import { OpenClawDialog } from './OpenClawDialog'
import styles from './OpenClawPage.module.css'
import { truncateText, type OpenClawStatus, type TeamProfile } from './OpenClawPageSupport'
import { OpenClawRequestNotice } from './OpenClawRequestNotice'

interface TeamTabProps {
  containers: OpenClawStatus[]
  readOnly: boolean
  teams: TeamProfile[]
  teamsError?: string | null
  teamsLoading: boolean
  onRetryTeams?: () => void
  onTeamsUpdated: () => void
}

const emptyTeamForm = {
  id: '',
  name: '',
  vibe: '',
  role: '',
  principal: '',
  description: '',
}

export const TeamTab: React.FC<TeamTabProps> = ({
  teams,
  teamsLoading,
  teamsError,
  containers,
  onRetryTeams,
  onTeamsUpdated,
  readOnly,
}) => {
  const nameId = useId()
  const teamId = useId()
  const vibeId = useId()
  const roleId = useId()
  const principalId = useId()
  const descriptionId = useId()
  const [searchQuery, setSearchQuery] = useState('')
  const deferredSearch = useDeferredValue(searchQuery)
  const [filter, setFilter] = useState<TeamCatalogFilter>('all')
  const [sort, setSort] = useState<TeamCatalogSort>('name-asc')
  const [page, setPage] = useState(1)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingTeamId, setEditingTeamId] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [formError, setFormError] = useState('')
  const [deleteError, setDeleteError] = useState('')
  const [pendingDeleteTeam, setPendingDeleteTeam] = useState<TeamProfile | null>(null)
  const [form, setForm] = useState(emptyTeamForm)
  const mutationRequestRef = useRef<LatestOpenClawRequest | null>(null)
  if (!mutationRequestRef.current) mutationRequestRef.current = createLatestOpenClawRequest()

  const teamStats = useMemo(() => buildTeamRuntimeStats(containers), [containers])
  const filteredTeams = useMemo(
    () => filterAndSortOpenClawTeams(teams, teamStats, deferredSearch, filter, sort),
    [deferredSearch, filter, sort, teamStats, teams],
  )
  const pageCount = getOpenClawPageCount(filteredTeams.length, OPENCLAW_TEAMS_PAGE_SIZE)
  const safePage = Math.min(page, pageCount)
  const visibleTeams = paginateOpenClawItems(filteredTeams, safePage, OPENCLAW_TEAMS_PAGE_SIZE)

  useEffect(() => setPage(1), [deferredSearch, filter, sort])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])
  useEffect(() => () => mutationRequestRef.current?.cancel(), [])
  useEffect(() => {
    if (!readOnly) return
    mutationRequestRef.current?.cancel()
    setIsModalOpen(false)
    setPendingDeleteTeam(null)
    setSaving(false)
    setFormError('')
    setDeleteError('')
  }, [readOnly])

  const updateForm = (field: keyof typeof form, value: string) => {
    setForm((current) => ({ ...current, [field]: value }))
  }

  const resetForm = () => {
    setEditingTeamId(null)
    setForm(emptyTeamForm)
    setFormError('')
  }

  const closeModal = () => {
    if (saving) return
    setIsModalOpen(false)
    resetForm()
  }

  const openCreateModal = () => {
    if (readOnly) return
    resetForm()
    setIsModalOpen(true)
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
    setFormError('')
    setIsModalOpen(true)
  }

  const handleSave = async () => {
    if (readOnly) return
    const name = form.name.trim()
    if (!name) {
      setFormError('Team name is required.')
      return
    }
    const endpoint = editingTeamId
      ? `/api/openclaw/teams/${encodeURIComponent(editingTeamId)}`
      : '/api/openclaw/teams'
    const payload = {
      id: form.id.trim(),
      name,
      vibe: form.vibe.trim(),
      role: form.role.trim(),
      principal: form.principal.trim(),
      description: form.description.trim(),
    }

    await mutationRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<TeamProfile>(
          endpoint,
          {
            method: editingTeamId ? 'PUT' : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          },
          signal,
        ),
      {
        onStart: () => {
          setSaving(true)
          setFormError('')
        },
        onSuccess: () => {
          setIsModalOpen(false)
          resetForm()
          onTeamsUpdated()
        },
        onError: (error) => {
          setFormError(getOpenClawErrorMessage(error, 'Failed to save the team.'))
        },
        onFinish: () => setSaving(false),
      },
    )
  }

  const requestDelete = (team: TeamProfile) => {
    if (readOnly) return
    setDeleteError('')
    setPendingDeleteTeam(team)
  }

  const confirmDelete = async () => {
    const team = pendingDeleteTeam
    if (readOnly || !team) return
    await mutationRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<void>(
          `/api/openclaw/teams/${encodeURIComponent(team.id)}`,
          { method: 'DELETE' },
          signal,
        ),
      {
        onStart: () => {
          setSaving(true)
          setDeleteError('')
        },
        onSuccess: () => {
          if (editingTeamId === team.id) {
            setIsModalOpen(false)
            resetForm()
          }
          setPendingDeleteTeam(null)
          onTeamsUpdated()
        },
        onError: (error) => {
          setDeleteError(getOpenClawErrorMessage(error, 'Failed to delete the team.'))
        },
        onFinish: () => setSaving(false),
      },
    )
  }

  return (
    <div className={styles.teamManager}>
      <div className={styles.entityToolbarActionsTop}>
        {!readOnly ? (
          <button type="button" className={styles.btnPrimary} onClick={openCreateModal}>
            New Team
          </button>
        ) : null}
      </div>

      {teamsError ? (
        <OpenClawRequestNotice
          title="Teams are unavailable"
          message={teamsError}
          onRetry={onRetryTeams}
        />
      ) : null}

      {teamsLoading && teams.length === 0 ? (
        <div className={styles.loading} role="status">
          <div className={styles.spinner} />
          <p>Loading teams…</p>
        </div>
      ) : (
        <>
          <OpenClawCatalogControls
            searchLabel="Search teams"
            searchValue={searchQuery}
            filterLabel="Roster"
            filterValue={filter}
            filterOptions={[
              { value: 'all', label: 'All teams' },
              { value: 'with-workers', label: 'With workers' },
              { value: 'empty', label: 'Without workers' },
            ]}
            sortValue={sort}
            sortOptions={[
              { value: 'name-asc', label: 'Name A–Z' },
              { value: 'workers-desc', label: 'Most workers' },
              { value: 'running-desc', label: 'Most running' },
              { value: 'updated-desc', label: 'Recently updated' },
            ]}
            itemCount={filteredTeams.length}
            totalCount={teams.length}
            itemLabel="teams"
            page={safePage}
            pageSize={OPENCLAW_TEAMS_PAGE_SIZE}
            onSearchChange={setSearchQuery}
            onFilterChange={(value) => setFilter(value as TeamCatalogFilter)}
            onSortChange={(value) => setSort(value as TeamCatalogSort)}
            onPageChange={setPage}
          />

          {visibleTeams.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyStateText}>
                {teams.length === 0
                  ? readOnly
                    ? 'No OpenClaw teams are configured.'
                    : 'No teams yet. Create one to organize workers.'
                  : 'No teams match the current search and roster filter.'}
              </div>
            </div>
          ) : (
            <div className={styles.teamCardGrid}>
              {visibleTeams.map((team) => {
                const stats = teamStats.get(team.id) || { total: 0, running: 0, healthy: 0 }
                return (
                  <article key={team.id} className={styles.teamEntityCard}>
                    <div className={styles.teamEntityHeader}>
                      <div>
                        <h3 className={styles.teamEntityName}>{team.name}</h3>
                        <div className={styles.teamEntityId}>{team.id}</div>
                      </div>
                      {!readOnly ? (
                        <div className={styles.teamEntityActions}>
                          <button
                            type="button"
                            className={styles.btnSmall}
                            onClick={() => handleEdit(team)}
                          >
                            Edit
                          </button>
                          <button
                            type="button"
                            className={`${styles.btnSmall} ${styles.btnSmallDanger}`}
                            onClick={() => requestDelete(team)}
                            disabled={saving}
                          >
                            Delete
                          </button>
                        </div>
                      ) : null}
                    </div>
                    <div className={styles.teamEntityMeta}>
                      <span>
                        <strong>Role:</strong> {team.role || 'Not set'}
                      </span>
                      <span>
                        <strong>Vibe:</strong> {team.vibe || 'Not set'}
                      </span>
                      <span>
                        <strong>Principal:</strong> {team.principal || 'Not set'}
                      </span>
                    </div>
                    {team.description ? (
                      <p className={styles.teamEntityDesc}>{truncateText(team.description, 180)}</p>
                    ) : null}
                    <div className={styles.teamEntityStats}>
                      <span>
                        {stats.total} worker{stats.total === 1 ? '' : 's'}
                      </span>
                      <span>{stats.running} running</span>
                      <span>{stats.healthy} healthy</span>
                    </div>
                  </article>
                )
              })}
            </div>
          )}
        </>
      )}

      <OpenClawDialog
        isOpen={isModalOpen}
        title={editingTeamId ? 'Edit Team' : 'New Team'}
        busy={saving}
        onClose={closeModal}
        footer={
          <>
            <button
              type="button"
              className={styles.btnSecondary}
              onClick={closeModal}
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="button"
              className={styles.btnPrimary}
              onClick={() => void handleSave()}
              disabled={saving}
            >
              {saving ? 'Saving…' : editingTeamId ? 'Update Team' : 'Create Team'}
            </button>
          </>
        }
      >
        {formError ? (
          <OpenClawRequestNotice
            title="Team could not be saved"
            message={formError}
            retryLabel="Retry save"
            onRetry={() => void handleSave()}
          />
        ) : null}
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={nameId}>
              Team Name
            </label>
            <input
              id={nameId}
              className={styles.textInput}
              value={form.name}
              onChange={(event) => updateForm('name', event.target.value)}
              placeholder="Routing Core"
              data-dialog-initial-focus
            />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={teamId}>
              Team ID (optional)
            </label>
            <input
              id={teamId}
              className={styles.textInput}
              value={form.id}
              onChange={(event) => updateForm('id', event.target.value)}
              placeholder="routing-core"
              disabled={Boolean(editingTeamId)}
            />
          </div>
        </div>
        <div className={styles.formRowThree}>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={vibeId}>
              Vibe
            </label>
            <input
              id={vibeId}
              className={styles.textInput}
              value={form.vibe}
              onChange={(event) => updateForm('vibe', event.target.value)}
              placeholder="Calm, decisive"
            />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={roleId}>
              Role
            </label>
            <input
              id={roleId}
              className={styles.textInput}
              value={form.role}
              onChange={(event) => updateForm('role', event.target.value)}
              placeholder="Research pod"
            />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={principalId}>
              Principal
            </label>
            <input
              id={principalId}
              className={styles.textInput}
              value={form.principal}
              onChange={(event) => updateForm('principal', event.target.value)}
              placeholder="Safety first"
            />
          </div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel} htmlFor={descriptionId}>
            Description
          </label>
          <textarea
            id={descriptionId}
            className={styles.textArea}
            value={form.description}
            onChange={(event) => updateForm('description', event.target.value)}
            rows={4}
            placeholder="What this team is responsible for…"
          />
        </div>
      </OpenClawDialog>

      <ConfirmDialog
        isOpen={Boolean(pendingDeleteTeam)}
        eyebrow="Delete OpenClaw team"
        title={`Delete ${pendingDeleteTeam?.name || 'team'}?`}
        description="Assigned workers must be removed or reassigned before this team can be deleted."
        details={
          pendingDeleteTeam ? (
            <div>
              <div>Team ID: {pendingDeleteTeam.id}</div>
              {deleteError ? (
                <div className={styles.confirmInlineError} role="alert">
                  {deleteError} Retry the deletion or cancel.
                </div>
              ) : null}
            </div>
          ) : undefined
        }
        confirmLabel="Delete team"
        pending={saving}
        tone="danger"
        onCancel={() => {
          setDeleteError('')
          setPendingDeleteTeam(null)
        }}
        onConfirm={confirmDelete}
      />
    </div>
  )
}
