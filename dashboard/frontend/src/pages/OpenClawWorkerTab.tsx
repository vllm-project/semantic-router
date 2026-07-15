import React, { useDeferredValue, useEffect, useId, useMemo, useRef, useState } from 'react'

import ConfirmDialog from '../components/ConfirmDialog'
import {
  filterAndSortOpenClawWorkers,
  getOpenClawPageCount,
  getOpenClawWorkerHealth,
  OPENCLAW_WORKERS_PAGE_SIZE,
  paginateOpenClawItems,
  type WorkerCatalogSort,
  type WorkerHealthFilter,
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
import { WorkerProvisionWizard } from './OpenClawWorkerProvisionWizard'

interface WorkerTabProps {
  containers: OpenClawStatus[]
  readOnly: boolean
  teams: TeamProfile[]
  workersError?: string | null
  workersLoading?: boolean
  onProvisioned: () => void
  onRetryWorkers?: () => void
  onSwitchToStatus: () => void
  onSwitchToTeam: () => void
}

export const WorkerTab: React.FC<WorkerTabProps> = ({
  containers,
  teams,
  onProvisioned,
  onRetryWorkers,
  onSwitchToTeam,
  onSwitchToStatus,
  readOnly,
  workersError,
  workersLoading = false,
}) => {
  const teamId = useId()
  const emojiId = useId()
  const nameId = useId()
  const roleId = useId()
  const vibeId = useId()
  const principlesId = useId()
  const [searchQuery, setSearchQuery] = useState('')
  const deferredSearch = useDeferredValue(searchQuery)
  const [healthFilter, setHealthFilter] = useState<WorkerHealthFilter>('all')
  const [teamFilter, setTeamFilter] = useState('all')
  const [sort, setSort] = useState<WorkerCatalogSort>('name-asc')
  const [page, setPage] = useState(1)
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false)
  const [isEditModalOpen, setIsEditModalOpen] = useState(false)
  const [wizardBusy, setWizardBusy] = useState(false)
  const [saving, setSaving] = useState(false)
  const [editError, setEditError] = useState('')
  const [deleteError, setDeleteError] = useState('')
  const [editingWorker, setEditingWorker] = useState<OpenClawStatus | null>(null)
  const [pendingDeleteWorker, setPendingDeleteWorker] = useState<OpenClawStatus | null>(null)
  const [editForm, setEditForm] = useState({
    teamId: '',
    name: '',
    emoji: '',
    role: '',
    vibe: '',
    principles: '',
  })
  const mutationRequestRef = useRef<LatestOpenClawRequest | null>(null)
  if (!mutationRequestRef.current) mutationRequestRef.current = createLatestOpenClawRequest()

  const filteredWorkers = useMemo(
    () => filterAndSortOpenClawWorkers(containers, deferredSearch, healthFilter, teamFilter, sort),
    [containers, deferredSearch, healthFilter, sort, teamFilter],
  )
  const pageCount = getOpenClawPageCount(filteredWorkers.length, OPENCLAW_WORKERS_PAGE_SIZE)
  const safePage = Math.min(page, pageCount)
  const visibleWorkers = paginateOpenClawItems(
    filteredWorkers,
    safePage,
    OPENCLAW_WORKERS_PAGE_SIZE,
  )

  useEffect(() => setPage(1), [deferredSearch, healthFilter, sort, teamFilter])
  useEffect(() => {
    if (page > pageCount) setPage(pageCount)
  }, [page, pageCount])
  useEffect(() => () => mutationRequestRef.current?.cancel(), [])
  useEffect(() => {
    if (!readOnly) return
    mutationRequestRef.current?.cancel()
    setIsCreateModalOpen(false)
    setIsEditModalOpen(false)
    setEditingWorker(null)
    setPendingDeleteWorker(null)
    setSaving(false)
    setEditError('')
    setDeleteError('')
  }, [readOnly])

  const openCreateModal = () => {
    if (readOnly) return
    setWizardBusy(false)
    setIsCreateModalOpen(true)
  }

  const closeCreateModal = () => {
    if (!wizardBusy) setIsCreateModalOpen(false)
  }

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
    setEditError('')
    setIsEditModalOpen(true)
  }

  const closeEditModal = () => {
    if (saving) return
    setIsEditModalOpen(false)
    setEditingWorker(null)
    setEditError('')
  }

  const updateEditForm = (field: keyof typeof editForm, value: string) => {
    setEditForm((current) => ({ ...current, [field]: value }))
  }

  const handleUpdateWorker = async () => {
    if (readOnly || !editingWorker) return
    if (!editForm.teamId.trim()) {
      setEditError('Worker team is required.')
      return
    }
    const worker = editingWorker
    await mutationRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<OpenClawStatus>(
          `/api/openclaw/workers/${encodeURIComponent(worker.containerName)}`,
          {
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
          },
          signal,
        ),
      {
        onStart: () => {
          setSaving(true)
          setEditError('')
        },
        onSuccess: () => {
          setIsEditModalOpen(false)
          setEditingWorker(null)
          onProvisioned()
        },
        onError: (error) => {
          setEditError(getOpenClawErrorMessage(error, 'Failed to update the worker.'))
        },
        onFinish: () => setSaving(false),
      },
    )
  }

  const requestDeleteWorker = (worker: OpenClawStatus) => {
    if (readOnly) return
    setDeleteError('')
    setPendingDeleteWorker(worker)
  }

  const confirmDeleteWorker = async () => {
    const worker = pendingDeleteWorker
    if (readOnly || !worker) return
    await mutationRequestRef.current?.run(
      (signal) =>
        fetchOpenClawJSON<void>(
          `/api/openclaw/workers/${encodeURIComponent(worker.containerName)}`,
          { method: 'DELETE' },
          signal,
        ),
      {
        onStart: () => {
          setSaving(true)
          setDeleteError('')
        },
        onSuccess: () => {
          if (editingWorker?.containerName === worker.containerName) {
            setIsEditModalOpen(false)
            setEditingWorker(null)
          }
          setPendingDeleteWorker(null)
          onProvisioned()
        },
        onError: (error) => {
          setDeleteError(getOpenClawErrorMessage(error, 'Failed to delete the worker.'))
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
            New Worker
          </button>
        ) : null}
      </div>

      {workersError ? (
        <OpenClawRequestNotice
          title="Workers are unavailable"
          message={workersError}
          onRetry={onRetryWorkers}
        />
      ) : null}

      {workersLoading && containers.length === 0 ? (
        <div className={styles.loading} role="status">
          <div className={styles.spinner} />
          <p>Loading workers…</p>
        </div>
      ) : (
        <>
          <OpenClawCatalogControls
            searchLabel="Search workers"
            searchValue={searchQuery}
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
            itemCount={filteredWorkers.length}
            totalCount={containers.length}
            itemLabel="workers"
            page={safePage}
            pageSize={OPENCLAW_WORKERS_PAGE_SIZE}
            onSearchChange={setSearchQuery}
            onFilterChange={(value) => setHealthFilter(value as WorkerHealthFilter)}
            onSortChange={(value) => setSort(value as WorkerCatalogSort)}
            onPageChange={setPage}
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

          {visibleWorkers.length === 0 ? (
            <div className={styles.emptyState}>
              <div className={styles.emptyStateText}>
                {containers.length === 0
                  ? readOnly
                    ? 'No OpenClaw workers are configured.'
                    : 'No workers created yet.'
                  : 'No workers match the current search, team, and health filters.'}
              </div>
            </div>
          ) : (
            <div className={styles.agentGrid}>
              {visibleWorkers.map((worker) => {
                const name = worker.agentName?.trim() || worker.containerName
                const emoji = worker.agentEmoji?.trim() || '\u{1F916}'
                const health = getOpenClawWorkerHealth(worker)
                return (
                  <article key={worker.containerName} className={styles.agentCard}>
                    <div className={styles.agentCardHeader}>
                      <div className={styles.agentAvatar}>{emoji}</div>
                      <div className={styles.agentHeaderMeta}>
                        <div className={styles.agentName}>{name}</div>
                        <div className={styles.agentContainerRef}>{worker.containerName}</div>
                        <div className={styles.teamTag}>
                          {worker.teamName?.trim() || 'Unassigned'}
                        </div>
                      </div>
                      <span className={`${styles.healthBadge} ${styles[`healthBadge_${health}`]}`}>
                        {health === 'healthy'
                          ? 'Healthy'
                          : health === 'starting'
                            ? 'Starting'
                            : 'Stopped'}
                      </span>
                    </div>
                    <div className={styles.agentBody}>
                      <WorkerFact label="Role" value={worker.agentRole?.trim() || 'Not set'} />
                      <WorkerFact label="Vibe" value={worker.agentVibe?.trim() || 'Not set'} />
                      <WorkerFact label="Team" value={worker.teamName?.trim() || 'Unassigned'} />
                      <WorkerFact
                        label="Principal"
                        value={truncateText(worker.agentPrinciples, 160) || 'Not set'}
                      />
                    </div>
                    <div className={styles.agentFooter}>
                      <div className={styles.entityRowActions}>
                        {!readOnly ? (
                          <button
                            type="button"
                            className={styles.btnSmall}
                            onClick={() => openEditModal(worker)}
                          >
                            Edit
                          </button>
                        ) : null}
                        <button
                          type="button"
                          className={styles.btnSmall}
                          onClick={onSwitchToStatus}
                        >
                          Status
                        </button>
                        {!readOnly ? (
                          <button
                            type="button"
                            className={`${styles.btnSmall} ${styles.btnSmallDanger}`}
                            onClick={() => requestDeleteWorker(worker)}
                            disabled={saving}
                          >
                            Delete
                          </button>
                        ) : null}
                      </div>
                      {worker.createdAt ? (
                        <span className={styles.agentTimestamp}>
                          Created {new Date(worker.createdAt).toLocaleString()}
                        </span>
                      ) : null}
                    </div>
                  </article>
                )
              })}
            </div>
          )}
        </>
      )}

      <OpenClawDialog
        isOpen={isCreateModalOpen}
        title="New Worker"
        wide
        busy={wizardBusy}
        onClose={closeCreateModal}
      >
        <WorkerProvisionWizard
          teams={teams}
          onBusyChange={setWizardBusy}
          onProvisioned={onProvisioned}
          onSwitchToTeam={onSwitchToTeam}
          onSwitchToStatus={onSwitchToStatus}
          onCreated={() => setIsCreateModalOpen(false)}
        />
      </OpenClawDialog>

      <OpenClawDialog
        isOpen={Boolean(isEditModalOpen && editingWorker)}
        title="Edit Worker"
        busy={saving}
        onClose={closeEditModal}
        footer={
          <>
            <button
              type="button"
              className={styles.btnSecondary}
              onClick={closeEditModal}
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="button"
              className={styles.btnPrimary}
              onClick={() => void handleUpdateWorker()}
              disabled={saving}
            >
              {saving ? 'Saving…' : 'Update Worker'}
            </button>
          </>
        }
      >
        {editError ? (
          <OpenClawRequestNotice
            title="Worker could not be updated"
            message={editError}
            retryLabel="Retry update"
            onRetry={() => void handleUpdateWorker()}
          />
        ) : null}
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={teamId}>
              Team
            </label>
            <select
              id={teamId}
              className={styles.selectInput}
              value={editForm.teamId}
              onChange={(event) => updateEditForm('teamId', event.target.value)}
              data-dialog-initial-focus
            >
              <option value="">Select a team…</option>
              {teams.map((team) => (
                <option key={team.id} value={team.id}>
                  {team.name}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={emojiId}>
              Emoji
            </label>
            <input
              id={emojiId}
              className={styles.textInput}
              value={editForm.emoji}
              onChange={(event) => updateEditForm('emoji', event.target.value)}
              placeholder="🤖"
            />
          </div>
        </div>
        <div className={styles.formRow}>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={nameId}>
              Worker Name
            </label>
            <input
              id={nameId}
              className={styles.textInput}
              value={editForm.name}
              onChange={(event) => updateEditForm('name', event.target.value)}
              placeholder="Atlas"
            />
          </div>
          <div className={styles.formGroup}>
            <label className={styles.formLabel} htmlFor={roleId}>
              Role
            </label>
            <input
              id={roleId}
              className={styles.textInput}
              value={editForm.role}
              onChange={(event) => updateEditForm('role', event.target.value)}
              placeholder="AI operations engineer"
            />
          </div>
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel} htmlFor={vibeId}>
            Vibe
          </label>
          <input
            id={vibeId}
            className={styles.textInput}
            value={editForm.vibe}
            onChange={(event) => updateEditForm('vibe', event.target.value)}
            placeholder="Calm, precise, opinionated"
          />
        </div>
        <div className={styles.formGroup}>
          <label className={styles.formLabel} htmlFor={principlesId}>
            Principal
          </label>
          <textarea
            id={principlesId}
            className={styles.textArea}
            value={editForm.principles}
            onChange={(event) => updateEditForm('principles', event.target.value)}
            rows={5}
            placeholder="Core truths and operating principles…"
          />
        </div>
      </OpenClawDialog>

      <ConfirmDialog
        isOpen={Boolean(pendingDeleteWorker)}
        eyebrow="Delete OpenClaw worker"
        title={`Delete ${pendingDeleteWorker?.agentName || pendingDeleteWorker?.containerName || 'worker'}?`}
        description="The worker container and its OpenClaw runtime state will be removed."
        details={
          pendingDeleteWorker ? (
            <div>
              <div>Container: {pendingDeleteWorker.containerName}</div>
              {deleteError ? (
                <div className={styles.confirmInlineError} role="alert">
                  {deleteError} Retry the deletion or cancel.
                </div>
              ) : null}
            </div>
          ) : undefined
        }
        confirmLabel="Delete worker"
        pending={saving}
        tone="danger"
        onCancel={() => {
          setDeleteError('')
          setPendingDeleteWorker(null)
        }}
        onConfirm={confirmDeleteWorker}
      />
    </div>
  )
}

function WorkerFact({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.agentMetaRow}>
      <span className={styles.agentMetaLabel}>{label}</span>
      <span className={styles.agentMetaValue}>{value}</span>
    </div>
  )
}
