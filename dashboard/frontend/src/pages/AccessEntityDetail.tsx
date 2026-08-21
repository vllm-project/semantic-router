import { useEffect, useState } from 'react'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessBudget,
  type AccessGroup,
  type AccessTeam,
  type AccessUser,
  type UsageSummary,
} from '../utils/inferenceAccessApi'
import { formatNumber } from './AccessControlDetailSupport'
import styles from './AccessControlPage.module.css'

export type EntityDetailKind = 'user' | 'team' | 'group' | 'budget'
export type EntityDetailValue = AccessUser | AccessTeam | AccessGroup | AccessBudget

export function AccessEntityDetail({
  kind,
  id,
  users,
  teams,
  keys,
  canManage,
  onEdit,
  onDelete,
  onClose,
}: {
  kind: EntityDetailKind
  id: string
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  canManage: boolean
  onEdit: (kind: EntityDetailKind, item: EntityDetailValue) => void
  onDelete: (kind: EntityDetailKind, id: string) => void
  onClose: () => void
}) {
  const [item, setItem] = useState<EntityDetailValue | null>(null)
  const [usage, setUsage] = useState<UsageSummary | null>(null)
  const [teamMembers, setTeamMembers] = useState<AccessUser[]>([])
  const [error, setError] = useState('')
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen: true, onClose })
  useEffect(() => {
    setError('')
    const entityRequest =
      kind === 'user'
        ? inferenceAccessApi.user(id)
        : kind === 'team'
          ? inferenceAccessApi.team(id)
          : kind === 'group'
            ? inferenceAccessApi.group(id)
            : inferenceAccessApi.budget(id)
    const from = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
    const usageRequest =
      kind === 'user'
        ? inferenceAccessApi.usage({ userId: id, from })
        : kind === 'team'
          ? inferenceAccessApi.usage({ teamId: id, from })
          : Promise.resolve(null)
    setTeamMembers([])
    void Promise.all([entityRequest, usageRequest])
      .then(async ([next, nextUsage]) => {
        setItem(next)
        setUsage(nextUsage)
        if (kind === 'team') {
          const nextTeam = next as AccessTeam
          const knownMembers = new Map(users.map((user) => [user.id, user]))
          const missingMembers = await Promise.all(
            nextTeam.userIds
              .filter((userID) => !knownMembers.has(userID))
              .map((userID) => inferenceAccessApi.user(userID).catch(() => null)),
          )
          missingMembers.forEach((member) => {
            if (member) knownMembers.set(member.id, member)
          })
          setTeamMembers(
            nextTeam.userIds
              .map((userID) => knownMembers.get(userID))
              .filter((member): member is AccessUser => Boolean(member)),
          )
        }
      })
      .catch((nextError) =>
        setError(nextError instanceof Error ? nextError.message : 'Could not load details'),
      )
  }, [id, kind, users])

  const title = item?.name || `${kind.charAt(0).toUpperCase()}${kind.slice(1)} details`
  const user = kind === 'user' ? (item as AccessUser | null) : null
  const team = kind === 'team' ? (item as AccessTeam | null) : null
  const group = kind === 'group' ? (item as AccessGroup | null) : null
  const budget = kind === 'budget' ? (item as AccessBudget | null) : null
  const ownedKeys = keys.filter((key) => key.userId === user?.id || key.teamId === team?.id)
  const memberTeams = user ? teams.filter((candidate) => candidate.userIds.includes(user.id)) : []
  const linkedKeys = budget ? keys.filter((key) => key.budgetId === budget.id) : []

  return (
    <div
      className={`${styles.detailBackdrop} ${styles.entityDetailBackdrop}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <section
        ref={dialogRef}
        className={styles.entityDetailDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="entity-detail-title"
        tabIndex={-1}
      >
        <header className={styles.detailHeader}>
          <div className={styles.detailHeaderIdentity}>
            <div className={styles.modalLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span>{kind}</span>
              <h2 id="entity-detail-title">{title}</h2>
              <p>
                {user?.email ||
                  team?.description ||
                  group?.description ||
                  (budget ? `${budget.scopeType} quota` : 'Loading…')}
              </p>
            </div>
          </div>
          <button type="button" className={styles.modalClose} onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className={styles.detailBody}>
          {error ? (
            <div className={styles.modalError} role="alert">
              <span>!</span>
              <div>
                <strong>Couldn’t load details</strong>
                <p>{error}</p>
              </div>
            </div>
          ) : null}
          {!item && !error ? <div className={styles.detailLoading}>Loading details…</div> : null}
          {usage ? (
            <div className={styles.detailMetrics}>
              <article>
                <span>7-day requests</span>
                <strong>{formatNumber(usage.requests)}</strong>
              </article>
              <article>
                <span>7-day tokens</span>
                <strong>{formatNumber(usage.totalTokens)}</strong>
              </article>
              <article>
                <span>Active keys</span>
                <strong>{formatNumber(usage.activeKeys)}</strong>
              </article>
              <article>
                <span>P95 latency</span>
                <strong>{formatNumber(usage.p95LatencyMs)} ms</strong>
              </article>
            </div>
          ) : null}
          {item ? (
            <section className={styles.detailSection}>
              <div className={styles.detailSectionHeading}>
                <span>Identity & policy</span>
                <h3>Effective access</h3>
              </div>
              <dl className={styles.detailGrid}>
                {'status' in item ? (
                  <div>
                    <dt>Status</dt>
                    <dd>{String(item.status)}</dd>
                  </div>
                ) : null}
                <div>
                  <dt>ID</dt>
                  <dd>
                    <code>{item.id}</code>
                  </dd>
                </div>
                {user ? (
                  <>
                    <div>
                      <dt>API keys</dt>
                      <dd>{ownedKeys.length}</dd>
                    </div>
                    <div>
                      <dt>Teams</dt>
                      <dd>
                        {memberTeams.length
                          ? memberTeams.map((value) => value.name).join(', ')
                          : 'None'}
                      </dd>
                    </div>
                  </>
                ) : null}
                {team ? (
                  <>
                    <div>
                      <dt>Members</dt>
                      <dd>{team.userIds.length}</dd>
                    </div>
                    <div>
                      <dt>API keys</dt>
                      <dd>{ownedKeys.length}</dd>
                    </div>
                    <div>
                      <dt>Access groups</dt>
                      <dd>{team.accessGroupIds.length}</dd>
                    </div>
                    <div>
                      <dt>Team budget</dt>
                      <dd>
                        {team.budget
                          ? `${formatNumber(team.budget.rpm)} RPM · ${formatNumber(team.budget.tpm)} TPM`
                          : 'Not configured'}
                      </dd>
                    </div>
                  </>
                ) : null}
                {group ? (
                  <>
                    <div>
                      <dt>Assignments</dt>
                      <dd>{group.bindings.length}</dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Visible models</dt>
                      <dd className={styles.detailTags}>
                        {group.modelPatterns.map((pattern) => (
                          <code key={pattern}>{pattern}</code>
                        ))}
                      </dd>
                    </div>
                  </>
                ) : null}
                {budget ? (
                  <>
                    <div>
                      <dt>Scope</dt>
                      <dd>
                        {budget.scopeType} · {budget.scopeId || 'all traffic'}
                      </dd>
                    </div>
                    <div>
                      <dt>RPM</dt>
                      <dd>{formatNumber(budget.rpm)}</dd>
                    </div>
                    <div>
                      <dt>TPM</dt>
                      <dd>{formatNumber(budget.tpm)}</dd>
                    </div>
                    <div>
                      <dt>Daily tokens</dt>
                      <dd>{formatNumber(budget.dailyTokens)}</dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Linked API keys</dt>
                      <dd className={styles.detailTags}>
                        {linkedKeys.length
                          ? linkedKeys.map((key) => <code key={key.id}>{key.name}</code>)
                          : 'None'}
                      </dd>
                    </div>
                  </>
                ) : null}
              </dl>
            </section>
          ) : null}
          {team ? (
            <section className={styles.detailSection}>
              <div className={styles.detailSectionHeading}>
                <span>Members</span>
                <h3>{teamMembers.length ? `${teamMembers.length} people` : 'No members yet'}</h3>
              </div>
              <div className={styles.teamMemberList}>
                {teamMembers.map((member) => (
                  <article key={member.id}>
                    <div className={styles.teamMemberAvatar} aria-hidden="true">
                      {member.name
                        .split(/\s+/)
                        .filter(Boolean)
                        .slice(0, 2)
                        .map((part) => part[0]?.toUpperCase())
                        .join('') || 'U'}
                    </div>
                    <div>
                      <strong>{member.name}</strong>
                      <span>{member.email}</span>
                    </div>
                    <small>{member.status}</small>
                  </article>
                ))}
                {!teamMembers.length ? (
                  <p className={styles.teamMemberEmpty}>Add members from Edit team.</p>
                ) : null}
              </div>
            </section>
          ) : null}
        </div>
        <footer className={styles.detailFooter}>
          {item && canManage && confirmingDelete ? (
            <div className={styles.detailConfirm} role="alert">
              <span>Delete {item.name}?</span>
              <button type="button" onClick={() => setConfirmingDelete(false)}>
                Cancel
              </button>
              <button type="button" onClick={() => onDelete(kind, item.id)}>
                Delete
              </button>
            </div>
          ) : item && canManage ? (
            <>
              <button
                type="button"
                className={styles.dangerButton}
                onClick={() => setConfirmingDelete(true)}
              >
                Delete
              </button>
              <button
                type="button"
                className={styles.secondaryButton}
                onClick={() => onEdit(kind, item)}
              >
                Edit
              </button>
            </>
          ) : null}
          <button type="button" className={styles.primaryButton} onClick={onClose}>
            Done
          </button>
        </footer>
      </section>
    </div>
  )
}
