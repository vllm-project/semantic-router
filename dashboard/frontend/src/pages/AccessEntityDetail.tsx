import { useEffect, useState } from 'react'
import ProductIcon from '../components/ProductIcon'
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
import {
  costCoverageLabel,
  formatCosts,
  formatNumber,
  rateLimitRuleLabel,
} from './AccessControlDetailSupport'
import styles from './AccessControlPage.module.css'

export type EntityDetailKind = 'user' | 'team' | 'group' | 'budget'
export type EntityDetailValue = AccessUser | AccessTeam | AccessGroup | AccessBudget

export function AccessEntityDetail({
  kind,
  id,
  users,
  teams,
  keys,
  groups,
  budgets,
  canEdit,
  canDelete,
  selfService = false,
  onEdit,
  onDelete,
  onClose,
}: {
  kind: EntityDetailKind
  id: string
  users: AccessUser[]
  teams: AccessTeam[]
  keys: AccessAPIKey[]
  groups: AccessGroup[]
  budgets: AccessBudget[]
  canEdit: boolean
  canDelete: boolean
  selfService?: boolean
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
    let cancelled = false
    setError('')
    setItem(null)
    setUsage(null)
    const entityRequest =
      kind === 'user'
        ? inferenceAccessApi.user(id)
        : kind === 'team'
          ? selfService
            ? inferenceAccessApi.selfTeam(id)
            : inferenceAccessApi.team(id)
          : kind === 'group'
            ? inferenceAccessApi.group(id)
            : inferenceAccessApi.budget(id)
    const from = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
    const usageRequest =
      kind === 'user'
        ? inferenceAccessApi.userUsage(id, { from })
        : kind === 'team'
          ? inferenceAccessApi.teamUsage(id, { from })
          : Promise.resolve(null)
    setTeamMembers([])
    void entityRequest
      .then(async (next) => {
        if (cancelled) return
        setItem(next)
        if (kind === 'team') {
          const nextTeam = next as AccessTeam
          const knownMembers = new Map(users.map((user) => [user.id, user]))
          const missingMembers = await Promise.all(
            nextTeam.members
              .map((member) => member.userId)
              .filter((userID) => !knownMembers.has(userID))
              .map((userID) =>
                selfService
                  ? Promise.resolve(null)
                  : inferenceAccessApi.user(userID).catch(() => null),
              ),
          )
          missingMembers.forEach((member) => {
            if (member) knownMembers.set(member.id, member)
          })
          if (cancelled) return
          setTeamMembers(
            nextTeam.members
              .map((member) => knownMembers.get(member.userId))
              .filter((member): member is AccessUser => Boolean(member)),
          )
        }
      })
      .catch((nextError) =>
        !cancelled
          ? setError(nextError instanceof Error ? nextError.message : 'Could not load details')
          : undefined,
      )
    void usageRequest
      .then((nextUsage) => {
        if (!cancelled) setUsage(nextUsage)
      })
      .catch(() => {
        // Detail access remains useful when this principal cannot read analytics.
        if (!cancelled) setUsage(null)
      })
    return () => {
      cancelled = true
    }
  }, [id, kind, selfService, users])

  const title = item?.name || `${kind.charAt(0).toUpperCase()}${kind.slice(1)} details`
  const user = kind === 'user' ? (item as AccessUser | null) : null
  const team = kind === 'team' ? (item as AccessTeam | null) : null
  const group = kind === 'group' ? (item as AccessGroup | null) : null
  const budget = kind === 'budget' ? (item as AccessBudget | null) : null
  const ownedKeys = keys.filter(
    (key) =>
      (key.ownerType === 'user' && key.ownerId === user?.id) ||
      (key.ownerType === 'team' && key.ownerId === team?.id),
  )
  const memberTeams = user
    ? teams.filter((candidate) => candidate.members.some((member) => member.userId === user.id))
    : []
  const linkedKeys = budget ? keys.filter((key) => key.budgetId === budget.id) : []
  const linkedUsers = budget ? users.filter((candidate) => candidate.budgetId === budget.id) : []
  const linkedTeams = budget ? teams.filter((candidate) => candidate.budgetId === budget.id) : []
  const groupUsers = group
    ? users.filter((candidate) => candidate.accessGroupIds.includes(group.id))
    : []
  const groupTeams = group
    ? teams.filter((candidate) => candidate.accessGroupIds.includes(group.id))
    : []
  const groupKeys = group
    ? keys.filter((candidate) => candidate.accessGroupIds.includes(group.id))
    : []

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
                  (budget ? budget.description || 'Reusable quota' : 'Loading…')}
              </p>
            </div>
          </div>
          <button type="button" className={styles.modalClose} onClick={onClose} aria-label="Close">
            <ProductIcon name="close" />
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
                <span>7-day spend</span>
                <strong title={costCoverageLabel(usage.costs)}>{formatCosts(usage.costs)}</strong>
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
                    <div>
                      <dt>Model access</dt>
                      <dd>
                        {user.accessGroupIds.length
                          ? user.accessGroupIds
                              .map((id) => groups.find((group) => group.id === id)?.name || id)
                              .join(', ')
                          : 'Inherited from Team context'}
                      </dd>
                    </div>
                    <div>
                      <dt>Budget</dt>
                      <dd>
                        {user.budgetId
                          ? budgets.find((value) => value.id === user.budgetId)?.name ||
                            user.budgetId
                          : 'Inherited from Team context'}
                      </dd>
                    </div>
                  </>
                ) : null}
                {team ? (
                  <>
                    <div>
                      <dt>Members</dt>
                      <dd>{team.members.length}</dd>
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
                        {budgets.find((value) => value.id === team.budgetId)?.name || team.budgetId}
                      </dd>
                    </div>
                  </>
                ) : null}
                {group ? (
                  <>
                    <div>
                      <dt>Assignments</dt>
                      <dd>{group.assignmentCount}</dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Visible models</dt>
                      <dd className={styles.detailTags}>
                        {group.resources.map((resource) => (
                          <code key={`${resource.resourceType}:${resource.resourceId}`}>
                            {resource.resourceId}
                          </code>
                        ))}
                      </dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Assigned to</dt>
                      <dd className={styles.detailTags}>
                        {[...groupUsers, ...groupTeams, ...groupKeys].length
                          ? [...groupUsers, ...groupTeams, ...groupKeys].map((value) => (
                              <code key={value.id}>{value.name}</code>
                            ))
                          : 'None'}
                      </dd>
                    </div>
                  </>
                ) : null}
                {budget ? (
                  <>
                    <div>
                      <dt>Assignments</dt>
                      <dd>{budget.assignmentCount}</dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Limits</dt>
                      <dd className={styles.detailTags}>
                        {budget.rules.length
                          ? budget.rules.map((rule, index) => (
                              <code key={rule.ruleId || index}>{rateLimitRuleLabel(rule)}</code>
                            ))
                          : 'No limits'}
                      </dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Linked API keys</dt>
                      <dd className={styles.detailTags}>
                        {linkedKeys.length
                          ? linkedKeys.map((key) => <code key={key.id}>{key.name}</code>)
                          : 'None'}
                      </dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Linked users & Teams</dt>
                      <dd className={styles.detailTags}>
                        {[...linkedUsers, ...linkedTeams].length
                          ? [...linkedUsers, ...linkedTeams].map((value) => (
                              <code key={value.id}>{value.name}</code>
                            ))
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
                    <small>
                      {team.members.find((membership) => membership.userId === member.id)?.role ===
                      'admin'
                        ? 'Team admin'
                        : 'Member'}
                    </small>
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
          {item && canDelete && confirmingDelete ? (
            <div className={styles.detailConfirm} role="alert">
              <span>Delete {item.name}?</span>
              <button type="button" onClick={() => setConfirmingDelete(false)}>
                Cancel
              </button>
              <button type="button" onClick={() => onDelete(kind, item.id)}>
                <ProductIcon name="trash" /> Delete
              </button>
            </div>
          ) : item && (canEdit || canDelete) ? (
            <>
              {canDelete ? (
                <button
                  type="button"
                  className={styles.dangerButton}
                  onClick={() => setConfirmingDelete(true)}
                >
                  <ProductIcon name="trash" /> Delete
                </button>
              ) : null}
              {canEdit ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => onEdit(kind, item)}
                >
                  <ProductIcon name="edit" /> Edit
                </button>
              ) : null}
            </>
          ) : null}
          <button type="button" className={styles.primaryButton} onClick={onClose}>
            <ProductIcon name="check" /> Done
          </button>
        </footer>
      </section>
    </div>
  )
}
