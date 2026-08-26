import { useEffect, useMemo, useRef, useState } from 'react'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import {
  inferenceAccessApi,
  type AccessAPIKey,
  type AccessAssignment,
  type AccessBudget,
  type AccessGroup,
  type AccessPage,
  type AccessTeam,
  type AccessUser,
  type TeamMembership,
  type UsageSummary,
} from '../utils/inferenceAccessApi'
import {
  costCoverageLabel,
  formatCosts,
  formatNumber,
  rateLimitRuleLabel,
} from './AccessControlDetailSupport'
import {
  AccessGroupResourceTags,
  createEntityPolicyNameResolver,
  formatEntityPolicyNames,
  resolveEntityPolicyNames,
  type EntityPolicyNameResolver,
} from './AccessEntityDetailSupport'
import styles from './AccessControlPage.module.css'

export type EntityDetailKind = 'user' | 'team' | 'group' | 'budget'
export type EntityDetailValue = AccessUser | AccessTeam | AccessGroup | AccessBudget

const ENTITY_LABELS: Record<EntityDetailKind, string> = {
  user: 'User',
  team: 'Team',
  group: 'Access group',
  budget: 'Budget',
}

type EntityRelationKind =
  | 'memberships'
  | 'members'
  | 'ownedKeys'
  | 'accessAssignments'
  | 'budgetAssignments'

interface EntityRelationships {
  memberships: AccessPage<TeamMembership> | null
  members: AccessPage<TeamMembership> | null
  ownedKeys: AccessPage<AccessAPIKey> | null
  accessAssignments: AccessPage<AccessAssignment> | null
  budgetAssignments: AccessPage<AccessAssignment> | null
}

const appendPage = <T,>(current: AccessPage<T>, next: AccessPage<T>): AccessPage<T> => ({
  ...next,
  items: [...current.items, ...next.items],
  total: current.total,
})

const assignmentLabel = (assignment: AccessAssignment) =>
  `${assignment.subjectType === 'api_key' ? 'API key' : assignment.subjectType} · ${assignment.subjectId}`

async function safeRelationPage<T>(request: Promise<AccessPage<T>>): Promise<AccessPage<T> | null> {
  return request.catch(() => null)
}

async function loadEntityRelationships(
  kind: EntityDetailKind,
  id: string,
): Promise<EntityRelationships> {
  const emptyMemberships = Promise.resolve(null as AccessPage<TeamMembership> | null)
  const emptyKeys = Promise.resolve(null as AccessPage<AccessAPIKey> | null)
  const emptyAssignments = Promise.resolve(null as AccessPage<AccessAssignment> | null)
  const [memberships, members, ownedKeys, accessAssignments, budgetAssignments] = await Promise.all(
    [
      kind === 'user' ? safeRelationPage(inferenceAccessApi.userMemberships(id)) : emptyMemberships,
      kind === 'team' ? safeRelationPage(inferenceAccessApi.teamMembers(id)) : emptyMemberships,
      kind === 'user' || kind === 'team'
        ? safeRelationPage(inferenceAccessApi.ownedKeys(kind, id))
        : emptyKeys,
      kind === 'budget'
        ? emptyAssignments
        : safeRelationPage(
            inferenceAccessApi.accessAssignments(
              kind === 'group'
                ? { policyId: id }
                : {
                    subjectType: kind === 'user' ? 'user' : kind === 'team' ? 'team' : 'api_key',
                    subjectId: id,
                  },
            ),
          ),
      kind === 'group'
        ? emptyAssignments
        : safeRelationPage(
            inferenceAccessApi.budgetAssignments(
              kind === 'budget'
                ? { policyId: id }
                : {
                    subjectType: kind === 'user' ? 'user' : kind === 'team' ? 'team' : 'api_key',
                    subjectId: id,
                  },
            ),
          ),
    ],
  )
  return { memberships, members, ownedKeys, accessAssignments, budgetAssignments }
}

function loadEntityRelationship(
  kind: EntityDetailKind,
  id: string,
  relation: EntityRelationKind,
  cursor: string,
): Promise<AccessPage<TeamMembership | AccessAPIKey | AccessAssignment>> {
  const params = { cursor, limit: 12, includeTotal: false }
  switch (relation) {
    case 'memberships':
      return inferenceAccessApi.userMemberships(id, params)
    case 'members':
      return inferenceAccessApi.teamMembers(id, params)
    case 'ownedKeys':
      return inferenceAccessApi.ownedKeys(kind as 'user' | 'team', id, params)
    case 'accessAssignments':
      return inferenceAccessApi.accessAssignments(
        kind === 'group'
          ? { policyId: id }
          : { subjectType: kind === 'user' ? 'user' : 'team', subjectId: id },
        params,
      )
    case 'budgetAssignments':
      return inferenceAccessApi.budgetAssignments(
        kind === 'budget'
          ? { policyId: id }
          : { subjectType: kind === 'user' ? 'user' : 'team', subjectId: id },
        params,
      )
  }
}

export function AccessEntityDetail({
  kind,
  id,
  canEdit,
  canDelete,
  selfService = false,
  selfUserId,
  resourceName,
  onEdit,
  onDelete,
  onClose,
}: {
  kind: EntityDetailKind
  id: string
  canEdit: boolean
  canDelete: boolean
  selfService?: boolean
  selfUserId: string
  resourceName: (resourceType: 'model' | 'entrypoint', resourceId: string) => string
  onEdit: (kind: EntityDetailKind, item: EntityDetailValue) => void
  onDelete: (kind: EntityDetailKind, id: string) => Promise<void>
  onClose: () => void
}) {
  const [item, setItem] = useState<EntityDetailValue | null>(null)
  const [usage, setUsage] = useState<UsageSummary | null>(null)
  const [memberships, setMemberships] = useState<AccessPage<TeamMembership> | null>(null)
  const [members, setMembers] = useState<AccessPage<TeamMembership> | null>(null)
  const [ownedKeys, setOwnedKeys] = useState<AccessPage<AccessAPIKey> | null>(null)
  const [accessAssignments, setAccessAssignments] = useState<AccessPage<AccessAssignment> | null>(
    null,
  )
  const [budgetAssignments, setBudgetAssignments] = useState<AccessPage<AccessAssignment> | null>(
    null,
  )
  const [accessPolicyNames, setAccessPolicyNames] = useState<ReadonlyMap<string, string>>(
    () => new Map(),
  )
  const [budgetPolicyNames, setBudgetPolicyNames] = useState<ReadonlyMap<string, string>>(
    () => new Map(),
  )
  const policyNameResolverRef = useRef<EntityPolicyNameResolver | null>(null)
  if (!policyNameResolverRef.current) {
    policyNameResolverRef.current = createEntityPolicyNameResolver(async (policyKind, policyId) => {
      const policy =
        policyKind === 'access'
          ? await inferenceAccessApi.groupSummary(policyId)
          : await inferenceAccessApi.budgetSummary(policyId)
      return policy.name
    })
  }
  const policyNameResolver = policyNameResolverRef.current
  const [error, setError] = useState('')
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [deletePending, setDeletePending] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !deletePending,
  })
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
    setMemberships(null)
    setMembers(null)
    setOwnedKeys(null)
    setAccessAssignments(null)
    setBudgetAssignments(null)
    void entityRequest
      .then((next) => {
        if (cancelled) return
        setItem(next)
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
        if (!cancelled) setUsage(null)
      })
    void loadEntityRelationships(kind, id)
      .then((relations) => {
        if (cancelled) return
        setMemberships(relations.memberships)
        setMembers(relations.members)
        setOwnedKeys(relations.ownedKeys)
        setAccessAssignments(relations.accessAssignments)
        setBudgetAssignments(relations.budgetAssignments)
      })
      .catch(() => {
        // The entity itself remains useful when one related resource family is hidden.
      })
    return () => {
      cancelled = true
    }
  }, [id, kind, selfService])

  const accessPolicyIds = useMemo(
    () => accessAssignments?.items.map((assignment) => assignment.policyId) ?? [],
    [accessAssignments],
  )
  const budgetPolicyIds = useMemo(
    () => budgetAssignments?.items.map((assignment) => assignment.policyId) ?? [],
    [budgetAssignments],
  )

  useEffect(() => {
    let cancelled = false
    void resolveEntityPolicyNames('access', accessPolicyIds, policyNameResolver).then((names) => {
      if (!cancelled) setAccessPolicyNames(names)
    })
    return () => {
      cancelled = true
    }
  }, [accessPolicyIds, policyNameResolver])

  useEffect(() => {
    let cancelled = false
    void resolveEntityPolicyNames('budget', budgetPolicyIds, policyNameResolver).then((names) => {
      if (!cancelled) setBudgetPolicyNames(names)
    })
    return () => {
      cancelled = true
    }
  }, [budgetPolicyIds, policyNameResolver])

  const entityLabel = ENTITY_LABELS[kind]
  const title = item?.name || `${entityLabel} details`
  const user = kind === 'user' ? (item as AccessUser | null) : null
  const team = kind === 'team' ? (item as AccessTeam | null) : null
  const group = kind === 'group' ? (item as AccessGroup | null) : null
  const budget = kind === 'budget' ? (item as AccessBudget | null) : null
  const effectiveCanEdit = Boolean(
    canEdit ||
      (selfService &&
        team?.members.some(
          (membership) => membership.userId === selfUserId && membership.role === 'admin',
        )),
  )
  const remove = async () => {
    if (!item || deletePending) return
    setDeletePending(true)
    setDeleteError('')
    try {
      await onDelete(kind, item.id)
      onClose()
    } catch (nextError) {
      setDeleteError(
        nextError instanceof Error ? nextError.message : `Could not delete ${entityLabel}`,
      )
    } finally {
      setDeletePending(false)
    }
  }
  const loadMore = async (
    relation: 'memberships' | 'members' | 'ownedKeys' | 'accessAssignments' | 'budgetAssignments',
  ) => {
    const current = { memberships, members, ownedKeys, accessAssignments, budgetAssignments }[
      relation
    ]
    if (!current?.hasMore || !current.nextCursor) return
    const next = await loadEntityRelationship(kind, id, relation, current.nextCursor)
    switch (relation) {
      case 'memberships':
        setMemberships((previous) =>
          previous ? appendPage(previous, next as AccessPage<TeamMembership>) : previous,
        )
        break
      case 'members':
        setMembers((previous) =>
          previous ? appendPage(previous, next as AccessPage<TeamMembership>) : previous,
        )
        break
      case 'ownedKeys':
        setOwnedKeys((previous) =>
          previous ? appendPage(previous, next as AccessPage<AccessAPIKey>) : previous,
        )
        break
      case 'accessAssignments':
        setAccessAssignments((previous) =>
          previous ? appendPage(previous, next as AccessPage<AccessAssignment>) : previous,
        )
        break
      case 'budgetAssignments':
        setBudgetAssignments((previous) =>
          previous ? appendPage(previous, next as AccessPage<AccessAssignment>) : previous,
        )
        break
    }
  }

  return (
    <div
      className={`${styles.detailBackdrop} ${styles.entityDetailBackdrop}`}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !deletePending) onClose()
      }}
    >
      <section
        ref={dialogRef}
        className={styles.entityDetailDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="entity-detail-title"
        aria-busy={deletePending}
        tabIndex={-1}
      >
        <header className={styles.detailHeader}>
          <div className={styles.detailHeaderIdentity}>
            <div className={styles.modalLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span>{entityLabel}</span>
              <h2 id="entity-detail-title">{title}</h2>
              <p>
                {user?.email ||
                  team?.description ||
                  group?.description ||
                  (budget ? budget.description || 'Reusable quota' : 'Loading…')}
              </p>
            </div>
          </div>
          <button
            type="button"
            className={styles.modalClose}
            onClick={onClose}
            disabled={deletePending}
            aria-label="Close"
          >
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
          {deleteError ? (
            <div className={styles.modalError} role="alert">
              <ProductIcon name="alert" aria-hidden="true" />
              <div>
                <strong>Couldn’t delete {entityLabel.toLowerCase()}</strong>
                <p>{deleteError}</p>
              </div>
            </div>
          ) : null}
          {!item && !error ? (
            <ProductLoadingState compact label={`Loading ${entityLabel.toLowerCase()} details`} />
          ) : null}
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
                <span>Overview</span>
                <h3>Access at a glance</h3>
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
                      <dd>{ownedKeys?.total ?? '—'}</dd>
                    </div>
                    <div>
                      <dt>Teams</dt>
                      <dd>
                        {memberships?.items.length
                          ? memberships.items
                              .map((value) => value.teamName || value.teamId)
                              .join(', ')
                          : 'None'}
                      </dd>
                    </div>
                    <div>
                      <dt>Model access</dt>
                      <dd>
                        {accessAssignments?.items.length
                          ? formatEntityPolicyNames(accessAssignments.items, accessPolicyNames)
                          : 'Inherited from Team context'}
                      </dd>
                    </div>
                    <div>
                      <dt>Budget</dt>
                      <dd>
                        {budgetAssignments?.items.length
                          ? formatEntityPolicyNames(budgetAssignments.items, budgetPolicyNames)
                          : 'Inherited from Team context'}
                      </dd>
                    </div>
                  </>
                ) : null}
                {team ? (
                  <>
                    <div>
                      <dt>Members</dt>
                      <dd>{members?.total ?? '—'}</dd>
                    </div>
                    <div>
                      <dt>API keys</dt>
                      <dd>{ownedKeys?.total ?? '—'}</dd>
                    </div>
                    <div>
                      <dt>Model access</dt>
                      <dd>
                        {accessAssignments?.items.length
                          ? formatEntityPolicyNames(accessAssignments.items, accessPolicyNames)
                          : 'None'}
                      </dd>
                    </div>
                    <div>
                      <dt>Quota</dt>
                      <dd>
                        {budgetAssignments?.items.length
                          ? formatEntityPolicyNames(budgetAssignments.items, budgetPolicyNames)
                          : 'Inherited'}
                      </dd>
                    </div>
                  </>
                ) : null}
                {group ? (
                  <>
                    <div>
                      <dt>Assignments</dt>
                      <dd>{accessAssignments?.total ?? '—'}</dd>
                    </div>
                    <div className={styles.detailGridWide}>
                      <dt>Visible models</dt>
                      <dd className={styles.detailTags}>
                        <AccessGroupResourceTags
                          resources={group.resources}
                          resourceName={resourceName}
                        />
                      </dd>
                    </div>
                  </>
                ) : null}
                {budget ? (
                  <>
                    <div>
                      <dt>Assignments</dt>
                      <dd>{budgetAssignments?.total ?? '—'}</dd>
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
                  </>
                ) : null}
              </dl>
              {(user || team) && (accessAssignments?.hasMore || budgetAssignments?.hasMore) ? (
                <div className={styles.detailTags}>
                  {accessAssignments?.hasMore ? (
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={() => void loadMore('accessAssignments')}
                    >
                      More access groups
                    </button>
                  ) : null}
                  {budgetAssignments?.hasMore ? (
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={() => void loadMore('budgetAssignments')}
                    >
                      More budgets
                    </button>
                  ) : null}
                </div>
              ) : null}
            </section>
          ) : null}
          {user && memberships ? (
            <section className={styles.detailSection}>
              <div className={styles.detailSectionHeading}>
                <span>Teams</span>
                <h3>{memberships.total ? `${memberships.total} memberships` : 'No memberships'}</h3>
              </div>
              <div className={styles.teamMemberList}>
                {memberships.items.map((membership) => (
                  <article key={membership.teamId}>
                    <div className={styles.teamMemberAvatar} aria-hidden="true">
                      {(membership.teamName || 'T').slice(0, 2).toUpperCase()}
                    </div>
                    <div>
                      <strong>{membership.teamName || membership.teamId}</strong>
                      <span>{membership.teamId}</span>
                    </div>
                    <small>{membership.role === 'admin' ? 'Team admin' : 'Member'}</small>
                  </article>
                ))}
              </div>
              {memberships.hasMore ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => void loadMore('memberships')}
                >
                  Load more
                </button>
              ) : null}
            </section>
          ) : null}
          {(user || team) && ownedKeys ? (
            <section className={styles.detailSection}>
              <div className={styles.detailSectionHeading}>
                <span>API keys</span>
                <h3>{ownedKeys.total ? `${ownedKeys.total} owned keys` : 'No owned keys'}</h3>
              </div>
              <div className={styles.teamMemberList}>
                {ownedKeys.items.map((ownedKey) => (
                  <article key={ownedKey.id}>
                    <div className={styles.teamMemberAvatar} aria-hidden="true">
                      <ProductIcon name="key" />
                    </div>
                    <div>
                      <strong>{ownedKey.name}</strong>
                      <span>{ownedKey.id}</span>
                    </div>
                    <small>{ownedKey.status}</small>
                  </article>
                ))}
              </div>
              {ownedKeys.hasMore ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => void loadMore('ownedKeys')}
                >
                  Load more
                </button>
              ) : null}
            </section>
          ) : null}
          {group && accessAssignments ? (
            <section className={styles.detailSection}>
              <div className={styles.detailSectionHeading}>
                <span>Assignments</span>
                <h3>
                  {accessAssignments.total ? `${accessAssignments.total} subjects` : 'Not assigned'}
                </h3>
              </div>
              <div className={styles.detailTags}>
                {accessAssignments.items.map((assignment) => (
                  <code key={assignment.id}>{assignmentLabel(assignment)}</code>
                ))}
              </div>
              {accessAssignments.hasMore ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => void loadMore('accessAssignments')}
                >
                  Load more
                </button>
              ) : null}
            </section>
          ) : null}
          {budget && budgetAssignments ? (
            <section className={styles.detailSection}>
              <div className={styles.detailSectionHeading}>
                <span>Assignments</span>
                <h3>
                  {budgetAssignments.total ? `${budgetAssignments.total} subjects` : 'Not assigned'}
                </h3>
              </div>
              <div className={styles.detailTags}>
                {budgetAssignments.items.map((assignment) => (
                  <code key={assignment.id}>{assignmentLabel(assignment)}</code>
                ))}
              </div>
              {budgetAssignments.hasMore ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => void loadMore('budgetAssignments')}
                >
                  Load more
                </button>
              ) : null}
            </section>
          ) : null}
          {team && members ? (
            <section className={styles.detailSection}>
              <div className={styles.detailSectionHeading}>
                <span>Members</span>
                <h3>{members.total ? `${members.total} people` : 'No members yet'}</h3>
              </div>
              <div className={styles.teamMemberList}>
                {members.items.map((member) => (
                  <article key={member.userId}>
                    <div className={styles.teamMemberAvatar} aria-hidden="true">
                      {(member.userName || 'User')
                        .split(/\s+/)
                        .filter(Boolean)
                        .slice(0, 2)
                        .map((part) => part[0]?.toUpperCase())
                        .join('') || 'U'}
                    </div>
                    <div>
                      <strong>{member.userName || member.userId}</strong>
                      <span>{member.userEmail || member.userId}</span>
                    </div>
                    <small>{member.role === 'admin' ? 'Team admin' : 'Member'}</small>
                  </article>
                ))}
                {!members.items.length ? (
                  <p className={styles.teamMemberEmpty}>Add members from Edit team.</p>
                ) : null}
              </div>
              {members.hasMore ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => void loadMore('members')}
                >
                  Load more
                </button>
              ) : null}
            </section>
          ) : null}
        </div>
        <footer className={styles.detailFooter}>
          {item && canDelete && confirmingDelete ? (
            <div className={styles.detailConfirm} role="alert">
              <span>Delete {item.name}?</span>
              <button
                type="button"
                onClick={() => setConfirmingDelete(false)}
                disabled={deletePending}
              >
                Cancel
              </button>
              <button type="button" onClick={() => void remove()} disabled={deletePending}>
                <ProductIcon name="trash" /> {deletePending ? 'Deleting…' : 'Delete'}
              </button>
            </div>
          ) : item && (effectiveCanEdit || canDelete) ? (
            <>
              {canDelete ? (
                <button
                  type="button"
                  className={styles.dangerButton}
                  onClick={() => {
                    setDeleteError('')
                    setConfirmingDelete(true)
                  }}
                  disabled={deletePending}
                >
                  <ProductIcon name="trash" /> Delete
                </button>
              ) : null}
              {effectiveCanEdit ? (
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
          <button
            type="button"
            className={styles.primaryButton}
            onClick={onClose}
            disabled={deletePending}
          >
            <ProductIcon name="check" /> Done
          </button>
        </footer>
      </section>
    </div>
  )
}
