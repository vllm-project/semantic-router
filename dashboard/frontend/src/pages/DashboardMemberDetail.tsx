import { useEffect, useState } from 'react'
import PermissionList from '../components/PermissionList'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import { inferenceAccessApi, type AccessUser } from '../utils/inferenceAccessApi'
import { findAccessUserByEmail } from './accessIdentityDirectory'
import type { DashboardMember } from './AccessControlViewTypes'
import DashboardAccessDialog from './DashboardAccessDialog'
import { formatDate } from './AccessControlDetailSupport'
import {
  findLinkedModelUser,
  UnifiedUserDeletionError,
  type UnifiedUserDeletionProgress,
} from './unifiedUserDeletion'
import styles from './AccessControlPage.module.css'

const EMPTY_DELETION_PROGRESS: UnifiedUserDeletionProgress = {
  dashboardLoginRemoved: false,
  modelIdentityDeleted: false,
}

export function DashboardMemberDetail({
  memberId,
  users,
  canManage,
  canManageModelUser,
  canDeleteUnifiedUser,
  onChanged,
  onEditModelAccess,
  onRemoveLogin,
  onDeleteUser,
  onClose,
}: {
  memberId: string
  users: AccessUser[]
  canManage: boolean
  canManageModelUser: boolean
  canDeleteUnifiedUser: boolean
  onChanged: () => void
  onEditModelAccess: (user: AccessUser) => void
  onRemoveLogin: (memberId: string) => Promise<void>
  onDeleteUser: (
    memberId: string,
    modelUserId: string,
    progress: UnifiedUserDeletionProgress,
  ) => Promise<UnifiedUserDeletionProgress>
  onClose: () => void
}) {
  const [member, setMember] = useState<DashboardMember | null>(null)
  const [error, setError] = useState('')
  const [modelUser, setModelUser] = useState<AccessUser | null>(null)
  const [editing, setEditing] = useState(false)
  const [confirming, setConfirming] = useState<'remove-login' | 'delete-user' | null>(null)
  const [actionPending, setActionPending] = useState(false)
  const [actionError, setActionError] = useState('')
  const [deletionProgress, setDeletionProgress] = useState(EMPTY_DELETION_PROGRESS)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen: true,
    onClose,
    dismissible: !actionPending,
  })

  useEffect(() => {
    let cancelled = false
    setError('')
    setActionError('')
    setMember(null)
    setModelUser(null)
    setConfirming(null)
    setDeletionProgress(EMPTY_DELETION_PROGRESS)
    void fetch(`/api/admin/users/${encodeURIComponent(memberId)}`)
      .then(async (response) => {
        if (!response.ok)
          throw new Error((await response.text()) || 'Could not load Dashboard identity')
        return response.json() as Promise<DashboardMember>
      })
      .then((nextMember) => {
        if (!cancelled) setMember(nextMember)
      })
      .catch((nextError) => {
        if (!cancelled) {
          setError(
            nextError instanceof Error ? nextError.message : 'Could not load Dashboard identity',
          )
        }
      })
    return () => {
      cancelled = true
    }
  }, [memberId])

  useEffect(() => {
    let cancelled = false
    if (!member) {
      setModelUser(null)
      return () => {
        cancelled = true
      }
    }
    const linkedUser = findLinkedModelUser(member, users)
    if (linkedUser) {
      setModelUser(linkedUser)
      return () => {
        cancelled = true
      }
    }
    setModelUser(null)
    void findAccessUserByEmail(member.email, inferenceAccessApi.users).then(
      (nextUser) => {
        if (!cancelled) setModelUser(nextUser)
      },
      () => undefined,
    )
    return () => {
      cancelled = true
    }
  }, [member, users])

  const removeLogin = async () => {
    if (!member || actionPending) return
    setActionPending(true)
    setActionError('')
    try {
      await onRemoveLogin(member.id)
      onClose()
    } catch (nextError) {
      setActionError(
        nextError instanceof Error ? nextError.message : 'Could not remove Dashboard login',
      )
    } finally {
      setActionPending(false)
    }
  }

  const deleteUser = async () => {
    if (!member || !modelUser || actionPending) return
    setActionPending(true)
    setActionError('')
    try {
      const progress = await onDeleteUser(member.id, modelUser.id, deletionProgress)
      setDeletionProgress(progress)
      onClose()
    } catch (nextError) {
      if (nextError instanceof UnifiedUserDeletionError) {
        setDeletionProgress(nextError.progress)
      }
      setActionError(nextError instanceof Error ? nextError.message : 'Could not delete user')
    } finally {
      setActionPending(false)
    }
  }

  return (
    <div
      className={styles.detailBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !actionPending) onClose()
      }}
    >
      <aside
        ref={dialogRef}
        className={styles.detailDialog}
        role="dialog"
        aria-modal="true"
        aria-labelledby="member-detail-title"
        aria-busy={actionPending}
        tabIndex={-1}
      >
        <header className={styles.detailHeader}>
          <div className={styles.detailHeaderIdentity}>
            <div className={styles.modalLogo} aria-hidden="true">
              <img src="/vllm.png" alt="" />
            </div>
            <div>
              <span>User</span>
              <h2 id="member-detail-title">{member?.name || 'User details'}</h2>
              <p>{member?.email || 'Loading identity…'}</p>
            </div>
          </div>
          <button
            type="button"
            className={styles.modalClose}
            onClick={onClose}
            disabled={actionPending}
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
          {actionError ? (
            <div className={styles.modalError} role="alert">
              <ProductIcon name="alert" aria-hidden="true" />
              <div>
                <strong>Couldn’t finish</strong>
                <p>{actionError}</p>
                {deletionProgress.dashboardLoginRemoved && !deletionProgress.modelIdentityDeleted ? (
                  <small>
                    Dashboard login is removed. Retry to delete the remaining model identity.
                  </small>
                ) : null}
              </div>
            </div>
          ) : null}
          {!member && !error ? (
            <ProductLoadingState compact label="Loading user details" />
          ) : null}
          {member ? (
            <>
              <div className={styles.detailMetrics}>
                <article>
                  <span>Role</span>
                  <strong>{member.role}</strong>
                </article>
                <article>
                  <span>Status</span>
                  <strong>{member.status}</strong>
                </article>
                <article>
                  <span>Last sign-in</span>
                  <strong>
                    {formatDate(
                      member.lastLoginAt
                        ? new Date(member.lastLoginAt * 1000).toISOString()
                        : undefined,
                    )}
                  </strong>
                </article>
                <article>
                  <span>Teams</span>
                  <strong>{modelUser?.memberships.length || 'None'}</strong>
                </article>
              </div>
              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Authorization</span>
                  <h3>Effective permissions</h3>
                </div>
                <PermissionList permissions={member.permissions || []} />
              </section>
              {modelUser ? (
                <section className={styles.detailSection}>
                  <div className={styles.detailSectionHeading}>
                    <span>Model access</span>
                    <h3>Inherited policy</h3>
                  </div>
                  <dl className={styles.detailGrid}>
                    <div>
                      <dt>Teams</dt>
                      <dd>{modelUser.memberships.length || 'None'}</dd>
                    </div>
                    <div>
                      <dt>API policy</dt>
                      <dd>
                        {modelUser.accessGroupIds.length
                          ? `${modelUser.accessGroupIds.length} user overrides`
                          : 'Team defaults'}
                      </dd>
                    </div>
                    <div>
                      <dt>Budget</dt>
                      <dd>{modelUser.budgetId ? 'User override' : 'Team default'}</dd>
                    </div>
                    <div>
                      <dt>Status</dt>
                      <dd>{modelUser.status}</dd>
                    </div>
                  </dl>
                </section>
              ) : null}
              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Identity</span>
                  <h3>Account details</h3>
                </div>
                <dl className={styles.detailGrid}>
                  <div>
                    <dt>User ID</dt>
                    <dd>
                      <code>{member.id}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Created</dt>
                    <dd>
                      {member.createdAt
                        ? formatDate(new Date(member.createdAt * 1000).toISOString())
                        : 'Unknown'}
                    </dd>
                  </div>
                  <div className={styles.detailGridWide}>
                    <dt>Model access</dt>
                    <dd>{modelUser ? 'Linked model identity' : 'Dashboard login only'}</dd>
                  </div>
                </dl>
              </section>
            </>
          ) : null}
        </div>
        <footer className={styles.detailFooter}>
          {member && confirming === 'remove-login' ? (
            <div className={styles.detailConfirm} role="alert">
              <span>Remove Dashboard login? Model access stays active.</span>
              <button
                type="button"
                onClick={() => setConfirming(null)}
                disabled={actionPending}
              >
                Cancel
              </button>
              <button type="button" onClick={() => void removeLogin()} disabled={actionPending}>
                <ProductIcon name="trash" />
                {actionPending ? 'Removing…' : 'Remove login'}
              </button>
            </div>
          ) : member && modelUser && confirming === 'delete-user' ? (
            <div className={styles.detailConfirm} role="alert">
              <span>
                {deletionProgress.dashboardLoginRemoved
                  ? 'Delete the remaining model identity?'
                  : 'Delete this user, Dashboard login, and model identity?'}
              </span>
              <button
                type="button"
                onClick={() => setConfirming(null)}
                disabled={actionPending}
              >
                Cancel
              </button>
              <button type="button" onClick={() => void deleteUser()} disabled={actionPending}>
                <ProductIcon name="trash" />
                {actionPending ? 'Deleting…' : 'Delete user'}
              </button>
            </div>
          ) : member &&
            (canManage || (modelUser && (canManageModelUser || canDeleteUnifiedUser))) ? (
            <>
              {modelUser && canDeleteUnifiedUser ? (
                <button
                  type="button"
                  className={styles.dangerButton}
                  onClick={() => {
                    setActionError('')
                    setConfirming('delete-user')
                  }}
                >
                  <ProductIcon name="trash" /> Delete user
                </button>
              ) : null}
              {canManage ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => {
                    setActionError('')
                    setConfirming('remove-login')
                  }}
                >
                  <ProductIcon name="power" /> Remove login
                </button>
              ) : null}
              {modelUser && canManageModelUser ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => onEditModelAccess(modelUser)}
                >
                  <ProductIcon name="shield" /> Model access
                </button>
              ) : null}
              {canManage ? (
                <button
                  type="button"
                  className={styles.secondaryButton}
                  onClick={() => setEditing(true)}
                >
                  <ProductIcon name="edit" /> Manage login
                </button>
              ) : null}
            </>
          ) : null}
          <button
            type="button"
            className={styles.primaryButton}
            onClick={onClose}
            disabled={actionPending}
          >
            <ProductIcon name="check" /> Done
          </button>
        </footer>
      </aside>
      {editing && member ? (
        <DashboardAccessDialog
          member={member}
          onClose={() => setEditing(false)}
          onChanged={() => {
            setEditing(false)
            onChanged()
            onClose()
          }}
        />
      ) : null}
    </div>
  )
}
