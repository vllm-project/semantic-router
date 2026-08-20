import { useEffect, useState } from 'react'
import PermissionList from '../components/PermissionList'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import type { DashboardMember } from './AccessControlViewTypes'
import DashboardAccessDialog from './DashboardAccessDialog'
import { formatDate } from './AccessControlDetailSupport'
import styles from './AccessControlPage.module.css'

export function DashboardMemberDetail({
  memberId,
  canManage,
  onChanged,
  onClose,
}: {
  memberId: string
  canManage: boolean
  onChanged: () => void
  onClose: () => void
}) {
  const [member, setMember] = useState<DashboardMember | null>(null)
  const [error, setError] = useState('')
  const [editing, setEditing] = useState(false)
  const dialogRef = useAccessibleDialog<HTMLDivElement>({ isOpen: true, onClose })

  useEffect(() => {
    setError('')
    void fetch(`/api/admin/users/${encodeURIComponent(memberId)}`)
      .then(async (response) => {
        if (!response.ok)
          throw new Error((await response.text()) || 'Could not load Dashboard identity')
        return response.json() as Promise<DashboardMember>
      })
      .then(setMember)
      .catch((nextError) =>
        setError(
          nextError instanceof Error ? nextError.message : 'Could not load Dashboard identity',
        ),
      )
  }, [memberId])

  return (
    <div
      className={styles.detailBackdrop}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <aside
        ref={dialogRef}
        className={styles.detailDrawer}
        role="dialog"
        aria-modal="true"
        aria-labelledby="member-detail-title"
        tabIndex={-1}
      >
        <header className={styles.detailHeader}>
          <div>
            <span>Dashboard identity</span>
            <h2 id="member-detail-title">{member?.name || 'User details'}</h2>
            <p>{member?.email || 'Loading identity…'}</p>
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
          {!member && !error ? (
            <div className={styles.detailLoading}>Loading user details…</div>
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
                  <span>Model identity</span>
                  <strong>{member.inferenceConsumerId ? 'Active' : 'Provisioning'}</strong>
                </article>
              </div>
              <section className={styles.detailSection}>
                <div className={styles.detailSectionHeading}>
                  <span>Authorization</span>
                  <h3>Effective permissions</h3>
                </div>
                <PermissionList permissions={member.permissions || []} />
              </section>
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
                    <dd>Managed with this Dashboard identity</dd>
                  </div>
                </dl>
              </section>
            </>
          ) : null}
        </div>
        <footer className={styles.detailFooter}>
          {member && canManage ? (
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => setEditing(true)}
            >
              Manage login
            </button>
          ) : null}
          <button type="button" className={styles.primaryButton} onClick={onClose}>
            Done
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
