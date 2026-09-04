import { useCallback, useEffect, useMemo, useState } from 'react'

import { DataTable, type Column } from '../components/DataTable'
import ProductIcon from '../components/ProductIcon'
import ProductLoadingState from '../components/ProductLoadingState'

import type { DashboardInvitation } from './dashboardInvitationTypes'
import styles from './UsersPage.module.css'

interface UsersPageInvitationsPanelProps {
  onInvite: () => void
  refreshToken: number
}

const formatTimestamp = (value?: number) => {
  if (!value) return '-'
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value * 1000))
}

const responseError = async (response: Response) =>
  (await response.text()) || `Request failed (${response.status})`

export default function UsersPageInvitationsPanel({
  onInvite,
  refreshToken,
}: UsersPageInvitationsPanelProps) {
  const [invitations, setInvitations] = useState<DashboardInvitation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadInvitations = useCallback(async (signal?: AbortSignal) => {
    setLoading(true)
    try {
      const response = await fetch('/api/admin/invitations', { signal })
      if (!response.ok) throw new Error(await responseError(response))
      const payload = (await response.json()) as { invitations?: DashboardInvitation[] }
      setInvitations(payload.invitations ?? [])
      setError(null)
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') return
      setError(cause instanceof Error ? cause.message : 'Could not load invitations.')
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void loadInvitations(controller.signal)
    return () => controller.abort()
  }, [loadInvitations, refreshToken])

  const columns = useMemo<Column<DashboardInvitation>[]>(
    () => [
      {
        key: 'identity',
        header: 'Invitation',
        width: '300px',
        render: (invitation) => (
          <div className={styles.invitationIdentity}>
            <strong>{invitation.kind === 'shared' ? 'Shared invitation' : invitation.name}</strong>
            <span>
              {invitation.kind === 'shared' ? 'Self-registration link' : invitation.email}
            </span>
          </div>
        ),
      },
      {
        key: 'kind',
        header: 'Type',
        width: '130px',
        render: (invitation) => (invitation.kind === 'shared' ? 'Shared link' : 'One person'),
      },
      {
        key: 'role',
        header: 'Role',
        width: '120px',
        render: (invitation) => (
          <span className={styles.invitationRole}>
            {invitation.role === 'write'
              ? 'Builder'
              : invitation.role.charAt(0).toUpperCase() + invitation.role.slice(1)}
          </span>
        ),
      },
      {
        key: 'progress',
        header: 'Progress',
        width: '260px',
        render: (invitation) => <InvitationProgress invitation={invitation} />,
      },
      {
        key: 'status',
        header: 'Status',
        width: '130px',
        render: (invitation) => (
          <span
            className={`${styles.invitationStatus} ${styles[`invitationStatus${capitalize(invitation.status)}`]}`}
          >
            {invitation.status === 'accepted' ? 'Complete' : capitalize(invitation.status)}
          </span>
        ),
      },
      {
        key: 'expiresAt',
        header: 'Expires',
        width: '190px',
        render: (invitation) => formatTimestamp(invitation.expiresAt),
      },
    ],
    [],
  )

  return (
    <section className={styles.card}>
      <div className={styles.sectionHeader}>
        <div>
          <h2 className={styles.sectionTitle}>Invitations</h2>
          <p className={styles.sectionDescription}>
            See who joined and how many shared tickets remain.
          </p>
        </div>
        <div className={styles.sectionActions}>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() => void loadInvitations()}
            disabled={loading}
          >
            <ProductIcon name="refresh" aria-hidden="true" />
            Refresh
          </button>
          <button type="button" className={styles.primaryButton} onClick={onInvite}>
            <ProductIcon name="plus" aria-hidden="true" />
            Invite user
          </button>
        </div>
      </div>

      {error ? (
        <div className={styles.auditError} role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => void loadInvitations()}>
            Retry
          </button>
        </div>
      ) : null}

      {loading && invitations.length === 0 ? (
        <ProductLoadingState label="Loading invitations" compact />
      ) : (
        <DataTable
          columns={columns}
          data={invitations}
          keyExtractor={(invitation) => invitation.id}
          className={styles.invitationTable}
          emptyMessage="No invitations yet."
        />
      )}
    </section>
  )
}

export function InvitationProgress({ invitation }: { invitation: DashboardInvitation }) {
  if (invitation.kind === 'personal') {
    const joined = invitation.usedCount > 0 || invitation.status === 'accepted'
    return (
      <div className={styles.invitationProgressCopy}>
        <strong>{joined ? 'Accepted and signed in' : 'Waiting to join'}</strong>
        <span>{joined ? formatTimestamp(invitation.acceptedAt) : '1 ticket available'}</span>
      </div>
    )
  }

  const percentage = Math.min(100, Math.round((invitation.usedCount / invitation.maxUses) * 100))
  return (
    <div className={styles.invitationProgress}>
      <div>
        <strong>
          {invitation.usedCount} / {invitation.maxUses} used
        </strong>
        <span>{invitation.remainingUses} tickets left</span>
      </div>
      <span className={styles.invitationProgressTrack} aria-hidden="true">
        <span style={{ width: `${percentage}%` }} />
      </span>
    </div>
  )
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1)
}
