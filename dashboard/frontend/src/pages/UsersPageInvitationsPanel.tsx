import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link2, Mail, RefreshCw, UserPlus, X } from 'lucide-react'

import ConfirmDialog from '../components/ConfirmDialog'
import {
  Alert,
  Badge,
  Button,
  CopyField,
  DataTable,
  DataToolbar,
  Dialog,
  DialogBody,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DropdownMenuItem,
  DropdownMenuSeparator,
  EmptyState,
  FilterPill,
  Label,
  Progress,
  StatusLabel,
  toast,
  type Column,
  type StatusTone,
} from '../ui'
import type { DashboardInvitation, InvitationStatus } from './dashboardInvitationTypes'
import {
  ROLE_META,
  absoluteTime,
  invitationUrl,
  relativeTime,
  responseText,
} from './usersPageSupport'

interface UsersPageInvitationsPanelProps {
  onInvite: () => void
  refreshToken: number
  onPendingChange?: (pending: number) => void
}

const STATUS_META: Record<InvitationStatus, { label: string; tone: StatusTone }> = {
  pending: { label: 'Pending', tone: 'info' },
  accepted: { label: 'Complete', tone: 'success' },
  expired: { label: 'Expired', tone: 'warning' },
  revoked: { label: 'Revoked', tone: 'neutral' },
}

export function InvitationProgress({ invitation }: { invitation: DashboardInvitation }) {
  if (invitation.kind === 'personal') {
    const joined = invitation.usedCount > 0 || invitation.status === 'accepted'
    return (
      <div className="flex flex-col">
        <span className="text-[13px]">{joined ? 'Accepted and signed in' : 'Waiting to join'}</span>
        <span className="text-xs text-muted-foreground">
          {joined ? absoluteTime(invitation.acceptedAt) : '1 ticket available'}
        </span>
      </div>
    )
  }
  const percentage = Math.min(
    100,
    Math.round((invitation.usedCount / Math.max(1, invitation.maxUses)) * 100),
  )
  return (
    <div className="flex min-w-36 flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3 text-xs">
        <span className="font-medium tabular text-foreground/90">
          {invitation.usedCount} / {invitation.maxUses} used
        </span>
        <span className="tabular text-muted-foreground">
          {invitation.remainingUses} tickets left
        </span>
      </div>
      <Progress value={percentage} aria-label={`${percentage}% of tickets used`} />
    </div>
  )
}

export default function UsersPageInvitationsPanel({
  onInvite,
  refreshToken,
  onPendingChange,
}: UsersPageInvitationsPanelProps) {
  const [invitations, setInvitations] = useState<DashboardInvitation[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statuses, setStatuses] = useState<string[]>([])
  const [revoking, setRevoking] = useState<DashboardInvitation | null>(null)
  const [revokePending, setRevokePending] = useState(false)
  const [rotated, setRotated] = useState<{ invitation: DashboardInvitation; url: string } | null>(
    null,
  )

  const loadInvitations = useCallback(async (signal?: AbortSignal) => {
    setLoading(true)
    try {
      const response = await fetch('/api/admin/invitations', { signal })
      if (!response.ok) throw new Error(await responseText(response))
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

  const pending = invitations.filter((invitation) => invitation.status === 'pending').length
  useEffect(() => onPendingChange?.(pending), [onPendingChange, pending])

  const rotate = async (invitation: DashboardInvitation) => {
    try {
      const response = await fetch(
        `/api/admin/invitations/${encodeURIComponent(invitation.id)}/rotate`,
        { method: 'POST' },
      )
      if (!response.ok) throw new Error(await responseText(response))
      const payload = (await response.json()) as { invitation: DashboardInvitation; token: string }
      setRotated({ invitation: payload.invitation, url: invitationUrl(payload.token) })
      void loadInvitations()
    } catch (cause) {
      toast.error('No new link was made', {
        description: cause instanceof Error ? cause.message : String(cause),
      })
    }
  }

  const revoke = async () => {
    if (!revoking) return
    setRevokePending(true)
    try {
      const response = await fetch(`/api/admin/invitations/${encodeURIComponent(revoking.id)}`, {
        method: 'DELETE',
      })
      if (!response.ok && response.status !== 204) throw new Error(await responseText(response))
      toast(
        `Invitation for ${revoking.kind === 'shared' ? 'the shared link' : revoking.email} revoked`,
      )
      setRevoking(null)
      void loadInvitations()
    } catch (cause) {
      toast.error('The invitation was not revoked', {
        description: cause instanceof Error ? cause.message : String(cause),
      })
    } finally {
      setRevokePending(false)
    }
  }

  const visible = invitations.filter(
    (invitation) => statuses.length === 0 || statuses.includes(invitation.status),
  )

  const columns = useMemo<Column<DashboardInvitation>[]>(
    () => [
      {
        id: 'identity',
        header: 'Invitation',
        sortValue: (row) => (row.kind === 'shared' ? 'shared' : (row.email ?? '')),
        cell: (row) => (
          <div className="flex min-w-0 items-center gap-3">
            <span className="grid size-[30px] shrink-0 place-items-center rounded-full border border-dashed border-border-strong text-subtle-foreground">
              {row.kind === 'shared' ? (
                <Link2 className="size-3.5" />
              ) : (
                <Mail className="size-3.5" />
              )}
            </span>
            <span className="flex min-w-0 flex-col">
              <span className="truncate text-[13px] font-medium">
                {row.kind === 'shared' ? 'Shared invitation' : row.name}
              </span>
              <span className="truncate text-xs text-subtle-foreground">
                {row.kind === 'shared' ? 'Self-registration link' : row.email}
              </span>
            </span>
          </div>
        ),
      },
      {
        id: 'role',
        header: 'Role',
        cell: (row) => (
          <Badge tone={ROLE_META[row.role]?.tone ?? 'neutral'}>
            {ROLE_META[row.role]?.label ?? row.role}
          </Badge>
        ),
      },
      {
        id: 'progress',
        header: 'Progress',
        hideBelow: 'lg',
        cell: (row) => <InvitationProgress invitation={row} />,
      },
      {
        id: 'status',
        header: 'Status',
        sortValue: (row) => row.status,
        cell: (row) => (
          <StatusLabel tone={STATUS_META[row.status].tone}>
            {STATUS_META[row.status].label}
          </StatusLabel>
        ),
      },
      {
        id: 'expires',
        header: 'Expires',
        sortValue: (row) => row.expiresAt,
        hideBelow: 'md',
        cell: (row) =>
          row.status === 'pending' ? (
            <span className="text-xs text-muted-foreground">{absoluteTime(row.expiresAt)}</span>
          ) : row.status === 'expired' ? (
            <span className="text-xs text-warning-foreground" title={absoluteTime(row.expiresAt)}>
              Expired {relativeTime(row.expiresAt).toLowerCase()}
            </span>
          ) : (
            <span className="text-xs text-subtle-foreground">—</span>
          ),
      },
    ],
    [],
  )

  return (
    <div className="flex flex-col gap-3">
      {error ? (
        <Alert
          tone="danger"
          title="Invitations could not be loaded"
          actions={
            <Button size="sm" variant="outline" onClick={() => void loadInvitations()}>
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      ) : null}
      <div>
        <DataToolbar
          filters={
            <FilterPill
              label="Status"
              selected={statuses}
              onChange={setStatuses}
              options={(Object.keys(STATUS_META) as InvitationStatus[]).map((status) => ({
                value: status,
                label: STATUS_META[status].label,
                count: invitations.filter((invitation) => invitation.status === status).length,
              }))}
            />
          }
          actions={
            <>
              <span className="text-xs text-muted-foreground">
                <span className="tabular">{pending}</span> pending ·{' '}
                <span className="tabular">{invitations.length}</span> total
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Refresh invitations"
                onClick={() => void loadInvitations()}
                disabled={loading}
              >
                <RefreshCw className={loading ? 'animate-spin' : undefined} />
              </Button>
            </>
          }
        />
        <DataTable
          data={visible}
          columns={columns}
          getRowId={(row) => row.id}
          loading={loading && invitations.length === 0}
          rowActions={(row) => (
            <>
              <DropdownMenuItem
                disabled={row.status !== 'pending'}
                onSelect={() => void rotate(row)}
              >
                <Link2 />
                New link
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant="destructive"
                disabled={row.status !== 'pending'}
                onSelect={() => setRevoking(row)}
              >
                <X />
                Revoke
              </DropdownMenuItem>
            </>
          )}
          empty={
            <EmptyState
              compact
              icon={Mail}
              tone="slate"
              title={invitations.length ? 'No invitations match' : 'No invitations yet'}
              description={
                invitations.length
                  ? 'Clear the status filter to see every invitation.'
                  : 'Invite teammates and they show up here until they join.'
              }
              actions={
                invitations.length ? (
                  <Button variant="outline" size="sm" onClick={() => setStatuses([])}>
                    Clear filter
                  </Button>
                ) : (
                  <Button size="sm" onClick={onInvite}>
                    <UserPlus />
                    Invite user
                  </Button>
                )
              }
            />
          }
        />
      </div>
      <p className="text-xs text-subtle-foreground">
        Links are shown once, when they are made. New link replaces the old one and starts a fresh
        expiry.
      </p>

      <ConfirmDialog
        isOpen={revoking !== null}
        title={
          revoking?.kind === 'shared'
            ? 'Revoke the shared link?'
            : `Revoke the invitation for ${revoking?.email ?? ''}?`
        }
        description="The link stops working right away. People who already joined keep their accounts."
        confirmLabel="Revoke invitation"
        tone="danger"
        pending={revokePending}
        onCancel={() => setRevoking(null)}
        onConfirm={revoke}
      />

      <Dialog open={rotated !== null} onOpenChange={(open) => !open && setRotated(null)}>
        <DialogContent size="md">
          <DialogHeader>
            <DialogTitle>New invitation link</DialogTitle>
            <DialogDescription>
              The old link no longer works. This one expires{' '}
              {rotated ? absoluteTime(rotated.invitation.expiresAt) : ''}.
            </DialogDescription>
          </DialogHeader>
          <DialogBody className="flex flex-col gap-1.5 pb-5">
            <Label>
              {rotated?.invitation.kind === 'shared'
                ? 'Shared invitation link'
                : `Link for ${rotated?.invitation.email ?? ''}`}
            </Label>
            {rotated ? <CopyField value={rotated.url} /> : null}
          </DialogBody>
          <DialogFooter>
            <Button onClick={() => setRotated(null)}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
