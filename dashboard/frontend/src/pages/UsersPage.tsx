import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  ArrowDownWideNarrow,
  ArrowUpNarrowWide,
  Ban,
  Check,
  Lock,
  Minus,
  RefreshCw,
  ShieldCheck,
  Trash2,
  UserCheck,
  UserCog,
  UserPlus,
  UserRound,
  UsersRound,
} from 'lucide-react'

import { useAuth } from '../contexts/AuthContext'
import ConfirmDialog from '../components/ConfirmDialog'
import {
  Alert,
  Avatar,
  Badge,
  BulkBar,
  Button,
  DataTable,
  DataToolbar,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
  EmptyState,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Segmented,
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
  StatusLabel,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Tooltip,
  cn,
  toast,
  type Column,
} from '../ui'
import {
  canManageUsers as canManageDashboardUsers,
  canViewUsers as canViewDashboardUsers,
} from '../utils/accessControl'
import DashboardInviteDialog from './DashboardInviteDialog'
import type { DashboardInvitation } from './dashboardInvitationTypes'
import { SettingsSection } from './SettingsLayout'
import UsersPageAuditPanel, { type AuditActor } from './UsersPageAuditPanel'
import UsersPageInvitationsPanel from './UsersPageInvitationsPanel'
import UsersPageMemberSheet, { type MemberDraft } from './UsersPageMemberSheet'
import {
  DASHBOARD_ROLES,
  EMPTY_ROLE_PERMISSIONS,
  ROLE_META,
  absoluteTime,
  createLatestUsersRequest,
  describeUserUpdateFailure,
  isUsersRequestAbortError,
  relativeTime,
  responseText,
  roleLabel,
  validateUserPassword,
  type AdminUser,
  type UsersPagePayload,
  type UsersPageRolePermissions,
  type UsersPageRolePermissionsPayload,
} from './usersPageSupport'

type UsersView = 'users' | 'invitations' | 'audit'

const PAGE_SIZE_OPTIONS = [10, 20, 50] as const
const SORT_OPTIONS = [
  { value: 'createdAt', label: 'Joined' },
  { value: 'email', label: 'Email' },
  { value: 'name', label: 'Name' },
  { value: 'role', label: 'Role' },
  { value: 'lastLoginAt', label: 'Last sign-in' },
] as const

const JSON_HEADERS = { 'Content-Type': 'application/json' }

function RoleBadge({ role }: { role: string }) {
  return <Badge tone={ROLE_META[role]?.tone ?? 'neutral'}>{roleLabel(role)}</Badge>
}

function RolesPopover({
  rolePermissions,
  loading,
}: {
  rolePermissions: UsersPageRolePermissions
  loading: boolean
}) {
  const permissions = [...new Set(Object.values(rolePermissions).flat())].sort()
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline">
          <ShieldCheck />
          Roles
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[min(480px,calc(100vw-2rem))] p-0">
        <div className="border-b border-border px-4 py-3">
          <p className="text-[13px] font-semibold">Roles and permissions</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Every person has one role. Admins can change it at any time.
          </p>
        </div>
        {loading ? (
          <p className="px-4 py-6 text-center text-xs text-muted-foreground">
            Loading permissions…
          </p>
        ) : permissions.length ? (
          <div className="max-h-[min(60vh,420px)] overflow-y-auto">
            <table className="w-full text-[12.5px]">
              <thead className="sticky top-0 bg-surface-subtle">
                <tr className="border-b border-border">
                  <th className="px-4 py-2 text-left text-xs font-medium text-muted-foreground">
                    Permission
                  </th>
                  {DASHBOARD_ROLES.map((role) => (
                    <th key={role} className="w-20 px-1 py-2 text-center">
                      <RoleBadge role={role} />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {permissions.map((permission) => (
                  <tr key={permission} className="border-b border-border last:border-0">
                    <td className="px-4 py-2">
                      <code className="font-mono text-[12px]">{permission}</code>
                    </td>
                    {DASHBOARD_ROLES.map((role) => (
                      <td key={role} className="text-center">
                        {(rolePermissions[role] ?? []).includes(permission) ? (
                          <Check
                            className="mx-auto size-4 text-success"
                            strokeWidth={2.4}
                            aria-label="Allowed"
                          />
                        ) : (
                          <Minus
                            className="mx-auto size-3.5 text-subtle-foreground/60"
                            aria-label="Not allowed"
                          />
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="px-4 py-6 text-center text-xs text-muted-foreground">
            Permissions are not available.
          </p>
        )}
      </PopoverContent>
    </Popover>
  )
}

const UsersPage: React.FC = () => {
  const { user: currentUser } = useAuth()
  const canManageUsers = canManageDashboardUsers(currentUser)
  const canViewUsers = canViewDashboardUsers(currentUser)
  const [params, setParams] = useSearchParams()

  const [users, setUsers] = useState<AdminUser[]>([])
  const [totalUsers, setTotalUsers] = useState(0)
  const [activeUserCount, setActiveUserCount] = useState(0)
  const [privilegedUserCount, setPrivilegedUserCount] = useState(0)
  const [loadingUsers, setLoadingUsers] = useState(true)
  const [usersError, setUsersError] = useState<string | null>(null)
  const [rolePermissions, setRolePermissions] =
    useState<UsersPageRolePermissions>(EMPTY_ROLE_PERMISSIONS)
  const [loadingRolePermissions, setLoadingRolePermissions] = useState(false)

  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'active' | 'inactive'>('all')
  const [sortField, setSortField] = useState('createdAt')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState<number>(10)

  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [inviteOpen, setInviteOpen] = useState(false)
  const inviteButtonRef = useRef<HTMLButtonElement>(null)
  const [invitationRefreshToken, setInvitationRefreshToken] = useState(0)
  const [pendingInvites, setPendingInvites] = useState(0)
  const [sheetSubmitting, setSheetSubmitting] = useState(false)
  const [sheetError, setSheetError] = useState<string | null>(null)
  const [deleteTargets, setDeleteTargets] = useState<AdminUser[] | null>(null)
  const [deletePending, setDeletePending] = useState(false)
  const [actors, setActors] = useState<Record<string, AuditActor>>({})
  const [userRequests] = useState(createLatestUsersRequest)

  const requestedView = params.get('tab')
  const activeView: UsersView =
    canManageUsers && (requestedView === 'invitations' || requestedView === 'audit')
      ? requestedView
      : 'users'
  const activeUserId = params.get('user')
  const activeUser = users.find((user) => user.id === activeUserId) ?? null

  const fetchUsers = useCallback(async () => {
    const request = userRequests.start()
    setLoadingUsers(true)
    try {
      const q = new URLSearchParams()
      if (statusFilter !== 'all') q.set('status', statusFilter)
      if (debouncedQuery) q.set('q', debouncedQuery)
      q.set('page', String(page))
      q.set('limit', String(pageSize))
      q.set('sort', sortField)
      q.set('order', sortOrder)
      const response = await fetch(`/api/admin/users?${q.toString()}`, {
        method: 'GET',
        headers: JSON_HEADERS,
        signal: request.signal,
      })
      if (!response.ok) throw new Error(await responseText(response))
      const payload = (await response.json()) as UsersPagePayload
      if (!request.isCurrent()) return
      const nextUsers = payload.users || []
      const nextTotal = payload.total ?? nextUsers.length
      const lastPage = Math.max(1, Math.ceil(nextTotal / pageSize))
      if (page > lastPage) {
        setPage(lastPage)
        return
      }
      setUsers(nextUsers)
      setTotalUsers(nextTotal)
      setActiveUserCount(
        payload.active ?? nextUsers.filter((user) => user.status === 'active').length,
      )
      setPrivilegedUserCount(
        payload.privileged ?? nextUsers.filter((user) => user.role === 'admin').length,
      )
      setActors((current) => ({
        ...current,
        ...Object.fromEntries(
          nextUsers.map((user) => [user.id, { name: user.name, email: user.email }]),
        ),
      }))
      setUsersError(null)
    } catch (err) {
      if (!request.isCurrent() || isUsersRequestAbortError(err)) return
      setUsersError((err as Error).message)
    } finally {
      if (request.isCurrent()) setLoadingUsers(false)
      request.finish()
    }
  }, [debouncedQuery, page, pageSize, sortField, sortOrder, statusFilter, userRequests])

  const fetchRolePermissions = useCallback(async () => {
    if (!canManageUsers) {
      setRolePermissions(EMPTY_ROLE_PERMISSIONS)
      setLoadingRolePermissions(false)
      return
    }
    setLoadingRolePermissions(true)
    try {
      const response = await fetch('/api/admin/permissions', {
        method: 'GET',
        headers: JSON_HEADERS,
      })
      if (!response.ok) throw new Error(await responseText(response))
      const payload = (await response.json()) as UsersPageRolePermissionsPayload
      setRolePermissions(payload.rolePermissions ?? EMPTY_ROLE_PERMISSIONS)
    } catch (err) {
      setRolePermissions(EMPTY_ROLE_PERMISSIONS)
      toast.error('Role permissions could not be loaded', { description: (err as Error).message })
    } finally {
      setLoadingRolePermissions(false)
    }
  }, [canManageUsers])

  useEffect(() => {
    if (!canViewUsers) {
      userRequests.abort()
      setLoadingUsers(false)
      return
    }
    void fetchUsers()
    return () => userRequests.abort()
  }, [canViewUsers, fetchUsers, userRequests])

  useEffect(() => {
    void fetchRolePermissions()
  }, [fetchRolePermissions])

  useEffect(() => {
    if (!canManageUsers) return
    const controller = new AbortController()
    void fetch('/api/admin/invitations', { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : null))
      .then((payload: { invitations?: DashboardInvitation[] } | null) => {
        if (payload)
          setPendingInvites(
            (payload.invitations ?? []).filter((item) => item.status === 'pending').length,
          )
      })
      .catch(() => undefined)
    return () => controller.abort()
  }, [canManageUsers, invitationRefreshToken])

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query.trim()), 300)
    return () => window.clearTimeout(timer)
  }, [query])

  useEffect(() => setSelected(new Set()), [users])

  const updateParams = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params)
    for (const [key, value] of Object.entries(patch)) {
      if (value === null) next.delete(key)
      else next.set(key, value)
    }
    setParams(next)
  }

  const isSelf = (user: AdminUser) => user.id === currentUser?.id

  const patchUser = async (user: AdminUser, patch: Partial<Pick<AdminUser, 'role' | 'status'>>) => {
    const response = await fetch(`/api/admin/users/${user.id}`, {
      method: 'PATCH',
      headers: JSON_HEADERS,
      body: JSON.stringify({ role: patch.role ?? user.role, status: patch.status ?? user.status }),
    })
    if (!response.ok) throw new Error(await responseText(response))
  }

  const changeUsers = async (
    targets: AdminUser[],
    patch: Partial<Pick<AdminUser, 'role' | 'status'>>,
    outcome: string,
  ) => {
    const eligible = targets.filter((user) => !isSelf(user))
    if (eligible.length === 0) return
    const failures: string[] = []
    for (const user of eligible) {
      try {
        await patchUser(user, patch)
      } catch (err) {
        failures.push(`${user.email}: ${(err as Error).message}`)
      }
    }
    const done = eligible.length - failures.length
    if (done) {
      const who =
        done === 1
          ? eligible.find((user) => !failures.some((line) => line.startsWith(user.email)))?.name ||
            eligible[0].email
          : `${done} people`
      toast.success(`${who}: ${outcome}`, { description: 'Takes effect on their next request.' })
    }
    if (failures.length)
      toast.error(`${failures.length} ${failures.length === 1 ? 'change' : 'changes'} failed`, {
        description: failures.join('\n'),
      })
    setSelected(new Set())
    await fetchUsers()
  }

  const handleSheetSubmit = async (values: MemberDraft) => {
    if (!activeUser || !canManageUsers) return
    // Reject before sending: a server-side rejection arrives after the role change has already committed.
    const passwordError = validateUserPassword(values.password, 'edit')
    if (passwordError) {
      setSheetError(passwordError)
      return
    }
    setSheetSubmitting(true)
    setSheetError(null)
    let roleAndStatusSaved = false
    try {
      if (
        values.role !== activeUser.role ||
        values.status !== activeUser.status ||
        !values.password
      ) {
        await patchUser(activeUser, { role: values.role, status: values.status })
      }
      // Role and status are committed on the server from here.
      roleAndStatusSaved = true
      if (values.password.trim()) {
        const passwordResponse = await fetch('/api/admin/users/password', {
          method: 'POST',
          headers: JSON_HEADERS,
          body: JSON.stringify({ userId: activeUser.id, password: values.password }),
        })
        if (!passwordResponse.ok) throw new Error(await responseText(passwordResponse))
      }
      toast.success(values.password.trim() ? 'User updated and password rotated' : 'User updated', {
        description: `${activeUser.name || activeUser.email} · takes effect on their next request.`,
      })
      await fetchUsers()
    } catch (err) {
      setSheetError(describeUserUpdateFailure(err, roleAndStatusSaved))
      // A failed request does not prove nothing changed.
      void fetchUsers()
    } finally {
      setSheetSubmitting(false)
    }
  }

  const confirmDelete = async () => {
    if (!deleteTargets || !canManageUsers) return
    setDeletePending(true)
    const failures: string[] = []
    for (const user of deleteTargets) {
      try {
        const response = await fetch(`/api/admin/users/${user.id}`, {
          method: 'DELETE',
          headers: JSON_HEADERS,
        })
        if (!response.ok && response.status !== 204) throw new Error(await responseText(response))
      } catch (err) {
        failures.push(`${user.email}: ${(err as Error).message}`)
      }
    }
    setDeletePending(false)
    const done = deleteTargets.length - failures.length
    if (done)
      toast(
        done === 1
          ? `${deleteTargets[0].name || deleteTargets[0].email} deleted`
          : `${done} users deleted`,
      )
    if (failures.length)
      toast.error('Some users were not deleted', { description: failures.join('\n') })
    if (activeUserId && deleteTargets.some((user) => user.id === activeUserId))
      updateParams({ user: null })
    setDeleteTargets(null)
    setSelected(new Set())
    await fetchUsers()
  }

  const totalPages = Math.max(1, Math.ceil(totalUsers / pageSize))
  const currentPage = Math.min(page, totalPages)
  const selectedUsers = users.filter((user) => selected.has(user.id) && !isSelf(user))

  const columns = useMemo<Column<AdminUser>[]>(
    () => [
      {
        id: 'name',
        header: 'Person',
        cell: (row) => (
          <div className="flex min-w-0 items-center gap-3">
            <Avatar
              name={row.name || row.email}
              size={30}
              className={cn(row.status !== 'active' && 'opacity-50 grayscale')}
            />
            <div className="flex min-w-0 flex-col">
              <span className="flex items-center gap-1.5">
                <span
                  className={cn(
                    'truncate font-medium',
                    row.status !== 'active' && 'text-muted-foreground',
                  )}
                >
                  {row.name || row.email}
                </span>
                {row.id === currentUser?.id ? (
                  <Badge tone="outline" size="sm">
                    You
                  </Badge>
                ) : null}
              </span>
              <span className="truncate text-xs text-subtle-foreground">{row.email}</span>
            </div>
          </div>
        ),
      },
      { id: 'role', header: 'Role', cell: (row) => <RoleBadge role={row.role} /> },
      {
        id: 'status',
        header: 'Status',
        cell: (row) => (
          <StatusLabel tone={row.status === 'active' ? 'success' : 'neutral'}>
            {row.status === 'active' ? 'Active' : 'Inactive'}
          </StatusLabel>
        ),
      },
      {
        id: 'lastLogin',
        header: 'Last sign-in',
        hideBelow: 'md',
        cell: (row) => (
          <span className="text-xs text-muted-foreground" title={absoluteTime(row.lastLoginAt)}>
            {relativeTime(row.lastLoginAt)}
          </span>
        ),
      },
      {
        id: 'joined',
        header: 'Joined',
        hideBelow: 'xl',
        cell: (row) => (
          <span className="text-xs text-muted-foreground">{absoluteTime(row.createdAt)}</span>
        ),
      },
    ],
    [currentUser?.id],
  )

  if (!canViewUsers) {
    return (
      <SettingsSection
        title="Users & access"
        description="People who can sign in to this dashboard and what their role allows."
      >
        <EmptyState
          icon={Lock}
          tone="slate"
          title="Access required"
          description="You do not have permission to view dashboard user management. Ask an admin for the users.view permission."
        />
      </SettingsSection>
    )
  }

  const directory = (
    <div className="flex flex-col gap-3">
      {usersError ? (
        <Alert
          tone="danger"
          title="Users could not be loaded"
          actions={
            <Button size="sm" variant="outline" onClick={() => void fetchUsers()}>
              Retry
            </Button>
          }
        >
          {usersError}
        </Alert>
      ) : null}
      <div>
        <DataToolbar
          search={query}
          onSearchChange={(value) => {
            setQuery(value)
            setPage(1)
          }}
          placeholder="Search by name, email or role…"
          filters={
            <>
              <Segmented
                size="sm"
                aria-label="Status"
                value={statusFilter}
                onValueChange={(value) => {
                  setStatusFilter(value)
                  setPage(1)
                }}
                options={[
                  { value: 'all', label: 'All' },
                  { value: 'active', label: 'Active' },
                  { value: 'inactive', label: 'Inactive' },
                ]}
              />
              <span className="flex items-center gap-1">
                <Select
                  value={sortField}
                  onValueChange={(value) => {
                    setSortField(value)
                    setPage(1)
                  }}
                >
                  <SelectTrigger size="sm" aria-label="Sort by" className="min-w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SORT_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Tooltip content={sortOrder === 'desc' ? 'Descending' : 'Ascending'}>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    aria-label={sortOrder === 'desc' ? 'Sort ascending' : 'Sort descending'}
                    onClick={() => {
                      setSortOrder((current) => (current === 'desc' ? 'asc' : 'desc'))
                      setPage(1)
                    }}
                  >
                    {sortOrder === 'desc' ? <ArrowDownWideNarrow /> : <ArrowUpNarrowWide />}
                  </Button>
                </Tooltip>
              </span>
            </>
          }
          actions={
            <>
              <span className="hidden text-xs text-muted-foreground lg:inline">
                <span className="tabular">{totalUsers}</span> people ·{' '}
                <span className="tabular">{activeUserCount}</span> active ·{' '}
                <span className="tabular">{privilegedUserCount}</span>{' '}
                {privilegedUserCount === 1 ? 'admin' : 'admins'}
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Refresh users"
                onClick={() => void fetchUsers()}
                disabled={loadingUsers}
              >
                <RefreshCw className={cn(loadingUsers && 'animate-spin')} />
              </Button>
            </>
          }
        />
        <DataTable
          data={users}
          columns={columns}
          getRowId={(row) => row.id}
          loading={loadingUsers && users.length === 0}
          selectable={canManageUsers}
          selected={selected}
          onSelectedChange={setSelected}
          activeRowId={activeUserId ?? undefined}
          onRowClick={(row) => {
            setSheetError(null)
            updateParams({ user: row.id })
          }}
          rowActions={(row) => (
            <>
              <DropdownMenuItem onSelect={() => updateParams({ user: row.id })}>
                <UserRound />
                View profile
              </DropdownMenuItem>
              {canManageUsers ? (
                <>
                  <DropdownMenuSub>
                    <DropdownMenuSubTrigger disabled={isSelf(row)}>
                      <UserCog />
                      Change role
                    </DropdownMenuSubTrigger>
                    <DropdownMenuSubContent className="w-40">
                      <DropdownMenuRadioGroup
                        value={row.role}
                        onValueChange={(role) =>
                          void changeUsers([row], { role }, `role changed to ${roleLabel(role)}`)
                        }
                      >
                        {DASHBOARD_ROLES.map((role) => (
                          <DropdownMenuRadioItem key={role} value={role}>
                            {ROLE_META[role].label}
                          </DropdownMenuRadioItem>
                        ))}
                      </DropdownMenuRadioGroup>
                    </DropdownMenuSubContent>
                  </DropdownMenuSub>
                  <DropdownMenuItem
                    disabled={isSelf(row)}
                    onSelect={() =>
                      void changeUsers(
                        [row],
                        { status: row.status === 'active' ? 'inactive' : 'active' },
                        row.status === 'active' ? 'deactivated' : 'activated',
                      )
                    }
                  >
                    {row.status === 'active' ? <Ban /> : <UserCheck />}
                    {row.status === 'active' ? 'Deactivate' : 'Activate'}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    variant="destructive"
                    disabled={isSelf(row)}
                    onSelect={() => setDeleteTargets([row])}
                  >
                    <Trash2 />
                    Delete user
                  </DropdownMenuItem>
                </>
              ) : null}
            </>
          )}
          empty={
            <EmptyState
              compact
              icon={UsersRound}
              tone="slate"
              title="No users found"
              description={
                debouncedQuery || statusFilter !== 'all'
                  ? 'No users match the current filters. Try another name, or clear the filters.'
                  : 'Invite teammates to give them dashboard access.'
              }
              actions={
                debouncedQuery || statusFilter !== 'all' ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setQuery('')
                      setStatusFilter('all')
                    }}
                  >
                    Clear filters
                  </Button>
                ) : canManageUsers ? (
                  <Button size="sm" onClick={() => setInviteOpen(true)}>
                    <UserPlus />
                    Invite user
                  </Button>
                ) : null
              }
            />
          }
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
        <span className="flex items-center gap-2">
          Page {currentPage} / {totalPages} · {totalUsers} users
          <Select
            value={String(pageSize)}
            onValueChange={(value) => {
              setPageSize(Number.parseInt(value, 10))
              setPage(1)
            }}
          >
            <SelectTrigger size="sm" aria-label="Rows per page" className="h-7 w-[4.5rem] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PAGE_SIZE_OPTIONS.map((size) => (
                <SelectItem key={size} value={String(size)}>
                  {size}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </span>
        <span className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="xs"
            onClick={() => setPage(Math.max(1, currentPage - 1))}
            disabled={currentPage <= 1}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="xs"
            onClick={() => setPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage >= totalPages}
          >
            Next
          </Button>
        </span>
      </div>
    </div>
  )

  return (
    <SettingsSection
      title="Users & access"
      description="People who can sign in to this dashboard, what their role allows, and what they changed."
      actions={
        canManageUsers ? (
          <>
            <RolesPopover rolePermissions={rolePermissions} loading={loadingRolePermissions} />
            <Button ref={inviteButtonRef} onClick={() => setInviteOpen(true)}>
              <UserPlus />
              Invite user
            </Button>
          </>
        ) : null
      }
    >
      {canManageUsers ? (
        <Tabs
          value={activeView}
          onValueChange={(next) =>
            updateParams({ tab: next === 'users' ? null : next, user: null })
          }
          className="gap-4"
        >
          <TabsList aria-label="User management views">
            <TabsTrigger value="users">
              Users
              <span className="rounded-full bg-secondary px-1.5 font-mono text-[11px] tabular text-muted-foreground">
                {totalUsers}
              </span>
            </TabsTrigger>
            <TabsTrigger value="invitations">
              Invitations
              {pendingInvites ? (
                <span className="rounded-full bg-primary-soft px-1.5 font-mono text-[11px] tabular text-primary-soft-foreground">
                  {pendingInvites}
                </span>
              ) : null}
            </TabsTrigger>
            <TabsTrigger value="audit">Audit log</TabsTrigger>
          </TabsList>
          <TabsContent value="users">{directory}</TabsContent>
          <TabsContent value="invitations">
            {activeView === 'invitations' && canManageUsers ? (
              <UsersPageInvitationsPanel
                onInvite={() => setInviteOpen(true)}
                refreshToken={invitationRefreshToken}
                onPendingChange={setPendingInvites}
              />
            ) : null}
          </TabsContent>
          <TabsContent value="audit">
            {activeView === 'audit' && canManageUsers ? (
              <UsersPageAuditPanel actors={actors} />
            ) : null}
          </TabsContent>
        </Tabs>
      ) : (
        directory
      )}

      <UsersPageMemberSheet
        user={activeUser}
        open={Boolean(activeUser)}
        isSelf={activeUser ? isSelf(activeUser) : false}
        canManage={canManageUsers}
        rolePermissions={rolePermissions}
        loadingPermissions={loadingRolePermissions}
        submitting={sheetSubmitting}
        error={sheetError}
        onOpenChange={(open) => {
          if (!open) {
            setSheetError(null)
            updateParams({ user: null })
          }
        }}
        onSubmit={(draft) => void handleSheetSubmit(draft)}
        onDelete={(user) => setDeleteTargets([user])}
      />

      <DashboardInviteDialog
        isOpen={inviteOpen}
        onClose={() => {
          setInviteOpen(false)
          window.requestAnimationFrame(() =>
            inviteButtonRef.current?.focus({ preventScroll: true }),
          )
        }}
        onCreated={() => setInvitationRefreshToken((value) => value + 1)}
      />

      <ConfirmDialog
        isOpen={deleteTargets !== null}
        title={
          deleteTargets?.length === 1
            ? `Delete ${deleteTargets[0].name || deleteTargets[0].email}?`
            : `Delete ${deleteTargets?.length ?? 0} users?`
        }
        description="This permanently removes the dashboard account and its access. Audit records remain available to administrators."
        details={
          deleteTargets ? (
            <code className="font-mono text-[12px]">
              {deleteTargets.map((user) => user.email).join(', ')}
            </code>
          ) : null
        }
        confirmLabel={deleteTargets && deleteTargets.length > 1 ? 'Delete users' : 'Delete user'}
        tone="danger"
        pending={deletePending}
        onCancel={() => setDeleteTargets(null)}
        onConfirm={confirmDelete}
      />

      {canManageUsers && activeView === 'users' ? (
        <BulkBar count={selectedUsers.length} onClear={() => setSelected(new Set())}>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm">
                <UserCog />
                Change role
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="center" side="top" className="w-44">
              <DropdownMenuLabel>Set role for {selectedUsers.length}</DropdownMenuLabel>
              {DASHBOARD_ROLES.map((role) => (
                <DropdownMenuItem
                  key={role}
                  onSelect={() =>
                    void changeUsers(selectedUsers, { role }, `role changed to ${roleLabel(role)}`)
                  }
                >
                  {ROLE_META[role].label}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void changeUsers(selectedUsers, { status: 'inactive' }, 'deactivated')}
          >
            <Ban />
            Deactivate
          </Button>
          <Button
            variant="destructive-ghost"
            size="sm"
            onClick={() => setDeleteTargets(selectedUsers)}
          >
            <Trash2 />
            Delete
          </Button>
        </BulkBar>
      ) : null}
    </SettingsSection>
  )
}

export default UsersPage
