import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import DashboardSurfaceHero from '../components/DashboardSurfaceHero'
import { DataTable, type Column } from '../components/DataTable'
import ConfirmDialog from '../components/ConfirmDialog'
import styles from './UsersPage.module.css'
import UsersPageUserDialog, {
  type UsersPageUserDialogMode,
  type UsersPageUserDraft,
} from './UsersPageUserDialog'
import {
  createLatestUsersRequest,
  EMPTY_ROLE_PERMISSIONS,
  isUsersRequestAbortError,
  type UsersPageRolePermissions,
  type UsersPageRolePermissionsPayload,
} from './usersPageSupport'
import {
  canManageUsers as canManageDashboardUsers,
  canViewUsers as canViewDashboardUsers,
} from '../utils/accessControl'
import UsersPageAuditPanel from './UsersPageAuditPanel'

type AdminUser = {
  id: string
  email: string
  name: string
  role: string
  status: string
  createdAt?: number
  updatedAt?: number
  lastLoginAt?: number
}

type ToastType = 'error' | 'success'

type ToastState = {
  type: ToastType
  message: string
}

const ROLE_OPTIONS = ['admin', 'write', 'read'] as const
const STATUS_OPTIONS = ['active', 'inactive'] as const
const PAGE_SIZE_OPTIONS = [10, 20, 50] as const
const SORT_OPTIONS = [
  { value: 'createdAt', label: 'Created' },
  { value: 'email', label: 'Email' },
  { value: 'name', label: 'Name' },
  { value: 'role', label: 'Role' },
  { value: 'lastLoginAt', label: 'Last login' },
] as const

const EMPTY_USER_DRAFT: UsersPageUserDraft = {
  email: '',
  name: '',
  password: '',
  role: 'read',
  status: 'active',
}

const formatTs = (value?: number) => {
  if (!value) {
    return '-'
  }

  return new Date(value * 1000).toLocaleString()
}

const getResponseError = async (response: Response) => {
  const text = await response.text()
  return text || `Request failed: ${response.status}`
}

const UsersPage: React.FC = () => {
  const { user: currentUser } = useAuth()
  const canManageUsers = canManageDashboardUsers(currentUser)
  const canViewUsers = canViewDashboardUsers(currentUser)

  const [users, setUsers] = useState<AdminUser[]>([])
  const [totalUsers, setTotalUsers] = useState(0)
  const [activeUserCount, setActiveUserCount] = useState(0)
  const [privilegedUserCount, setPrivilegedUserCount] = useState(0)
  const [rolePermissions, setRolePermissions] =
    useState<UsersPageRolePermissions>(EMPTY_ROLE_PERMISSIONS)

  const [loadingUsers, setLoadingUsers] = useState(true)
  const [loadingRolePermissions, setLoadingRolePermissions] = useState(false)

  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [sortField, setSortField] = useState('createdAt')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState<number>(10)

  const [toast, setToast] = useState<ToastState | null>(null)

  const [showAudit, setShowAudit] = useState(false)
  const [dialogMode, setDialogMode] = useState<UsersPageUserDialogMode | null>(null)
  const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null)
  const [dialogError, setDialogError] = useState<string | null>(null)
  const [dialogSubmitting, setDialogSubmitting] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<AdminUser | null>(null)
  const [deletePending, setDeletePending] = useState(false)
  const [userRequests] = useState(createLatestUsersRequest)

  const userHeaders = useMemo(
    () => ({
      'Content-Type': 'application/json',
    }),
    [],
  )

  const fetchUsers = useCallback(async () => {
    const request = userRequests.start()
    setLoadingUsers(true)
    try {
      const q = new URLSearchParams()
      if (statusFilter !== 'all') {
        q.set('status', statusFilter)
      }
      if (debouncedQuery) {
        q.set('q', debouncedQuery)
      }
      q.set('page', String(page))
      q.set('limit', String(pageSize))
      q.set('sort', sortField)
      q.set('order', sortOrder)
      const querySuffix = q.toString() ? `?${q.toString()}` : ''
      const response = await fetch(`/api/admin/users${querySuffix}`, {
        method: 'GET',
        headers: userHeaders,
        signal: request.signal,
      })
      if (!response.ok) {
        throw new Error(await getResponseError(response))
      }
      const payload = (await response.json()) as {
        users: AdminUser[]
        total?: number
        page?: number
        active?: number
        privileged?: number
      }
      if (!request.isCurrent()) {
        return
      }
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
    } catch (err) {
      if (!request.isCurrent() || isUsersRequestAbortError(err)) {
        return
      }
      setToast({ type: 'error', message: (err as Error).message })
    } finally {
      if (request.isCurrent()) {
        setLoadingUsers(false)
      }
      request.finish()
    }
  }, [debouncedQuery, page, pageSize, sortField, sortOrder, statusFilter, userHeaders, userRequests])

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
        headers: userHeaders,
      })
      if (!response.ok) {
        throw new Error(await getResponseError(response))
      }
      const payload = (await response.json()) as UsersPageRolePermissionsPayload
      setRolePermissions(payload.rolePermissions ?? EMPTY_ROLE_PERMISSIONS)
    } catch (err) {
      setRolePermissions(EMPTY_ROLE_PERMISSIONS)
      setToast({ type: 'error', message: (err as Error).message })
    } finally {
      setLoadingRolePermissions(false)
    }
  }, [canManageUsers, userHeaders])

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
    if (!canManageUsers) {
      setRolePermissions(EMPTY_ROLE_PERMISSIONS)
      setLoadingRolePermissions(false)
      return
    }

    void fetchRolePermissions()
  }, [canManageUsers, fetchRolePermissions])

  const closeDialog = () => {
    setDialogMode(null)
    setSelectedUser(null)
    setDialogError(null)
    setDialogSubmitting(false)
  }

  const openCreateDialog = () => {
    if (!canManageUsers) {
      return
    }

    if (!loadingRolePermissions && Object.keys(rolePermissions).length === 0) {
      void fetchRolePermissions()
    }

    setDialogMode('create')
    setSelectedUser(null)
    setDialogError(null)
  }

  const openEditDialog = (user: AdminUser) => {
    if (!canManageUsers) {
      return
    }

    if (!loadingRolePermissions && Object.keys(rolePermissions).length === 0) {
      void fetchRolePermissions()
    }

    setDialogMode('edit')
    setSelectedUser(user)
    setDialogError(null)
  }

  const handleDialogSubmit = async (values: UsersPageUserDraft) => {
    if (!dialogMode || !canManageUsers) {
      return
    }

    setDialogSubmitting(true)
    setDialogError(null)

    try {
      if (dialogMode === 'create') {
        const response = await fetch('/api/admin/users', {
          method: 'POST',
          headers: userHeaders,
          body: JSON.stringify({
            email: values.email,
            name: values.name,
            password: values.password,
            role: values.role,
          }),
        })
        if (!response.ok) {
          throw new Error(await getResponseError(response))
        }

        closeDialog()
        setToast({ type: 'success', message: 'User created.' })
        await fetchUsers()
        return
      }

      if (!selectedUser) {
        throw new Error('No user selected for editing.')
      }

      const patchResponse = await fetch(`/api/admin/users/${selectedUser.id}`, {
        method: 'PATCH',
        headers: userHeaders,
        body: JSON.stringify({ role: values.role, status: values.status }),
      })
      if (!patchResponse.ok) {
        throw new Error(await getResponseError(patchResponse))
      }

      if (values.password.trim()) {
        const passwordResponse = await fetch('/api/admin/users/password', {
          method: 'POST',
          headers: userHeaders,
          body: JSON.stringify({ userId: selectedUser.id, password: values.password }),
        })
        if (!passwordResponse.ok) {
          throw new Error(await getResponseError(passwordResponse))
        }
      }

      closeDialog()
      setToast({
        type: 'success',
        message: values.password.trim() ? 'User updated and password rotated.' : 'User updated.',
      })
      await fetchUsers()
    } catch (err) {
      setDialogError((err as Error).message)
    } finally {
      setDialogSubmitting(false)
    }
  }

  const onDelete = (user: AdminUser) => {
    if (!canManageUsers) {
      return
    }
    setDeleteTarget(user)
  }

  const confirmDelete = async () => {
    if (!deleteTarget || !canManageUsers) {
      return
    }

    setDeletePending(true)
    try {
      const response = await fetch(`/api/admin/users/${deleteTarget.id}`, {
        method: 'DELETE',
        headers: userHeaders,
      })
      if (!response.ok && response.status !== 204) {
        throw new Error(await getResponseError(response))
      }
      setDeleteTarget(null)
      setToast({ type: 'success', message: 'User deleted.' })
      await fetchUsers()
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    } finally {
      setDeletePending(false)
    }
  }

  const totalPages = Math.max(1, Math.ceil(totalUsers / pageSize))
  const currentPage = Math.min(page, totalPages)

  const dialogInitialValues = useMemo<UsersPageUserDraft>(() => {
    if (!selectedUser) {
      return EMPTY_USER_DRAFT
    }

    return {
      email: selectedUser.email,
      name: selectedUser.name,
      password: '',
      role: selectedUser.role,
      status: selectedUser.status,
    }
  }, [selectedUser])

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(query.trim()), 300)
    return () => window.clearTimeout(timer)
  }, [query])

  const userColumns: Column<AdminUser>[] = useMemo(
    () => [
      { key: 'email', header: 'Email', width: '240px' },
      { key: 'name', header: 'Name', width: '180px' },
      { key: 'role', header: 'Role', width: '150px' },
      {
        key: 'status',
        header: 'Status',
        width: '130px',
        render: (row) => (
          <span
            className={`${styles.statusPill} ${row.status === 'inactive' ? styles.statusPillInactive : ''}`}
          >
            {row.status}
          </span>
        ),
      },
      {
        key: 'createdAt',
        header: 'Created',
        width: '170px',
        render: (row) => formatTs(row.createdAt),
      },
      {
        key: 'lastLoginAt',
        header: 'Last Login',
        width: '170px',
        render: (row) => formatTs(row.lastLoginAt),
      },
    ],
    [],
  )

  return (
    <div className={styles.page}>
      <DashboardSurfaceHero
        eyebrow="Access"
        title="Users"
        description="Manage dashboard users, privileged roles, and lifecycle controls without leaving the admin workspace."
        meta={[
          { label: 'Current surface', value: showAudit ? 'Audit logs' : 'User directory' },
          { label: 'Active accounts', value: `${activeUserCount} active` },
          { label: 'Privileged users', value: `${privilegedUserCount} elevated` },
        ]}
        panelEyebrow="Workspace access"
        panelTitle="Dashboard user control"
        panelDescription="Keep account provisioning, role changes, and audit history in one operator-facing surface."
        pills={[
          {
            label: 'User list',
            active: !showAudit,
            onClick: () => setShowAudit(false),
          },
          ...(canManageUsers
            ? [
                {
                  label: 'Audit logs',
                  active: showAudit,
                  onClick: () => setShowAudit(true),
                },
              ]
            : []),
        ]}
        panelFooter={
          canManageUsers ? (
            <button type="button" className={styles.heroActionButton} onClick={openCreateDialog}>
              Create user
            </button>
          ) : null
        }
      />

      {toast ? (
        <div
          className={`${styles.toast} ${toast.type === 'error' ? styles.toastError : styles.toastSuccess}`}
        >
          {toast.message}
        </div>
      ) : null}

      <div className={styles.body}>
        {!canViewUsers ? (
          <section className={styles.card}>
            <div className={styles.sectionHeader}>
              <div>
                <h2 className={styles.sectionTitle}>Access required</h2>
                <p className={styles.sectionDescription}>
                  You do not have permission to view dashboard user management.
                </p>
              </div>
            </div>
          </section>
        ) : showAudit && canManageUsers ? (
          <UsersPageAuditPanel />
        ) : (
          <section className={styles.card}>
            <div className={styles.sectionHeader}>
              <div>
                <h2 className={styles.sectionTitle}>User directory</h2>
                <p className={styles.sectionDescription}>
                  Search active accounts, review roles, and open the centered editor to update
                  access.
                </p>
              </div>
              {canManageUsers ? (
                <button type="button" className={styles.secondaryButton} onClick={openCreateDialog}>
                  New account
                </button>
              ) : null}
            </div>

            <div className={styles.toolbar}>
              <div className={styles.toolbarLeft}>
                <label className={styles.filterGroup}>
                  <span>Search</span>
                  <input
                    className={styles.search}
                    type="search"
                    value={query}
                    onChange={(event) => {
                      setQuery(event.target.value)
                      setPage(1)
                    }}
                    placeholder="Email, name, or role"
                  />
                </label>
                <label className={styles.filterGroup}>
                  <span>Status</span>
                  <select
                    className={styles.filter}
                    value={statusFilter}
                    onChange={(event) => {
                      setStatusFilter(event.target.value)
                      setPage(1)
                    }}
                  >
                    <option value="all">All</option>
                    <option value="active">Active</option>
                    <option value="inactive">Inactive</option>
                  </select>
                </label>

                <label className={styles.filterGroup}>
                  <span>Page size</span>
                  <select
                    className={styles.filter}
                    value={pageSize}
                    onChange={(event) => {
                      setPageSize(Number.parseInt(event.target.value, 10))
                      setPage(1)
                    }}
                  >
                    {PAGE_SIZE_OPTIONS.map((size) => (
                      <option key={size} value={size}>
                        {size}
                      </option>
                    ))}
                  </select>
                </label>

                <label className={styles.filterGroup}>
                  <span>Sort</span>
                  <select
                    className={styles.filter}
                    value={sortField}
                    onChange={(event) => {
                      setSortField(event.target.value)
                      setPage(1)
                    }}
                  >
                    {SORT_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className={styles.filterGroup}>
                  <span>Order</span>
                  <select
                    className={styles.filter}
                    value={sortOrder}
                    onChange={(event) => {
                      setSortOrder(event.target.value === 'asc' ? 'asc' : 'desc')
                      setPage(1)
                    }}
                  >
                    <option value="desc">Descending</option>
                    <option value="asc">Ascending</option>
                  </select>
                </label>
              </div>

              <button
                className={styles.secondaryButton}
                type="button"
                onClick={fetchUsers}
                disabled={loadingUsers}
              >
                Refresh
              </button>
            </div>

            {loadingUsers ? (
              <div className={styles.loading}>Loading users...</div>
            ) : (
              <DataTable
                columns={userColumns}
                data={users}
                keyExtractor={(row) => row.id}
                onEdit={canManageUsers ? openEditDialog : undefined}
                onDelete={canManageUsers ? onDelete : undefined}
                className={styles.tableContainer}
                emptyMessage="No users found for the current filters."
              />
            )}

            <div className={styles.pagination}>
              <span>
                Page {currentPage} / {totalPages} · {totalUsers} users
              </span>

              <div className={styles.paginationActions}>
                <button
                  type="button"
                  onClick={() => setPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage <= 1}
                >
                  Previous
                </button>
                <button
                  type="button"
                  onClick={() => setPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage >= totalPages}
                >
                  Next
                </button>
              </div>
            </div>
          </section>
        )}
      </div>

      <UsersPageUserDialog
        isOpen={dialogMode !== null}
        mode={dialogMode ?? 'create'}
        initialValues={dialogInitialValues}
        roleOptions={ROLE_OPTIONS}
        rolePermissions={rolePermissions}
        isLoadingRolePermissions={loadingRolePermissions}
        statusOptions={STATUS_OPTIONS}
        isSubmitting={dialogSubmitting}
        error={dialogError}
        onClose={closeDialog}
        onSubmit={handleDialogSubmit}
      />

      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title={`Delete ${deleteTarget?.name || deleteTarget?.email || 'this user'}?`}
        description={
          <p>
            This permanently removes the dashboard account and its access. Audit records remain
            available to administrators.
          </p>
        }
        eyebrow="Account lifecycle"
        confirmLabel="Delete user"
        pending={deletePending}
        details={deleteTarget ? <code>{deleteTarget.email}</code> : null}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={confirmDelete}
      />
    </div>
  )
}

export default UsersPage
