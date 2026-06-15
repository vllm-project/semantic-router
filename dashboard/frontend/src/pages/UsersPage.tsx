import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import DashboardSurfaceHero from '../components/DashboardSurfaceHero'
import { DataTable, type Column } from '../components/DataTable'
import styles from './UsersPage.module.css'
import UsersPageUserDialog, {
  type UsersPageUserDialogMode,
  type UsersPageUserDraft,
} from './UsersPageUserDialog'
import {
  EMPTY_ROLE_PERMISSIONS,
  type UsersPageRolePermissions,
  type UsersPageRolePermissionsPayload,
} from './usersPageSupport'

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

type AuditLog = {
  id: number
  userId?: string
  action: string
  resource: string
  method: string
  path: string
  ip: string
  userAgent: string
  statusCode: number
  createdAt: number
  extraJson?: string
}

type ToastType = 'error' | 'success'

type ToastState = {
  type: ToastType
  message: string
}

const ROLE_OPTIONS = ['admin', 'write', 'read'] as const
const STATUS_OPTIONS = ['active', 'inactive'] as const
const PAGE_SIZE_OPTIONS = [10, 20, 50] as const

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
  const canManageUsers = currentUser?.role === 'admin'
  const canViewUsers = canManageUsers

  const [users, setUsers] = useState<AdminUser[]>([])
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([])
  const [rolePermissions, setRolePermissions] = useState<UsersPageRolePermissions>(EMPTY_ROLE_PERMISSIONS)

  const [loadingUsers, setLoadingUsers] = useState(true)
  const [loadingAudits, setLoadingAudits] = useState(false)
  const [loadingRolePermissions, setLoadingRolePermissions] = useState(false)

  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState<number>(10)

  const [toast, setToast] = useState<ToastState | null>(null)

  const [showAudit, setShowAudit] = useState(false)
  const [dialogMode, setDialogMode] = useState<UsersPageUserDialogMode | null>(null)
  const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null)
  const [dialogError, setDialogError] = useState<string | null>(null)
  const [dialogSubmitting, setDialogSubmitting] = useState(false)

  const userHeaders = useMemo(
    () => ({
      'Content-Type': 'application/json',
    }),
    []
  )

  const fetchUsers = useCallback(async () => {
    setLoadingUsers(true)
    try {
      const q = new URLSearchParams()
      if (statusFilter !== 'all') {
        q.set('status', statusFilter)
      }
      const querySuffix = q.toString() ? `?${q.toString()}` : ''
      const response = await fetch(`/api/admin/users${querySuffix}`, {
        method: 'GET',
        headers: userHeaders,
      })
      if (!response.ok) {
        throw new Error(await getResponseError(response))
      }
      const payload = (await response.json()) as { users: AdminUser[] }
      setUsers(payload.users || [])
      setPage(1)
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    } finally {
      setLoadingUsers(false)
    }
  }, [statusFilter, userHeaders])

  const fetchAuditLogs = useCallback(async () => {
    if (!canManageUsers) {
      return
    }

    setLoadingAudits(true)
    try {
      const response = await fetch('/api/admin/audit-logs?limit=100', {
        method: 'GET',
        headers: userHeaders,
      })
      if (!response.ok) {
        throw new Error(await getResponseError(response))
      }
      const payload = (await response.json()) as AuditLog[]
      setAuditLogs(payload)
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    } finally {
      setLoadingAudits(false)
    }
  }, [canManageUsers, userHeaders])

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
      return
    }

    void fetchUsers()
  }, [canViewUsers, fetchUsers])

  useEffect(() => {
    if (showAudit) {
      void fetchAuditLogs()
    }
  }, [showAudit, fetchAuditLogs])

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

  const onDelete = async (id: string) => {
    if (!canManageUsers) {
      return
    }
    if (!window.confirm('Delete this user?')) {
      return
    }

    try {
      const response = await fetch(`/api/admin/users/${id}`, {
        method: 'DELETE',
        headers: userHeaders,
      })
      if (!response.ok && response.status !== 204) {
        throw new Error(await getResponseError(response))
      }
      setToast({ type: 'success', message: 'User deleted.' })
      await fetchUsers()
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    }
  }

  const filteredUsers = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) {
      return users
    }

    return users.filter(
      (user) =>
        user.email.toLowerCase().includes(normalized) ||
        user.name.toLowerCase().includes(normalized) ||
        user.role.toLowerCase().includes(normalized)
    )
  }, [users, query])

  const totalPages = Math.max(1, Math.ceil(filteredUsers.length / pageSize))
  const currentPage = Math.min(page, totalPages)
  const pagedUsers = useMemo(
    () => filteredUsers.slice((currentPage - 1) * pageSize, currentPage * pageSize),
    [filteredUsers, currentPage, pageSize]
  )

  const activeUsers = useMemo(
    () => users.filter((user) => user.status === 'active').length,
    [users]
  )

  const privilegedUsers = useMemo(
    () => users.filter((user) => user.role === 'admin').length,
    [users]
  )

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
    setPage(1)
  }, [query, pageSize])

  const userColumns: Column<AdminUser>[] = useMemo(
    () => [
      { key: 'email', header: 'Email', width: '240px', sortable: true },
      { key: 'name', header: 'Name', width: '180px', sortable: true },
      { key: 'role', header: 'Role', width: '150px', sortable: true },
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
      { key: 'createdAt', header: 'Created', width: '170px', render: (row) => formatTs(row.createdAt) },
      { key: 'lastLoginAt', header: 'Last Login', width: '170px', render: (row) => formatTs(row.lastLoginAt) },
    ],
    []
  )

  const auditColumns: Column<AuditLog>[] = useMemo(
    () => [
      { key: 'id', header: 'ID', width: '80px', sortable: true, render: (row) => `#${row.id}` },
      { key: 'createdAt', header: 'Time', width: '180px', render: (row) => formatTs(row.createdAt) },
      { key: 'action', header: 'Action', width: '150px' },
      { key: 'resource', header: 'Resource', width: '200px' },
      { key: 'method', header: 'Method', width: '90px' },
      { key: 'statusCode', header: 'Code', width: '90px', render: (row) => row.statusCode || '-' },
      {
        key: 'path',
        header: 'Path',
        width: '220px',
        render: (row) => <code className={styles.code}>{row.path}</code>,
      },
      { key: 'ip', header: 'IP', width: '150px', render: (row) => row.ip || '-' },
      { key: 'userId', header: 'User ID', width: '180px', render: (row) => row.userId || '-' },
    ],
    []
  )

  return (
    <div className={styles.page}>
      <DashboardSurfaceHero
        eyebrow="Access"
        title="Users"
        description="Manage dashboard users, privileged roles, and lifecycle controls without leaving the admin workspace."
        meta={[
          { label: 'Current surface', value: showAudit ? 'Audit logs' : 'User directory' },
          { label: 'Active accounts', value: `${activeUsers} active` },
          { label: 'Privileged users', value: `${privilegedUsers} elevated` },
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
        <div className={`${styles.toast} ${toast.type === 'error' ? styles.toastError : styles.toastSuccess}`}>
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
        ) : showAudit ? (
          <section className={styles.card}>
            <div className={styles.sectionHeader}>
              <div>
                <h2 className={styles.sectionTitle}>Audit logs</h2>
                <p className={styles.sectionDescription}>
                  Review user-management activity and privileged account changes across the dashboard.
                </p>
              </div>
            </div>

            {loadingAudits ? (
              <div className={styles.loading}>Loading audit logs...</div>
            ) : (
              <DataTable
                columns={auditColumns}
                data={auditLogs}
                keyExtractor={(row) => `${row.id}`}
                onEdit={undefined}
                onDelete={undefined}
                emptyMessage="No audit log entries found."
              />
            )}
          </section>
        ) : (
          <section className={styles.card}>
            <div className={styles.sectionHeader}>
              <div>
                <h2 className={styles.sectionTitle}>User directory</h2>
                <p className={styles.sectionDescription}>
                  Search active accounts, review roles, and open the centered editor to update access.
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
                <input
                  className={styles.search}
                  type="text"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search by email, name, role"
                />
                <label className={styles.filterGroup}>
                  <span>Status</span>
                  <select
                    className={styles.filter}
                    value={statusFilter}
                    onChange={(event) => setStatusFilter(event.target.value)}
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
                    onChange={(event) => setPageSize(Number.parseInt(event.target.value, 10))}
                  >
                    {PAGE_SIZE_OPTIONS.map((size) => (
                      <option key={size} value={size}>
                        {size}
                      </option>
                    ))}
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
                data={pagedUsers}
                keyExtractor={(row) => row.id}
                onEdit={canManageUsers ? openEditDialog : undefined}
                onDelete={canManageUsers ? (row) => onDelete(row.id) : undefined}
                className={styles.tableContainer}
                emptyMessage="No users found for the current filters."
              />
            )}

            <div className={styles.pagination}>
              <span>
                Page {currentPage} / {totalPages} · {filteredUsers.length} users
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
    </div>
  )
}

export default UsersPage
