import React, { useEffect, useMemo, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { DataTable, type Column } from '../components/DataTable'
import styles from './UsersPage.module.css'

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

const ROLE_OPTIONS = ['super_admin', 'admin', 'operator', 'user', 'readonly'] as const
const STATUS_OPTIONS = ['active', 'inactive'] as const

const PAGE_SIZE_OPTIONS = [10, 20, 50] as const

const initialForm = {
  email: '',
  name: '',
  password: '',
  role: 'user',
  status: 'active',
}

const formatTs = (value?: number) => {
  if (!value) {
    return '-'
  }
  return new Date(value * 1000).toLocaleString()
}

const UsersPage: React.FC = () => {
  const { user: currentUser } = useAuth()
  const canManageUsers = currentUser?.role === 'admin' || currentUser?.role === 'super_admin'
  const canViewUsers = canManageUsers || currentUser?.role === 'operator' || currentUser?.role === 'readonly' || currentUser?.role === 'user'

  const [users, setUsers] = useState<AdminUser[]>([])
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([])

  const [loadingUsers, setLoadingUsers] = useState(true)
  const [loadingAudits, setLoadingAudits] = useState(false)

  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState<number>(10)

  const [toast, setToast] = useState<ToastState | null>(null)

  const [editingUserId, setEditingUserId] = useState<string | null>(null)
  const [editRole, setEditRole] = useState('')
  const [editStatus, setEditStatus] = useState('')

  const [form, setForm] = useState(initialForm)

  const [showAudit, setShowAudit] = useState(false)

  const userHeaders = useMemo(
    () => ({
      'Content-Type': 'application/json',
    }),
    []
  )

  const fetchUsers = async () => {
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
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      const payload = (await response.json()) as { users: AdminUser[] }
      setUsers(payload.users || [])
      setPage(1)
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    } finally {
      setLoadingUsers(false)
    }
  }

  const fetchAuditLogs = async () => {
    if (!canManageUsers) return
    setLoadingAudits(true)
    try {
      const response = await fetch('/api/admin/audit-logs?limit=100', {
        method: 'GET',
        headers: userHeaders,
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      const payload = (await response.json()) as AuditLog[]
      setAuditLogs(payload)
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    } finally {
      setLoadingAudits(false)
    }
  }

  useEffect(() => {
    if (!canViewUsers) return
    void fetchUsers()
  }, [statusFilter, canViewUsers])

  useEffect(() => {
    if (showAudit) {
      void fetchAuditLogs()
    }
  }, [showAudit])

  const onCreate = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!canManageUsers) {
      setToast({ type: 'error', message: 'No permission to create users.' })
      return
    }

    try {
      const response = await fetch('/api/admin/users', {
        method: 'POST',
        headers: userHeaders,
        body: JSON.stringify(form),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setForm(initialForm)
      setToast({ type: 'success', message: 'User created.' })
      await fetchUsers()
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    }
  }

  const onSave = async (id: string) => {
    if (!canManageUsers) return
    try {
      const response = await fetch(`/api/admin/users/${id}`, {
        method: 'PATCH',
        headers: userHeaders,
        body: JSON.stringify({ role: editRole, status: editStatus }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setEditingUserId(null)
      setToast({ type: 'success', message: 'User updated.' })
      await fetchUsers()
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    }
  }

  const onDelete = async (id: string) => {
    if (!canManageUsers) return
    if (!window.confirm('Delete this user?')) return
    try {
      const response = await fetch(`/api/admin/users/${id}`, {
        method: 'DELETE',
        headers: userHeaders,
      })
      if (!response.ok && response.status !== 204) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setToast({ type: 'success', message: 'User deleted.' })
      await fetchUsers()
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    }
  }

  const onResetPassword = async (id: string) => {
    if (!canManageUsers) return
    const pw = window.prompt('Enter new password')
    if (!pw) return

    try {
      const response = await fetch('/api/admin/users/password', {
        method: 'POST',
        headers: userHeaders,
        body: JSON.stringify({ userId: id, password: pw }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setToast({ type: 'success', message: 'Password updated.' })
    } catch (err) {
      setToast({ type: 'error', message: (err as Error).message })
    }
  }

  const filteredUsers = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) return users

    return users.filter(
      (u) =>
        u.email.toLowerCase().includes(normalized) ||
        u.name.toLowerCase().includes(normalized) ||
        u.role.toLowerCase().includes(normalized)
    )
  }, [users, query])

  const totalPages = Math.max(1, Math.ceil(filteredUsers.length / pageSize))
  const currentPage = Math.min(page, totalPages)
  const pagedUsers = useMemo(
    () => filteredUsers.slice((currentPage - 1) * pageSize, currentPage * pageSize),
    [filteredUsers, currentPage, pageSize]
  )

  useEffect(() => {
    setPage(1)
  }, [query, pageSize])

  const userColumns: Column<AdminUser>[] = useMemo(() => [
    { key: 'email', header: 'Email', width: '220px', sortable: true },
    { key: 'name', header: 'Name', width: '180px', sortable: true },
    {
      key: 'role',
      header: 'Role',
      width: '150px',
      render: (row) => {
        if (editingUserId === row.id && canManageUsers) {
          return (
            <select
              value={editRole}
              onChange={(e) => setEditRole(e.target.value)}
              className={styles.fieldControl}
              onClick={(e) => e.stopPropagation()}
            >
              {ROLE_OPTIONS.map(role => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          )
        }
        return row.role
      },
    },
    {
      key: 'status',
      header: 'Status',
      width: '130px',
      render: (row) => {
        if (editingUserId === row.id && canManageUsers) {
          return (
            <select
              value={editStatus}
              onChange={(e) => setEditStatus(e.target.value)}
              className={styles.fieldControl}
              onClick={(e) => e.stopPropagation()}
            >
              {STATUS_OPTIONS.map(status => (
                <option key={status} value={status}>
                  {status}
                </option>
              ))}
            </select>
          )
        }
        return <span className={styles.statusPill}>{row.status}</span>
      },
    },
    { key: 'createdAt', header: 'Created', width: '170px', render: (row) => formatTs(row.createdAt) },
    { key: 'lastLoginAt', header: 'Last Login', width: '170px', render: (row) => formatTs(row.lastLoginAt) },
  ], [canManageUsers, editRole, editStatus, editingUserId])

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
      {
        key: 'userId',
        header: 'User ID',
        width: '180px',
        render: (row) => row.userId || '-',
      },
    ],
    []
  )

  if (!canViewUsers) {
    return (
      <div className={styles.page}>
        <section className={styles.card}>
          <h1 className={styles.title}>Users</h1>
          <p className={styles.subtitle}>You do not have permission to view users management.</p>
        </section>
      </div>
    )
  }

  return (
    <div className={styles.page}>
      <section className={styles.header}>
        <div>
          <h1 className={styles.title}>Users</h1>
          <p className={styles.subtitle}>
            Manage dashboard users and account lifecycle.
          </p>
        </div>

        <div className={styles.headerActions}>
          <button
            className={`${styles.toggleButton} ${!showAudit ? styles.toggleButtonActive : ''}`}
            onClick={() => setShowAudit(false)}
            type="button"
          >
            User list
          </button>
          {canManageUsers ? (
            <button
              className={`${styles.toggleButton} ${showAudit ? styles.toggleButtonActive : ''}`}
              onClick={() => setShowAudit(true)}
              type="button"
            >
              Audit logs
            </button>
          ) : null}
        </div>
      </section>

      {toast ? (
        <div className={`${styles.toast} ${toast.type === 'error' ? styles.toastError : styles.toastSuccess}`}>
          {toast.message}
        </div>
      ) : null}

      {showAudit ? (
        <>
          <section className={styles.card}>
            <div className={styles.sectionTitle}>Audit logs</div>
            {loadingAudits ? (
              <div className={styles.loading}>Loading audit logs...</div>
            ) : (
              <DataTable
                columns={auditColumns}
                data={auditLogs}
                keyExtractor={(row) => `${row.id}`}
                onEdit={undefined}
                onDelete={undefined}
              />
            )}
          </section>
        </>
      ) : (
        <>
          {canManageUsers ? (
            <section className={styles.card}>
              <div className={styles.sectionTitle}>Create user</div>
              <form className={styles.form} onSubmit={onCreate}>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Email</span>
                  <input
                    type="email"
                    className={styles.fieldControl}
                    value={form.email}
                    onChange={(e) => setForm((prev) => ({ ...prev, email: e.target.value }))}
                    placeholder="you@example.com"
                    required
                  />
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Name</span>
                  <input
                    className={styles.fieldControl}
                    value={form.name}
                    onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                    placeholder="Jane Doe"
                  />
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Password</span>
                  <input
                    type="password"
                    className={styles.fieldControl}
                    value={form.password}
                    onChange={(e) => setForm((prev) => ({ ...prev, password: e.target.value }))}
                    placeholder="Choose a strong password"
                    required
                  />
                </label>

                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Role</span>
                  <select
                    className={styles.fieldControl}
                    value={form.role}
                    onChange={(e) => setForm((prev) => ({ ...prev, role: e.target.value }))}
                  >
                    {ROLE_OPTIONS.map((role) => (
                      <option key={role} value={role}>
                        {role}
                      </option>
                    ))}
                  </select>
                </label>

                <button className={styles.primaryButton} type="submit">
                  Create user
                </button>
              </form>
            </section>
          ) : null}

          <section className={styles.card}>
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
                onEdit={(row) => {
                  if (!canManageUsers) return
                  setEditingUserId(row.id)
                  setEditRole(row.role)
                  setEditStatus(row.status)
                }}
                onDelete={(row) => onDelete(row.id)}
                className={styles.tableContainer}
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

            {editingUserId ? (
              <div className={styles.actionBar}>
                <span>Editing user: {users.find((u) => u.id === editingUserId)?.email}</span>
                <div className={styles.actionButtons}>
                  <button
                    className={styles.primaryButton}
                    type="button"
                    onClick={() => onSave(editingUserId)}
                  >
                    Save
                  </button>
                  <button
                    className={styles.secondaryButton}
                    type="button"
                    onClick={() => setEditingUserId(null)}
                  >
                    Cancel
                  </button>
                  <button
                    className={styles.secondaryButton}
                    type="button"
                    onClick={() => onResetPassword(editingUserId)}
                  >
                    Reset password
                  </button>
                </div>
              </div>
            ) : null}
          </section>
        </>
      )}
    </div>
  )
}

export default UsersPage
