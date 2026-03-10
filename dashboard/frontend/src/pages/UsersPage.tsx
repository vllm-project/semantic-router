import React, { useEffect, useMemo, useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

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

const ROLE_OPTIONS = ['super_admin', 'admin', 'operator', 'user', 'readonly'] as const
const STATUS_OPTIONS = ['active', 'inactive'] as const

const formatTs = (value?: number) => {
  if (!value) {
    return '-'
  }
  return new Date(value * 1000).toLocaleString()
}

const initialForm = {
  email: '',
  name: '',
  password: '',
  role: 'user',
  status: 'active',
}

const UsersPage: React.FC = () => {
  const { user: currentUser } = useAuth()
  const canManageUsers = currentUser?.role === 'admin' || currentUser?.role === 'super_admin'

  const [users, setUsers] = useState<AdminUser[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [editingUserId, setEditingUserId] = useState<string | null>(null)
  const [editRole, setEditRole] = useState('')
  const [editStatus, setEditStatus] = useState('')
  const [form, setForm] = useState(initialForm)
  const [success, setSuccess] = useState('')

  const headers = useMemo(
    () => ({
      'Content-Type': 'application/json',
    }),
    []
  )

  useEffect(() => {
    const controller = new AbortController()
    const run = async () => {
      setLoading(true)
      setError('')
      try {
        const response = await fetch('/api/admin/users', {
          method: 'GET',
          headers,
          signal: controller.signal,
        })
        if (!response.ok) {
          const text = await response.text()
          throw new Error(text || `Request failed: ${response.status}`)
        }
        const payload = (await response.json()) as { users: AdminUser[] }
        setUsers(payload.users || [])
      } catch (err) {
        if ((err as Error).name === 'AbortError') {
          return
        }
        setError((err as Error).message)
      } finally {
        setLoading(false)
      }
    }

    void run()
    return () => controller.abort()
  }, [headers])

  const refresh = async () => {
    setError('')
    setSuccess('')
    setLoading(true)
    try {
      const response = await fetch('/api/admin/users', { method: 'GET', headers })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      const payload = (await response.json()) as { users: AdminUser[] }
      setUsers(payload.users || [])
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const onCreate = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!canManageUsers) {
      setError('No permission to create users')
      return
    }
    setError('')
    setSuccess('')
    try {
      const response = await fetch('/api/admin/users', {
        method: 'POST',
        headers,
        body: JSON.stringify(form),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setForm(initialForm)
      setSuccess('User created')
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const onSave = async (id: string) => {
    if (!canManageUsers) {
      return
    }
    const payload = {
      role: editRole,
      status: editStatus,
    }
    try {
      const response = await fetch(`/api/admin/users/${id}`, {
        method: 'PATCH',
        headers,
        body: JSON.stringify(payload),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setEditingUserId(null)
      setSuccess('User updated')
      await refresh()
    } catch (err) {
      setError((err as Error).message)
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
        headers,
      })
      if (!response.ok && response.status !== 204) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setSuccess('User deleted')
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const onResetPassword = async (id: string) => {
    if (!canManageUsers) {
      return
    }
    const pw = window.prompt('Enter new password')
    if (!pw) {
      return
    }
    try {
      const response = await fetch('/api/admin/users/password', {
        method: 'POST',
        headers,
        body: JSON.stringify({ userId: id, password: pw }),
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `Request failed: ${response.status}`)
      }
      setSuccess('Password updated')
    } catch (err) {
      setError((err as Error).message)
    }
  }

  if (loading) {
    return <div style={{ padding: '2rem' }}>Loading users...</div>
  }

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      <h1>Users</h1>
      <p style={{ color: 'var(--color-text-secondary)' }}>
        Manage dashboard users and control-role permissions for the dashboard.
      </p>

      {error ? <div style={{ color: 'var(--color-danger)' }}>{error}</div> : null}
      {success ? <div style={{ color: 'var(--color-success)' }}>{success}</div> : null}

      {canManageUsers ? (
        <form
          onSubmit={onCreate}
          style={{
            display: 'grid',
            gridTemplateColumns: '1.5fr 1fr 1fr 1fr auto',
            gap: '0.75rem',
            alignItems: 'end',
            padding: '1rem',
            border: '1px solid var(--color-border)',
            borderRadius: '0.75rem',
          }}
        >
          <label>
            Email
            <input
              required
              value={form.email}
              onChange={e => setForm(prev => ({ ...prev, email: e.target.value }))}
              style={{ width: '100%', marginTop: 4 }}
              type="email"
            />
          </label>
          <label>
            Name
            <input
              value={form.name}
              onChange={e => setForm(prev => ({ ...prev, name: e.target.value }))}
              style={{ width: '100%', marginTop: 4 }}
            />
          </label>
          <label>
            Password
            <input
              required
              value={form.password}
              onChange={e => setForm(prev => ({ ...prev, password: e.target.value }))}
              style={{ width: '100%', marginTop: 4 }}
              type="password"
            />
          </label>
          <label>
            Role
            <select
              value={form.role}
              onChange={e => setForm(prev => ({ ...prev, role: e.target.value }))}
              style={{ width: '100%', marginTop: 4 }}
            >
              {ROLE_OPTIONS.map(role => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          </label>
          <button type="submit">Create user</button>
        </form>
      ) : null}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Email</th>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Role</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Created</th>
              <th style={thStyle}>Last Login</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {users.length === 0 ? (
              <tr>
                <td colSpan={7} style={{ ...tdStyle, textAlign: 'center' }}>
                  No users found.
                </td>
              </tr>
            ) : (
              users.map(row => {
                const isEditing = editingUserId === row.id
                return (
                  <tr key={row.id}>
                    <td style={tdStyle}>{row.email}</td>
                    <td style={tdStyle}>{row.name}</td>
                    <td style={tdStyle}>
                      {isEditing ? (
                        <select
                          value={editRole}
                          onChange={e => setEditRole(e.target.value)}
                        >
                          {ROLE_OPTIONS.map(role => (
                            <option key={role} value={role}>
                              {role}
                            </option>
                          ))}
                        </select>
                      ) : (
                        row.role
                      )}
                    </td>
                    <td style={tdStyle}>
                      {isEditing ? (
                        <select
                          value={editStatus}
                          onChange={e => setEditStatus(e.target.value)}
                        >
                          {STATUS_OPTIONS.map(status => (
                            <option key={status} value={status}>
                              {status}
                            </option>
                          ))}
                        </select>
                      ) : (
                        row.status
                      )}
                    </td>
                    <td style={tdStyle}>{formatTs(row.createdAt)}</td>
                    <td style={tdStyle}>{formatTs(row.lastLoginAt)}</td>
                    <td style={tdStyle}>
                      {isEditing ? (
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <button
                            type="button"
                            onClick={() => {
                              onSave(row.id)
                            }}
                          >
                            Save
                          </button>
                          <button
                            type="button"
                            onClick={() => setEditingUserId(null)}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                          <button
                            type="button"
                            onClick={() => {
                              setEditingUserId(row.id)
                              setEditRole(row.role)
                              setEditStatus(row.status)
                            }}
                          >
                            Edit
                          </button>
                          <button type="button" onClick={() => onResetPassword(row.id)}>
                            Reset password
                          </button>
                          <button type="button" onClick={() => onDelete(row.id)}>
                            Delete
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  borderBottom: '1px solid var(--color-border)',
  padding: '0.5rem',
  color: 'var(--color-text-secondary)',
  fontWeight: 600,
}

const tdStyle: React.CSSProperties = {
  borderBottom: '1px solid var(--color-border)',
  padding: '0.5rem',
}

export default UsersPage
