import React, { useEffect, useState } from 'react'
import styles from './UsersPageUserDialog.module.css'
import type { UsersPageRolePermissions } from './usersPageSupport'

export type UsersPageUserDialogMode = 'create' | 'edit'

export interface UsersPageUserDraft {
  email: string
  name: string
  password: string
  role: string
  status: string
}

interface UsersPageUserDialogProps {
  isOpen: boolean
  mode: UsersPageUserDialogMode
  initialValues: UsersPageUserDraft
  roleOptions: readonly string[]
  rolePermissions: UsersPageRolePermissions
  isLoadingRolePermissions: boolean
  statusOptions: readonly string[]
  isSubmitting: boolean
  error: string | null
  onClose: () => void
  onSubmit: (values: UsersPageUserDraft) => void
}

export default function UsersPageUserDialog({
  isOpen,
  mode,
  initialValues,
  roleOptions,
  rolePermissions,
  isLoadingRolePermissions,
  statusOptions,
  isSubmitting,
  error,
  onClose,
  onSubmit,
}: UsersPageUserDialogProps) {
  const [values, setValues] = useState<UsersPageUserDraft>(initialValues)

  useEffect(() => {
    if (!isOpen) {
      return
    }

    setValues(initialValues)
  }, [initialValues, isOpen])

  useEffect(() => {
    if (!isOpen) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !isSubmitting) {
        onClose()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [isOpen, isSubmitting, onClose])

  if (!isOpen) {
    return null
  }

  const isEditMode = mode === 'edit'
  const fieldIdPrefix = isEditMode ? 'edit-user' : 'create-user'

  const title = isEditMode ? 'Edit user' : 'Create user'
  const subtitle = isEditMode
    ? 'Adjust role access, account status, and optionally rotate the password in one place.'
    : 'Create a dashboard account with the right role before it enters the workspace.'

  const submitLabel = isSubmitting
    ? isEditMode
      ? 'Saving...'
      : 'Creating...'
    : isEditMode
      ? 'Save changes'
      : 'Create user'
  const selectedRolePermissions = rolePermissions[values.role] ?? []

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    onSubmit(values)
  }

  return (
    <div className={styles.overlay} onClick={!isSubmitting ? onClose : undefined}>
      <div
        className={styles.modal}
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="users-dialog-title"
      >
        <div className={styles.header}>
          <div>
            <p className={styles.eyebrow}>{isEditMode ? 'User update' : 'Access invite'}</p>
            <h2 id="users-dialog-title" className={styles.title}>
              {title}
            </h2>
            <p className={styles.subtitle}>{subtitle}</p>
          </div>
          <button
            type="button"
            className={styles.closeButton}
            onClick={onClose}
            disabled={isSubmitting}
            aria-label="Close user dialog"
          >
            ×
          </button>
        </div>

        <form className={styles.form} onSubmit={handleSubmit}>
          {error ? <div className={styles.error}>{error}</div> : null}

          <div className={styles.grid}>
            <label className={styles.field} htmlFor={`${fieldIdPrefix}-email`}>
              <span className={styles.label}>Email</span>
              <input
                id={`${fieldIdPrefix}-email`}
                type="email"
                className={styles.input}
                value={values.email}
                onChange={(event) => setValues((prev) => ({ ...prev, email: event.target.value }))}
                placeholder="you@example.com"
                disabled={isEditMode || isSubmitting}
                required
              />
              {isEditMode ? <span className={styles.hint}>Existing users keep their email address.</span> : null}
            </label>

            <label className={styles.field} htmlFor={`${fieldIdPrefix}-name`}>
              <span className={styles.label}>Name</span>
              <input
                id={`${fieldIdPrefix}-name`}
                type="text"
                className={styles.input}
                value={values.name}
                onChange={(event) => setValues((prev) => ({ ...prev, name: event.target.value }))}
                placeholder="Jane Doe"
                disabled={isEditMode || isSubmitting}
              />
              {isEditMode ? <span className={styles.hint}>Display name changes are not exposed in the current API.</span> : null}
            </label>

            <label className={styles.field} htmlFor={`${fieldIdPrefix}-role`}>
              <span className={styles.label}>Role</span>
              <select
                id={`${fieldIdPrefix}-role`}
                className={styles.select}
                value={values.role}
                onChange={(event) => setValues((prev) => ({ ...prev, role: event.target.value }))}
                disabled={isSubmitting}
              >
                {roleOptions.map((role) => (
                  <option key={role} value={role}>
                    {role}
                  </option>
                ))}
              </select>
            </label>

            {isEditMode ? (
              <label className={styles.field} htmlFor={`${fieldIdPrefix}-status`}>
                <span className={styles.label}>Status</span>
                <select
                  id={`${fieldIdPrefix}-status`}
                  className={styles.select}
                  value={values.status}
                  onChange={(event) => setValues((prev) => ({ ...prev, status: event.target.value }))}
                  disabled={isSubmitting}
                >
                  {statusOptions.map((status) => (
                    <option key={status} value={status}>
                      {status}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}

            <section className={`${styles.permissionsSection} ${styles.fieldWide}`} aria-live="polite">
              <div className={styles.permissionsHeader}>
                <span className={styles.label}>Permissions</span>
                <span className={styles.permissionCount}>{selectedRolePermissions.length}</span>
              </div>
              <span className={styles.hint}>Effective permissions granted by the selected role.</span>
              {isLoadingRolePermissions ? (
                <p className={styles.emptyState}>Loading role permissions...</p>
              ) : selectedRolePermissions.length > 0 ? (
                <ul className={styles.permissionList}>
                  {selectedRolePermissions.map((permission) => (
                    <li key={permission} className={styles.permissionPill}>
                      {permission}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className={styles.emptyState}>No permissions configured for this role.</p>
              )}
            </section>

            <label className={`${styles.field} ${styles.fieldWide}`} htmlFor={`${fieldIdPrefix}-password`}>
              <span className={styles.label}>{isEditMode ? 'New password' : 'Password'}</span>
              <input
                id={`${fieldIdPrefix}-password`}
                type="password"
                className={styles.input}
                value={values.password}
                onChange={(event) => setValues((prev) => ({ ...prev, password: event.target.value }))}
                placeholder={isEditMode ? 'Leave blank to keep the current password' : 'Choose a strong password'}
                disabled={isSubmitting}
                required={!isEditMode}
              />
              <span className={styles.hint}>
                {isEditMode
                  ? 'If set, the password reset endpoint runs after the role and status update.'
                  : 'A password is required for the user to sign in to the dashboard.'}
              </span>
            </label>
          </div>

          <div className={styles.footer}>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button type="submit" className={styles.primaryButton} disabled={isSubmitting}>
              {submitLabel}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
