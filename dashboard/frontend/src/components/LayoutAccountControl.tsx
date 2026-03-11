import React, { useEffect, useId } from 'react'
import styles from './LayoutAccountControl.module.css'

interface LayoutAccountControlProps {
  accountName: string
  accountEmail: string
  accountRole?: string
  accountPermissions: string[]
  isOpen: boolean
  onToggle: () => void
  onClose: () => void
  onLogout: () => void
}

function getAccountInitials(name?: string, email?: string): string {
  const source = (name || email || 'User').trim()
  if (!source) {
    return 'U'
  }

  const words = source.split(/\s+/).filter(Boolean)
  if (words.length >= 2) {
    return `${words[0][0]}${words[1][0]}`.toUpperCase()
  }

  return source.slice(0, 2).toUpperCase()
}

function formatRoleLabel(role?: string): string {
  if (!role) {
    return 'Unknown role'
  }

  return `${role.charAt(0).toUpperCase()}${role.slice(1)}`
}

const LayoutAccountControl: React.FC<LayoutAccountControlProps> = ({
  accountName,
  accountEmail,
  accountRole,
  accountPermissions,
  isOpen,
  onToggle,
  onClose,
  onLogout,
}) => {
  const initials = getAccountInitials(accountName, accountEmail)
  const roleLabel = formatRoleLabel(accountRole)
  const dialogId = useId()

  useEffect(() => {
    if (!isOpen) {
      return
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      document.body.style.overflow = previousOverflow
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [isOpen, onClose])

  return (
    <>
      <button
        type="button"
        className={`${styles.trigger} ${isOpen ? styles.triggerActive : ''}`}
        aria-controls={isOpen ? dialogId : undefined}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        aria-label={`Open account details for ${accountName}`}
        onClick={onToggle}
      >
        <span className={styles.triggerAvatar} aria-hidden="true">
          {initials}
        </span>
        <span className={styles.triggerName}>{accountName}</span>
        <svg
          className={`${styles.triggerChevron} ${isOpen ? styles.triggerChevronOpen : ''}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-hidden="true"
        >
          <path d="M3 4.5L6 7.5L9 4.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {isOpen ? (
        <div className={styles.overlay} onClick={onClose}>
          <div
            id={dialogId}
            className={styles.dialog}
            role="dialog"
            aria-modal="true"
            aria-labelledby="layout-account-dialog-title"
            data-testid="layout-account-dialog"
            onClick={(event) => event.stopPropagation()}
          >
            <div className={styles.header}>
              <div className={styles.headerCopy}>
                <span className={styles.eyebrow}>Account</span>
                <h2 id="layout-account-dialog-title" className={styles.title}>
                  Account details
                </h2>
              </div>
              <button
                type="button"
                className={styles.closeButton}
                aria-label="Close account dialog"
                onClick={onClose}
              >
                ×
              </button>
            </div>

            <div className={styles.body}>
              <section className={styles.identitySection}>
                <div className={styles.identityAvatar} aria-hidden="true">
                  {initials}
                </div>
                <dl className={styles.identityGrid}>
                  <div className={styles.identityRow}>
                    <dt>Name</dt>
                    <dd>{accountName}</dd>
                  </div>
                  <div className={styles.identityRow}>
                    <dt>Email</dt>
                    <dd>{accountEmail}</dd>
                  </div>
                  <div className={styles.identityRow}>
                    <dt>Role</dt>
                    <dd>{roleLabel}</dd>
                  </div>
                </dl>
              </section>

              <section className={styles.permissionsSection}>
                <div className={styles.sectionHeading}>
                  <span>Permissions</span>
                  <span className={styles.permissionCount}>{accountPermissions.length}</span>
                </div>
                {accountPermissions.length > 0 ? (
                  <ul className={styles.permissionList}>
                    {accountPermissions.map((permission) => (
                      <li key={permission} className={styles.permissionPill}>
                        {permission}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className={styles.emptyState}>No permissions returned for this session.</p>
                )}
              </section>
            </div>

            <div className={styles.footer}>
              <button type="button" className={styles.secondaryButton} onClick={onClose}>
                Close
              </button>
              <button type="button" className={styles.logoutButton} onClick={onLogout}>
                Logout
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}

export default LayoutAccountControl
