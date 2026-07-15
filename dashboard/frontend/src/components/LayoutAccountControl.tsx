import React, { useId } from 'react'
import { createPortal } from 'react-dom'
import useAccessibleDialog from '../hooks/useAccessibleDialog'
import {
  formatAccountRole,
  getAccountInitials,
  groupAccountPermissions,
} from './LayoutAccountControlSupport'
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
  variant?: 'header' | 'rail'
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
  variant = 'header',
}) => {
  const initials = getAccountInitials(accountName, accountEmail)
  const roleLabel = formatAccountRole(accountRole)
  const permissionGroups = groupAccountPermissions(accountPermissions)
  const permissionCount = permissionGroups.reduce(
    (count, group) => count + group.permissions.length,
    0,
  )
  const dialogId = useId()
  const dialogTitleId = `${dialogId}-title`
  const dialogDescriptionId = `${dialogId}-description`
  const isRail = variant === 'rail'

  const dialogRef = useAccessibleDialog<HTMLDivElement>({
    isOpen,
    onClose,
    lockBodyScroll: true,
  })

  return (
    <>
      <button
        type="button"
        className={`${styles.trigger} ${isOpen ? styles.triggerActive : ''} ${isRail ? styles.triggerRail : ''}`}
        aria-controls={isOpen ? dialogId : undefined}
        aria-expanded={isOpen}
        aria-haspopup="dialog"
        aria-label={`${isOpen ? 'Close' : 'Open'} account menu for ${accountName}`}
        data-testid={isRail ? 'playground-account-control' : undefined}
        onClick={onToggle}
        onKeyDown={(event) => {
          if (event.key === 'ArrowDown' && !isOpen) {
            event.preventDefault()
            onToggle()
          } else if (event.key === 'Escape' && isOpen) {
            event.preventDefault()
            onClose()
          }
        }}
        title={isRail ? accountName : 'Account'}
      >
        <span
          className={`${styles.triggerAvatar} ${isRail ? styles.triggerAvatarRail : ''}`}
          aria-hidden="true"
        >
          {initials}
        </span>
        {isRail ? null : <span className={styles.triggerName}>{accountName}</span>}
        {isRail ? null : (
          <svg
            className={`${styles.triggerChevron} ${isOpen ? styles.triggerChevronOpen : ''}`}
            viewBox="0 0 12 12"
            aria-hidden="true"
          >
            <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        )}
      </button>

      {isOpen && typeof document !== 'undefined'
        ? createPortal(
            <div
              className={styles.overlay}
              role="presentation"
              data-testid="layout-account-overlay"
              onMouseDown={onClose}
            >
              <div
                ref={dialogRef}
                id={dialogId}
                className={`${styles.dialog} ${isRail ? styles.dialogRail : styles.dialogHeader}`}
                role="dialog"
                aria-modal="true"
                aria-labelledby={dialogTitleId}
                aria-describedby={dialogDescriptionId}
                tabIndex={-1}
                data-testid="layout-account-dialog"
                onMouseDown={(event) => event.stopPropagation()}
              >
                <div className={styles.header}>
                  <div className={styles.identityAvatar} aria-hidden="true">
                    {initials}
                  </div>
                  <div className={styles.headerCopy}>
                    <span className={styles.eyebrow}>Signed in</span>
                    <h2 id={dialogTitleId} className={styles.title}>
                      {accountName}
                    </h2>
                    <p id={dialogDescriptionId} className={styles.email}>
                      {accountEmail}
                    </p>
                  </div>
                  <button
                    type="button"
                    className={styles.closeButton}
                    aria-label="Close account dialog"
                    onClick={onClose}
                    data-dialog-initial-focus
                  >
                    ×
                  </button>
                </div>

                <div className={styles.body}>
                  <section className={styles.permissionsSection}>
                    <div className={styles.sectionHeading}>
                      <span>Access</span>
                      <span className={styles.roleBadge}>{roleLabel}</span>
                    </div>
                    <div className={styles.permissionSummary}>
                      <span>Session permissions</span>
                      <span className={styles.permissionCount}>{permissionCount}</span>
                    </div>
                    {permissionGroups.length > 0 ? (
                      <div className={styles.permissionGroups}>
                        {permissionGroups.map((group) => (
                          <div key={group.key} className={styles.permissionGroup}>
                            <h3>{group.label}</h3>
                            <ul className={styles.permissionList}>
                              {group.permissions.map((permission) => (
                                <li key={permission} className={styles.permissionPill}>
                                  {permission}
                                </li>
                              ))}
                            </ul>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className={styles.emptyState}>
                        No explicit permissions returned for this session.
                      </p>
                    )}
                  </section>
                </div>

                <div className={styles.footer}>
                  <button type="button" className={styles.logoutButton} onClick={onLogout}>
                    Logout
                  </button>
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}
    </>
  )
}

export default LayoutAccountControl
