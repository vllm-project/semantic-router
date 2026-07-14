import React from 'react'
import styles from './AccountSecurityPage.module.css'

interface AccountSecurityIdentityUnavailableProps {
  onRetry: () => void
}

const AccountSecurityIdentityUnavailable: React.FC<AccountSecurityIdentityUnavailableProps> = ({
  onRetry,
}) => (
  <div className={styles.identityUnavailable} role="alert">
    <h3>Account identity unavailable</h3>
    <p>
      The dashboard could not load the email address required to associate this password change with
      the correct account. No password fields are available until the session is refreshed.
    </p>
    <button className={styles.secondaryButton} type="button" onClick={onRetry}>
      Retry account check
    </button>
  </div>
)

export default AccountSecurityIdentityUnavailable
