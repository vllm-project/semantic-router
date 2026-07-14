import React, { type FormEvent, useReducer, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import AccountSecurityForm from './AccountSecurityForm'
import AccountSecurityIdentityUnavailable from './AccountSecurityIdentityUnavailable'
import {
  accountSecurityFormReducer,
  createAccountSecurityFormState,
  hasAccountSecurityIdentity,
  type AccountSecurityField,
} from './accountSecuritySupport'
import styles from './AccountSecurityPage.module.css'

const AccountSecurityPage: React.FC = () => {
  const { user, changePassword, refreshSession } = useAuth()
  const [state, dispatch] = useReducer(
    accountSecurityFormReducer,
    undefined,
    createAccountSecurityFormState,
  )
  const formRef = useRef<HTMLFormElement>(null)
  const submittingRef = useRef(false)
  const accountEmail = user?.email?.trim() ?? ''
  const hasAccountIdentity = hasAccountSecurityIdentity(accountEmail)

  const handleFieldChange = (field: AccountSecurityField, value: string) => {
    dispatch({ type: 'fieldChanged', field, value })
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!hasAccountIdentity) {
      return
    }

    if (submittingRef.current) {
      return
    }

    submittingRef.current = true
    dispatch({ type: 'submissionStarted' })
    try {
      await changePassword(state.fields.currentPassword, state.fields.newPassword)
      formRef.current?.reset()
      dispatch({ type: 'submissionSucceeded' })
    } catch (error) {
      dispatch({
        type: 'submissionFailed',
        error: error instanceof Error ? error.message : 'Unable to change password. Try again.',
        clearCurrentPassword: true,
      })
    } finally {
      submittingRef.current = false
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.hero}>
        <p className={styles.eyebrow}>Account / Security</p>
        <h1>Password &amp; security</h1>
        <p>
          Change the password for your current dashboard account and rotate the authenticated
          session without exposing credentials to another origin.
        </p>
      </header>

      <section className={styles.securityCard} aria-labelledby="password-card-title">
        <div className={styles.cardHeader}>
          <div>
            <p className={styles.cardEyebrow}>Credential lifecycle</p>
            <h2 id="password-card-title">Change password</h2>
          </div>
          <span className={styles.sessionBadge}>Encrypted transport required</span>
        </div>

        {!hasAccountIdentity ? (
          <AccountSecurityIdentityUnavailable onRetry={() => void refreshSession()} />
        ) : state.status === 'complete' ? (
          <div className={styles.successPanel} role="status" aria-live="polite" tabIndex={-1}>
            <span className={styles.successIcon} aria-hidden="true">
              ✓
            </span>
            <div>
              <h3>Password changed</h3>
              <p>
                This browser now uses the rotated session. Other dashboard sessions have been signed
                out, and the password fields were cleared.
              </p>
              <button
                className={styles.secondaryButton}
                type="button"
                onClick={() => dispatch({ type: 'restart' })}
              >
                Change it again
              </button>
            </div>
          </div>
        ) : (
          <AccountSecurityForm
            accountEmail={accountEmail}
            fields={state.fields}
            error={state.error}
            pending={state.status === 'submitting'}
            formRef={formRef}
            onFieldChange={handleFieldChange}
            onSubmit={(event) => void handleSubmit(event)}
          />
        )}
      </section>

      <aside className={styles.securityNote} aria-labelledby="password-manager-heading">
        <h2 id="password-manager-heading">Password manager compatibility</h2>
        <p>
          The account, current-password, and new-password fields use standard browser metadata so
          Chrome and other password managers can update the saved credential after a successful
          change.
        </p>
      </aside>
    </main>
  )
}

export default AccountSecurityPage
