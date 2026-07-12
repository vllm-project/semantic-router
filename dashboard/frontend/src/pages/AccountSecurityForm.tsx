import React, { type FormEvent, type RefObject, useState } from 'react'
import {
  passwordFieldType,
  type AccountSecurityField,
  type AccountSecurityFields,
} from './accountSecuritySupport'
import styles from './AccountSecurityPage.module.css'

interface AccountSecurityFormProps {
  accountEmail: string
  fields: AccountSecurityFields
  error: string | null
  pending: boolean
  formRef: RefObject<HTMLFormElement>
  onFieldChange: (field: AccountSecurityField, value: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}

const AccountSecurityForm: React.FC<AccountSecurityFormProps> = ({
  accountEmail,
  fields,
  error,
  pending,
  formRef,
  onFieldChange,
  onSubmit,
}) => {
  const [passwordsVisible, setPasswordsVisible] = useState(false)
  const inputType = passwordFieldType(passwordsVisible)

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    setPasswordsVisible(false)
    onSubmit(event)
  }

  return (
    <form
      ref={formRef}
      id="change-password-form"
      name="change-password"
      className={styles.form}
      action="/api/auth/password"
      method="post"
      autoComplete="on"
      onSubmit={handleSubmit}
      data-testid="account-security-form"
    >
      <div className={styles.fieldGroup}>
        <label htmlFor="username">Account</label>
        <input
          id="username"
          className={styles.input}
          type="email"
          name="username"
          autoComplete="username"
          value={accountEmail}
          readOnly
          spellCheck={false}
          autoCapitalize="none"
        />
        <p className={styles.fieldHint}>The account whose saved password should be updated.</p>
      </div>

      <div className={styles.passwordVisibilityRow}>
        <p>Only reveal passwords when your screen cannot be observed.</p>
        <button
          className={styles.visibilityButton}
          type="button"
          aria-controls="current-password new-password"
          aria-pressed={passwordsVisible}
          onClick={() => setPasswordsVisible((visible) => !visible)}
          disabled={pending}
        >
          {passwordsVisible ? 'Hide passwords' : 'Show passwords'}
        </button>
      </div>

      <div className={styles.fieldGroup}>
        <label htmlFor="current-password">Current password</label>
        <input
          id="current-password"
          className={styles.input}
          type={inputType}
          name="current-password"
          autoComplete="current-password"
          value={fields.currentPassword}
          onChange={(event) => onFieldChange('currentPassword', event.target.value)}
          disabled={pending}
          required
        />
      </div>

      <div className={styles.fieldGroup}>
        <label htmlFor="new-password">New password</label>
        <input
          id="new-password"
          className={styles.input}
          type={inputType}
          name="new-password"
          autoComplete="new-password"
          aria-describedby="account-security-password-guidance"
          value={fields.newPassword}
          onChange={(event) => onFieldChange('newPassword', event.target.value)}
          disabled={pending}
          required
        />
        <p id="account-security-password-guidance" className={styles.fieldHint}>
          Use a unique password. Your password manager can generate and save one that meets the
          server policy.
        </p>
      </div>

      {error ? (
        <div className={styles.errorNotice} role="alert" aria-live="assertive">
          {error}
        </div>
      ) : null}

      <div className={styles.formFooter}>
        <p>Changing your password signs out every other dashboard session.</p>
        <button className={styles.primaryButton} type="submit" disabled={pending}>
          {pending ? 'Changing password…' : 'Change password'}
        </button>
      </div>
    </form>
  )
}

export default AccountSecurityForm
