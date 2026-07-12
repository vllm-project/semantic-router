import { describe, expect, it } from 'vitest'
import {
  accountSecurityFormReducer,
  createAccountSecurityFormState,
  hasAccountSecurityIdentity,
  passwordFieldType,
} from './accountSecuritySupport'

describe('account security form state', () => {
  it('requires a non-empty account identity before exposing password entry', () => {
    expect(hasAccountSecurityIdentity('')).toBe(false)
    expect(hasAccountSecurityIdentity('   ')).toBe(false)
    expect(hasAccountSecurityIdentity('user@example.test')).toBe(true)
  })

  it('keeps password fields hidden by default and only reveals them explicitly', () => {
    expect(passwordFieldType(false)).toBe('password')
    expect(passwordFieldType(true)).toBe('text')
  })

  it('clears every sensitive field when password rotation succeeds', () => {
    const populated = {
      fields: {
        currentPassword: 'current-value',
        newPassword: 'new-value',
      },
      status: 'submitting' as const,
      error: null,
    }

    expect(accountSecurityFormReducer(populated, { type: 'submissionSucceeded' })).toEqual({
      fields: {
        currentPassword: '',
        newPassword: '',
      },
      status: 'complete',
      error: null,
    })
  })

  it('clears the current password after a server failure while retaining the policy error', () => {
    const populated = {
      ...createAccountSecurityFormState(),
      fields: {
        currentPassword: 'current-value',
        newPassword: 'new-value',
      },
      status: 'submitting' as const,
    }

    expect(
      accountSecurityFormReducer(populated, {
        type: 'submissionFailed',
        error: 'Password must contain a symbol.',
        clearCurrentPassword: true,
      }),
    ).toEqual({
      fields: {
        currentPassword: '',
        newPassword: 'new-value',
      },
      status: 'editing',
      error: 'Password must contain a symbol.',
    })
  })
})
