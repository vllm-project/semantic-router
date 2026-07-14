export type AccountSecurityField = 'currentPassword' | 'newPassword'

export interface AccountSecurityFields {
  currentPassword: string
  newPassword: string
}

export interface AccountSecurityFormState {
  fields: AccountSecurityFields
  status: 'editing' | 'submitting' | 'complete'
  error: string | null
}

export type AccountSecurityFormAction =
  | { type: 'fieldChanged'; field: AccountSecurityField; value: string }
  | { type: 'submissionStarted' }
  | { type: 'submissionSucceeded' }
  | { type: 'submissionFailed'; error: string; clearCurrentPassword: boolean }
  | { type: 'restart' }

export function hasAccountSecurityIdentity(accountEmail: string): boolean {
  return accountEmail.trim().length > 0
}

export function passwordFieldType(visible: boolean): 'text' | 'password' {
  return visible ? 'text' : 'password'
}

function emptyFields(): AccountSecurityFields {
  return {
    currentPassword: '',
    newPassword: '',
  }
}

export function createAccountSecurityFormState(): AccountSecurityFormState {
  return {
    fields: emptyFields(),
    status: 'editing',
    error: null,
  }
}

export function accountSecurityFormReducer(
  state: AccountSecurityFormState,
  action: AccountSecurityFormAction,
): AccountSecurityFormState {
  switch (action.type) {
    case 'fieldChanged':
      return {
        ...state,
        fields: { ...state.fields, [action.field]: action.value },
        error: null,
      }
    case 'submissionStarted':
      return { ...state, status: 'submitting', error: null }
    case 'submissionSucceeded':
      return { fields: emptyFields(), status: 'complete', error: null }
    case 'submissionFailed':
      return {
        ...state,
        fields: action.clearCurrentPassword
          ? { ...state.fields, currentPassword: '' }
          : state.fields,
        status: 'editing',
        error: action.error,
      }
    case 'restart':
      return createAccountSecurityFormState()
  }
}
