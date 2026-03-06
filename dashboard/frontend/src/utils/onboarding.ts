export const ONBOARDING_STATUS_KEY = 'vllm-sr.onboarding.status'

export type OnboardingStatus = 'idle' | 'pending' | 'dismissed' | 'completed'

export function getOnboardingStatus(): OnboardingStatus {
  if (typeof window === 'undefined') {
    return 'idle'
  }

  const stored = window.localStorage.getItem(ONBOARDING_STATUS_KEY)
  if (
    stored === 'pending' ||
    stored === 'dismissed' ||
    stored === 'completed'
  ) {
    return stored
  }

  return 'idle'
}

export function setOnboardingStatus(status: OnboardingStatus): void {
  if (typeof window === 'undefined') {
    return
  }

  if (status === 'idle') {
    window.localStorage.removeItem(ONBOARDING_STATUS_KEY)
    return
  }

  window.localStorage.setItem(ONBOARDING_STATUS_KEY, status)
}

export function markOnboardingPending(): void {
  setOnboardingStatus('pending')
}
