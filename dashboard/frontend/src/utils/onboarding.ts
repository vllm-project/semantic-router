export const ONBOARDING_STATUS_KEY = 'vllm-sr.onboarding.status'
export const ONBOARDING_STEP_KEY = 'vllm-sr.onboarding.step'

export type OnboardingStatus = 'idle' | 'pending' | 'dismissed' | 'completed'

export function getOnboardingStatus(): OnboardingStatus {
  if (typeof window === 'undefined') {
    return 'idle'
  }

  const stored = window.localStorage.getItem(ONBOARDING_STATUS_KEY)
  if (stored === 'pending' || stored === 'dismissed' || stored === 'completed') {
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
    window.localStorage.removeItem(ONBOARDING_STEP_KEY)
    return
  }

  window.localStorage.setItem(ONBOARDING_STATUS_KEY, status)
}

export function getOnboardingStep(maxSteps?: number): number {
  if (typeof window === 'undefined') return 0
  const parsed = Number.parseInt(window.localStorage.getItem(ONBOARDING_STEP_KEY) || '0', 10)
  if (!Number.isFinite(parsed) || parsed < 0) return 0
  return maxSteps === undefined ? parsed : Math.min(parsed, Math.max(0, maxSteps - 1))
}

export function setOnboardingStep(step: number): void {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(ONBOARDING_STEP_KEY, String(Math.max(0, Math.floor(step))))
}

export function clearOnboardingStep(): void {
  if (typeof window === 'undefined') return
  window.localStorage.removeItem(ONBOARDING_STEP_KEY)
}

export function markOnboardingPending(): void {
  setOnboardingStep(0)
  setOnboardingStatus('pending')
}
