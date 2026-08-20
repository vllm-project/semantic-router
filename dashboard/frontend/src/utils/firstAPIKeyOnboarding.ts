const FIRST_API_KEY_PREFIX = 'vllm-sr.invitation.first-api-key.'

const storageKey = (userId: string) => `${FIRST_API_KEY_PREFIX}${userId.trim()}`

export function markFirstAPIKeyOnboardingPending(userId: string): void {
  if (!userId.trim()) return
  try {
    window.localStorage.setItem(storageKey(userId), 'pending')
  } catch {
    // Storage can be unavailable in privacy-restricted browsers.
  }
}

export function markFirstAPIKeyOnboardingHandled(userId: string): void {
  if (!userId.trim()) return
  try {
    window.localStorage.setItem(storageKey(userId), 'handled')
  } catch {
    // The prompt still closes for the current session.
  }
}

export function isFirstAPIKeyOnboardingPending(userId: string): boolean {
  if (!userId.trim()) return false
  try {
    return window.localStorage.getItem(storageKey(userId)) === 'pending'
  } catch {
    return false
  }
}
