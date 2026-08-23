import type { CreatedAccessAPIKey } from './inferenceAccessApi'

export interface InvitationOnboarding {
  displayName: string
  onboardingKey: CreatedAccessAPIKey
}

let pendingOnboarding: InvitationOnboarding | null = null

function isValid(
  value: InvitationOnboarding | null,
  userId?: string,
): value is InvitationOnboarding {
  if (
    !value ||
    typeof value.displayName !== 'string' ||
    !value.displayName.trim() ||
    !value.onboardingKey ||
    typeof value.onboardingKey.id !== 'string' ||
    !value.onboardingKey.id ||
    typeof value.onboardingKey.secret !== 'string' ||
    !value.onboardingKey.secret ||
    (userId !== undefined && value.onboardingKey.ownerId !== userId) ||
    (value.onboardingKey.deliveryExpiresAt !== undefined &&
      Date.parse(value.onboardingKey.deliveryExpiresAt) <= Date.now())
  ) {
    return false
  }
  return true
}

export function stageInvitationOnboarding(value: InvitationOnboarding): void {
  if (!isValid(value)) throw new TypeError('Invitation onboarding is incomplete or expired.')
  pendingOnboarding = value
}

export function peekInvitationOnboarding(userId?: string): InvitationOnboarding | null {
  if (!isValid(pendingOnboarding, userId)) {
    pendingOnboarding = null
    return null
  }
  return pendingOnboarding
}

export function claimInvitationOnboarding(userId?: string): InvitationOnboarding | null {
  const value = peekInvitationOnboarding(userId)
  pendingOnboarding = null
  return value
}

export function clearInvitationOnboarding(): void {
  pendingOnboarding = null
}
