import type { AccessAPIKey, AccessPage, CreatedAccessAPIKey } from './inferenceAccessApi'

const FIRST_API_KEY_PREFIX = 'vllm-sr.invitation.first-api-key.'
const createdKeyHandoffSymbol = Symbol.for('vllm-sr.created-api-key-handoff')

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

/**
 * Carries the one-time secret across the lazy route boundary without writing
 * it to browser history or persistent storage.
 */
export function handoffFirstAPIKey(key: CreatedAccessAPIKey): void {
  Reflect.set(globalThis, createdKeyHandoffSymbol, key)
}

export function takeFirstAPIKeyHandoff(): CreatedAccessAPIKey | null {
  const key = Reflect.get(globalThis, createdKeyHandoffSymbol) as CreatedAccessAPIKey | undefined
  Reflect.deleteProperty(globalThis, createdKeyHandoffSymbol)
  return key ?? null
}

interface FirstAPIKeyProvisioner {
  list: () => Promise<AccessPage<AccessAPIKey>>
  create: (name: string) => Promise<CreatedAccessAPIKey>
}

/**
 * Resolve or create the invitation user's one self-service key. The backend
 * serializes creation per user; the second read turns a cross-replica race
 * into an idempotent success for the browser that lost it.
 */
export async function ensureFirstAPIKey(
  displayName: string,
  provisioner: FirstAPIKeyProvisioner,
): Promise<AccessAPIKey | CreatedAccessAPIKey> {
  const existing = await provisioner.list()
  if (existing.items[0]) return existing.items[0]

  try {
    return await provisioner.create(displayName.trim() || 'My API key')
  } catch (createError) {
    const raced = await provisioner.list()
    if (raced.items[0]) return raced.items[0]
    throw createError
  }
}
