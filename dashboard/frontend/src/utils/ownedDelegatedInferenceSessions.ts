export interface DelegatedInferenceSessionClaim {
  generation: number
  keyId: string
}

interface OwnedDelegatedInferenceSession {
  keyId: string
  resourceId: string
}

type RevokeDelegatedInferenceSession = (resourceId: string) => void

/**
 * Tracks only delegated sessions issued through one browser hook instance.
 * Generation claims prevent an issuance that resolves after a key switch,
 * unmount, or React StrictMode replay from becoming the active credential.
 */
export class OwnedDelegatedInferenceSessions {
  private active = false
  private activeKeyId = ''
  private generation = 0
  private readonly owned = new Map<string, OwnedDelegatedInferenceSession>()

  constructor(private readonly revoke: RevokeDelegatedInferenceSession) {}

  activate(keyId: string): void {
    this.retireAll()
    this.generation += 1
    this.active = true
    this.activeKeyId = keyId
  }

  deactivate(): void {
    this.generation += 1
    this.active = false
    this.activeKeyId = ''
    this.retireAll()
  }

  begin(keyId: string): DelegatedInferenceSessionClaim | null {
    if (!this.active || !keyId || keyId !== this.activeKeyId) return null
    return { generation: this.generation, keyId }
  }

  claim(claim: DelegatedInferenceSessionClaim, session: OwnedDelegatedInferenceSession): boolean {
    if (
      !this.active ||
      claim.generation !== this.generation ||
      claim.keyId !== this.activeKeyId ||
      session.keyId !== claim.keyId
    ) {
      this.revoke(session.resourceId)
      return false
    }
    this.owned.set(session.resourceId, {
      keyId: session.keyId,
      resourceId: session.resourceId,
    })
    return true
  }

  retire(resourceId: string): void {
    if (!this.owned.delete(resourceId)) return
    this.revoke(resourceId)
  }

  private retireAll(): void {
    const resourceIds = [...this.owned.keys()]
    this.owned.clear()
    resourceIds.forEach((resourceId) => this.revoke(resourceId))
  }
}
