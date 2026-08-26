type IdempotencyKeyFactory = () => string

interface IssuanceIntent {
  idempotencyKey: string
  keyId: string
}

/**
 * Keeps one idempotency identity until the Router has returned a usable
 * delegated credential. A transport failure can happen after commit, so a
 * retry must replay the same operation instead of consuming another session.
 */
export class DelegatedInferenceIssuanceIntents {
  private intent: IssuanceIntent | null = null

  constructor(private readonly createIdempotencyKey: IdempotencyKeyFactory) {}

  keyFor(keyId: string): string {
    if (!keyId) throw new Error('A delegated inference key is required.')
    if (this.intent?.keyId !== keyId) {
      this.intent = { keyId, idempotencyKey: this.createIdempotencyKey() }
    }
    return this.intent.idempotencyKey
  }

  complete(keyId: string, idempotencyKey: string): void {
    if (this.intent?.keyId === keyId && this.intent.idempotencyKey === idempotencyKey) {
      this.intent = null
    }
  }

  reset(): void {
    this.intent = null
  }
}
