export interface ModelLiveVerification {
  verified: true
  model: string
  providerModel: string
  backend: string
  provider: string
  summary: string
  verifiedAt: string
  latencyMs: number
}

export class ModelVerificationApiError extends Error {
  readonly code: string
  readonly status: number

  constructor(message: string, status: number, code = 'verification_failed') {
    super(message)
    this.name = 'ModelVerificationApiError'
    this.code = code
    this.status = status
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function isModelLiveVerification(value: unknown): value is ModelLiveVerification {
  if (!isRecord(value)) return false
  return (
    value.verified === true &&
    isNonEmptyString(value.model) &&
    isNonEmptyString(value.providerModel) &&
    isNonEmptyString(value.backend) &&
    isNonEmptyString(value.provider) &&
    isNonEmptyString(value.summary) &&
    isNonEmptyString(value.verifiedAt) &&
    !Number.isNaN(Date.parse(value.verifiedAt)) &&
    Number.isInteger(value.latencyMs) &&
    Number(value.latencyMs) >= 0
  )
}

async function readModelVerificationError(response: Response): Promise<ModelVerificationApiError> {
  const fallback = `Live model verification failed (HTTP ${response.status}).`
  try {
    const body: unknown = await response.json()
    if (!isRecord(body)) return new ModelVerificationApiError(fallback, response.status)
    const message = isNonEmptyString(body.message) ? body.message.trim() : fallback
    const code = isNonEmptyString(body.error) ? body.error.trim() : 'verification_failed'
    return new ModelVerificationApiError(message, response.status, code)
  } catch {
    return new ModelVerificationApiError(fallback, response.status)
  }
}

export async function verifyProviderModel(
  model: string,
  signal?: AbortSignal,
): Promise<ModelLiveVerification> {
  const response = await fetch('/api/models/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
    signal,
  })
  if (!response.ok) throw await readModelVerificationError(response)

  const payload: unknown = await response.json()
  if (!isModelLiveVerification(payload) || payload.model !== model) {
    throw new ModelVerificationApiError(
      'Live model verification returned an invalid contract.',
      502,
      'invalid_verification_contract',
    )
  }
  return payload
}
