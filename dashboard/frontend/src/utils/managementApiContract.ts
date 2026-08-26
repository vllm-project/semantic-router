import {
  MANAGEMENT_API_BASE_PATH,
  MANAGEMENT_API_HEADERS,
  MANAGEMENT_API_MEDIA_TYPE,
  MANAGEMENT_API_OPERATIONS,
  assertManagementApiSchema,
  assertManagementApiOperationResponse,
  createManagementApiClient,
  managementApiPath,
  type ManagementApiClientOperationId,
  type ManagementApiClientResponse,
  type ManagementApiRequestOptions as GeneratedManagementApiRequestOptions,
  type ManagementApiResponse,
  type ManagementApiTransport,
  type ManagementApiOperationId,
  type ManagementApiPathParameters,
} from '../generated/managementApiContract'

import type { ManagementMe } from './routerManagementTypes'

const DASHBOARD_ROUTER_PROXY_PATH = '/api/router'

let selectedNamespace = ''

// Namespace selection is browser-session state, not persisted authority. The
// Router remains the source of truth and validates the header on every call.
export function setManagementNamespace(namespaceId: string | null | undefined): void {
  selectedNamespace = typeof namespaceId === 'string' ? namespaceId.trim() : ''
}

export function getManagementNamespace(): string {
  return selectedNamespace
}

export class ManagementApiError extends Error {
  readonly status: number
  readonly code?: string
  readonly requestId?: string
  readonly retryAfterMilliseconds?: number

  constructor(
    message: string,
    status: number,
    code?: string,
    requestId?: string,
    retryAfterMilliseconds?: number,
  ) {
    super(message)
    this.name = 'ManagementApiError'
    this.status = status
    this.code = code
    this.requestId = requestId
    this.retryAfterMilliseconds = retryAfterMilliseconds
  }
}

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

export const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.length > 0 && value.trim() === value

export const isOptionalString = (value: unknown): value is string | undefined =>
  value === undefined || isNonEmptyString(value)

export const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every(isNonEmptyString)

export function hasOnlyKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = [],
): boolean {
  const allowed = new Set([...required, ...optional])
  return (
    required.every((key) => Object.prototype.hasOwnProperty.call(value, key)) &&
    Object.keys(value).every((key) => allowed.has(key))
  )
}

export interface ManagementRequestOptions {
  body?: unknown
  headers?: Record<string, string>
  signal?: AbortSignal
  query?: URLSearchParams | Readonly<Record<string, string | number | boolean | null | undefined>>
  /** null deliberately omits namespace selection for cluster/session endpoints. */
  namespace?: string | null
}

type GeneratedManagementQuery<OperationId extends ManagementApiClientOperationId> =
  GeneratedManagementApiRequestOptions<OperationId> extends { query?: infer Query } ? Query : never

export type ManagementOperationRequestOptions<OperationId extends ManagementApiClientOperationId> =
  Omit<GeneratedManagementApiRequestOptions<OperationId>, 'query'> &
    ([GeneratedManagementQuery<OperationId>] extends [never]
      ? { query?: never }
      : { query?: GeneratedManagementQuery<OperationId> | URLSearchParams }) &
    ([ManagementApiPathParameters<OperationId>] extends [never]
      ? { pathParameters?: never }
      : { pathParameters: ManagementApiPathParameters<OperationId> })

type ManagementOperationRequestArguments<OperationId extends ManagementApiClientOperationId> = [
  ManagementApiPathParameters<OperationId>,
] extends [never]
  ? [options?: ManagementOperationRequestOptions<OperationId>]
  : [options: ManagementOperationRequestOptions<OperationId>]

function parseRetryAfter(value: string | null): number | undefined {
  if (!value) return undefined
  const seconds = Number(value)
  if (Number.isFinite(seconds) && seconds >= 0) return Math.round(seconds * 1_000)
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return undefined
  return Math.max(0, timestamp - Date.now())
}

function managementQueryString(query: ManagementRequestOptions['query']): string {
  if (!query) return ''
  if (query instanceof URLSearchParams) return query.toString()
  const parameters = new URLSearchParams()
  for (const [name, value] of Object.entries(query)) {
    if (value !== undefined && value !== null) parameters.set(name, String(value))
  }
  return parameters.toString()
}

function managementCustomHeaders(
  headers: Record<string, string> | undefined,
): Record<string, string> {
  const reserved = new Set([
    'accept',
    'content-type',
    MANAGEMENT_API_HEADERS.namespace.toLocaleLowerCase(),
  ])
  return Object.fromEntries(
    Object.entries(headers ?? {}).filter(([name]) => !reserved.has(name.toLocaleLowerCase())),
  )
}

function managementError(
  payload: unknown,
  status: number,
  fallback: string,
  retryAfterMilliseconds?: number,
): ManagementApiError {
  if (isRecord(payload) && isRecord(payload.error)) {
    const error = payload.error
    if (isNonEmptyString(error.message)) {
      return new ManagementApiError(
        error.message,
        status,
        isNonEmptyString(error.code) ? error.code : undefined,
        isNonEmptyString(error.requestId) ? error.requestId : undefined,
        retryAfterMilliseconds,
      )
    }
  }
  return new ManagementApiError(fallback, status, undefined, undefined, retryAfterMilliseconds)
}

export type ManagementOperationResponse<T = unknown> = ManagementApiClientResponse<T>

async function managementRequestWithMetadata(
  operationId: ManagementApiClientOperationId,
  options: ManagementRequestOptions & { pathParameters?: Record<string, string | number> },
): Promise<ManagementOperationResponse> {
  const buildPath = managementApiPath as (
    id: ManagementApiOperationId,
    parameters?: Record<string, string | number>,
  ) => string
  const path = buildPath(operationId, options.pathParameters)
  if (!path.startsWith(`${MANAGEMENT_API_BASE_PATH}/`)) {
    throw new TypeError('Management API requests require a generated operation path.')
  }
  const operation = MANAGEMENT_API_OPERATIONS[operationId]
  const successMediaTypes = operation.successMediaTypes as readonly string[]
  const acceptMediaType =
    operation.responseMode === 'yaml'
      ? (successMediaTypes[0] ?? MANAGEMENT_API_MEDIA_TYPE)
      : MANAGEMENT_API_MEDIA_TYPE
  const namespace = options.namespace === undefined ? selectedNamespace : (options.namespace ?? '')
  const query = managementQueryString(options.query)
  const response = await fetch(`${DASHBOARD_ROUTER_PROXY_PATH}${path}${query ? `?${query}` : ''}`, {
    method: operation.method,
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {
      ...managementCustomHeaders(options.headers),
      Accept: acceptMediaType,
      ...(options.body === undefined ? {} : { 'Content-Type': MANAGEMENT_API_MEDIA_TYPE }),
      ...(namespace ? { [MANAGEMENT_API_HEADERS.namespace]: namespace } : {}),
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  })
  const successResponses = operation.successResponses as Readonly<
    Record<number, readonly string[] | undefined>
  >
  const responseContract = successResponses[response.status]
  const hasBody = responseContract !== undefined && responseContract.length > 0
  const responseMediaType = response.headers.get('Content-Type')?.split(';', 1)[0]?.trim()
  if (!response.ok) {
    const errorPayload: unknown =
      responseMediaType === MANAGEMENT_API_MEDIA_TYPE
        ? await response.json().catch(() => null)
        : await response.text().catch(() => null)
    throw managementError(
      errorPayload,
      response.status,
      `Request failed (HTTP ${response.status}).`,
      parseRetryAfter(response.headers.get('Retry-After')),
    )
  }
  if (hasBody && (!responseMediaType || !successMediaTypes.includes(responseMediaType))) {
    throw new ManagementApiError('Management API returned an unsupported media type.', 502)
  }
  const payload: unknown = !hasBody
    ? undefined
    : operation.responseMode === 'yaml'
      ? await response.text()
      : await response.json().catch(() => null)
  const etag = response.headers.get(MANAGEMENT_API_HEADERS.etag)?.trim()
  const requestId = response.headers.get(MANAGEMENT_API_HEADERS.requestId)?.trim()
  const secretResultClaim = response.headers.get(MANAGEMENT_API_HEADERS.secretResultClaim)?.trim()
  const idempotencyReplayed =
    response.headers.get(MANAGEMENT_API_HEADERS.idempotencyReplayed)?.trim().toLowerCase() ===
    'true'
  return {
    data: payload,
    status: response.status,
    ...(responseMediaType ? { mediaType: responseMediaType } : {}),
    ...(etag ? { etag } : {}),
    ...(requestId ? { requestId } : {}),
    ...(secretResultClaim ? { secretResultClaim } : {}),
    ...(idempotencyReplayed ? { idempotencyReplayed: true } : {}),
  }
}

/**
 * Open a generated Management operation without buffering its response body.
 *
 * This is intentionally narrower than a general-purpose fetch escape hatch: callers
 * must still name an operation from the generated contract, and the Dashboard keeps
 * namespace/session transport policy in one place. It is used by resumable event
 * streams whose response cannot be decoded as one JSON document.
 */
export async function managementOperationStream<OperationId extends ManagementApiClientOperationId>(
  operationId: OperationId,
  ...args: ManagementOperationRequestArguments<OperationId>
): Promise<Response> {
  const options = (args[0] ?? {}) as ManagementRequestOptions & {
    pathParameters?: Record<string, string | number>
  }
  const buildPath = managementApiPath as (
    id: ManagementApiOperationId,
    parameters?: Record<string, string | number>,
  ) => string
  const path = buildPath(operationId, options.pathParameters)
  const contract = MANAGEMENT_API_OPERATIONS[operationId]
  const eventStreamMediaType = (contract.successMediaTypes as readonly string[]).find(
    (mediaType) => mediaType === 'text/event-stream',
  )
  if (
    contract.method !== 'GET' ||
    contract.responseMode !== 'json_or_event_stream' ||
    !eventStreamMediaType
  ) {
    throw new TypeError('Management event streams require a generated streaming operation.')
  }

  const namespace = options.namespace === undefined ? selectedNamespace : (options.namespace ?? '')
  const query = managementQueryString(options.query)
  const response = await fetch(`${DASHBOARD_ROUTER_PROXY_PATH}${path}${query ? `?${query}` : ''}`, {
    method: contract.method,
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {
      ...managementCustomHeaders(options.headers),
      Accept: eventStreamMediaType,
      ...(namespace ? { [MANAGEMENT_API_HEADERS.namespace]: namespace } : {}),
    },
    signal: options.signal,
  })
  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null)
    throw managementError(
      payload,
      response.status,
      `Event stream failed (HTTP ${response.status}).`,
      parseRetryAfter(response.headers.get('Retry-After')),
    )
  }
  const responseMediaType = response.headers.get('Content-Type')?.split(';', 1)[0]?.trim()
  if (responseMediaType !== eventStreamMediaType) {
    throw new ManagementApiError('Management API returned an unsupported event stream.', 502)
  }
  if (!response.body) {
    throw new ManagementApiError('Management API returned an empty event stream.', 502)
  }
  return response
}

export async function managementOperationRequest<
  OperationId extends ManagementApiClientOperationId,
>(
  operationId: OperationId,
  ...args: ManagementOperationRequestArguments<OperationId>
): Promise<ManagementApiResponse<OperationId>> {
  const options = (args[0] ?? {}) as ManagementRequestOptions & {
    pathParameters?: Record<string, string | number>
  }
  const response = await managementRequestWithMetadata(operationId, options)
  return validateManagementOperationResponse(
    operationId,
    response.data,
    response.status,
    response.mediaType,
  )
}

/** Read an operation payload together with the Router's opaque revision token. */
export async function managementOperationRequestWithMetadata<
  OperationId extends ManagementApiClientOperationId,
>(
  operationId: OperationId,
  ...args: ManagementOperationRequestArguments<OperationId>
): Promise<ManagementOperationResponse<ManagementApiResponse<OperationId>>> {
  const options = (args[0] ?? {}) as ManagementRequestOptions & {
    pathParameters?: Record<string, string | number>
  }
  const response = await managementRequestWithMetadata(operationId, options)
  return {
    ...response,
    data: validateManagementOperationResponse(
      operationId,
      response.data,
      response.status,
      response.mediaType,
    ),
  }
}

/** Browser transport for the generated Management client. */
export const managementApiTransport: ManagementApiTransport = {
  async request(operationId, options) {
    const requestOptions = options as ManagementRequestOptions & {
      pathParameters?: Record<string, string | number>
    }
    return managementRequestWithMetadata(operationId, requestOptions)
  },
}

export const managementApiClient = createManagementApiClient(managementApiTransport)

function validateManagementOperationResponse<OperationId extends ManagementApiClientOperationId>(
  operationId: OperationId,
  value: unknown,
  status: number,
  mediaType?: string,
): ManagementApiResponse<OperationId> {
  try {
    return assertManagementApiOperationResponse(operationId, value, status, mediaType)
  } catch (cause) {
    throw new ManagementApiError(
      cause instanceof Error ? cause.message : 'Router returned an invalid Management response.',
      502,
    )
  }
}

/** Validate the security-sensitive identity projection before it drives navigation or ownership. */
export function assertManagementMe(value: unknown): ManagementMe {
  return assertManagementApiSchema('Me', value)
}
