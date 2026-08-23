import {
  MANAGEMENT_API_BASE_PATH,
  MANAGEMENT_API_HEADERS,
  MANAGEMENT_API_MEDIA_TYPE,
  MANAGEMENT_API_OPERATIONS,
  managementApiPath,
  type ManagementApiAgentTransport,
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

export type ManagementOperationRequestOptions<OperationId extends ManagementApiOperationId> =
  ManagementRequestOptions &
    ([ManagementApiPathParameters<OperationId>] extends [never]
      ? { pathParameters?: never }
      : { pathParameters: ManagementApiPathParameters<OperationId> })

type ManagementOperationRequestArguments<OperationId extends ManagementApiOperationId> = [
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

export interface ManagementOperationResponse<T = unknown> {
  data: T
  etag?: string
}

async function managementRequestWithMetadata(
  path: string,
  method: (typeof MANAGEMENT_API_OPERATIONS)[ManagementApiOperationId]['method'],
  options: ManagementRequestOptions,
): Promise<ManagementOperationResponse> {
  if (!path.startsWith(`${MANAGEMENT_API_BASE_PATH}/`)) {
    throw new TypeError('Management API requests require a generated operation path.')
  }
  const namespace = options.namespace === undefined ? selectedNamespace : (options.namespace ?? '')
  const query = managementQueryString(options.query)
  const response = await fetch(`${DASHBOARD_ROUTER_PROXY_PATH}${path}${query ? `?${query}` : ''}`, {
    method,
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {
      Accept: MANAGEMENT_API_MEDIA_TYPE,
      ...(options.body === undefined ? {} : { 'Content-Type': MANAGEMENT_API_MEDIA_TYPE }),
      ...(namespace ? { [MANAGEMENT_API_HEADERS.namespace]: namespace } : {}),
      ...options.headers,
    },
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
    signal: options.signal,
  })
  const hasBody = response.status !== 204 && response.status !== 205
  const payload: unknown = hasBody ? await response.json().catch(() => null) : null
  if (!response.ok) {
    throw managementError(
      payload,
      response.status,
      `Request failed (HTTP ${response.status}).`,
      parseRetryAfter(response.headers.get('Retry-After')),
    )
  }
  const responseMediaType = response.headers.get('Content-Type')?.split(';', 1)[0]?.trim()
  if (hasBody && responseMediaType !== MANAGEMENT_API_MEDIA_TYPE) {
    throw new ManagementApiError('Management API returned an unsupported media type.', 502)
  }
  const etag = response.headers.get(MANAGEMENT_API_HEADERS.etag)?.trim()
  return { data: payload, ...(etag ? { etag } : {}) }
}

/**
 * Open a generated Management operation without buffering its response body.
 *
 * This is intentionally narrower than a general-purpose fetch escape hatch: callers
 * must still name an operation from the generated contract, and the Dashboard keeps
 * namespace/session transport policy in one place. It is used by resumable event
 * streams whose response cannot be decoded as one JSON document.
 */
export async function managementOperationStream<OperationId extends ManagementApiOperationId>(
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
  if (contract.method !== 'GET') {
    throw new TypeError('Management event streams require a generated GET operation.')
  }

  const namespace = options.namespace === undefined ? selectedNamespace : (options.namespace ?? '')
  const query = managementQueryString(options.query)
  const response = await fetch(`${DASHBOARD_ROUTER_PROXY_PATH}${path}${query ? `?${query}` : ''}`, {
    method: contract.method,
    cache: 'no-store',
    credentials: 'same-origin',
    headers: {
      Accept: 'text/event-stream',
      ...(namespace ? { [MANAGEMENT_API_HEADERS.namespace]: namespace } : {}),
      ...options.headers,
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
  if (responseMediaType !== 'text/event-stream') {
    throw new ManagementApiError('Management API returned an unsupported event stream.', 502)
  }
  if (!response.body) {
    throw new ManagementApiError('Management API returned an empty event stream.', 502)
  }
  return response
}

export async function managementOperationRequest<OperationId extends ManagementApiOperationId>(
  operationId: OperationId,
  ...args: ManagementOperationRequestArguments<OperationId>
): Promise<unknown> {
  const options = (args[0] ?? {}) as ManagementRequestOptions & {
    pathParameters?: Record<string, string | number>
  }
  const buildPath = managementApiPath as (
    id: ManagementApiOperationId,
    parameters?: Record<string, string | number>,
  ) => string
  const path = buildPath(operationId, options.pathParameters)
  return (
    await managementRequestWithMetadata(
      path,
      MANAGEMENT_API_OPERATIONS[operationId].method,
      options,
    )
  ).data
}

/** Read an operation payload together with the Router's opaque revision token. */
export async function managementOperationRequestWithMetadata<
  OperationId extends ManagementApiOperationId,
>(
  operationId: OperationId,
  ...args: ManagementOperationRequestArguments<OperationId>
): Promise<ManagementOperationResponse> {
  const options = (args[0] ?? {}) as ManagementRequestOptions & {
    pathParameters?: Record<string, string | number>
  }
  const buildPath = managementApiPath as (
    id: ManagementApiOperationId,
    parameters?: Record<string, string | number>,
  ) => string
  return managementRequestWithMetadata(
    buildPath(operationId, options.pathParameters),
    MANAGEMENT_API_OPERATIONS[operationId].method,
    options,
  )
}

/** Browser transport for the generated Agent client. */
export const managementApiAgentTransport: ManagementApiAgentTransport = {
  async request(operationId, options) {
    const requestOptions = options as ManagementRequestOptions & {
      pathParameters?: Record<string, string | number>
    }
    const buildPath = managementApiPath as (
      id: ManagementApiOperationId,
      parameters?: Record<string, string | number>,
    ) => string
    return managementRequestWithMetadata(
      buildPath(operationId, requestOptions.pathParameters),
      MANAGEMENT_API_OPERATIONS[operationId].method,
      requestOptions,
    )
  },
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

/** Validate the security-sensitive identity projection before it drives navigation or ownership. */
export function assertManagementMe(value: unknown): ManagementMe {
  if (!isRecord(value)) throw new Error('Router returned an invalid Management identity.')
  const principal = value.principal
  const session = value.session
  if (
    !isRecord(principal) ||
    !isNonEmptyString(principal.principalId) ||
    !isNonEmptyString(principal.displayName) ||
    !isNonEmptyString(principal.kind) ||
    !isNonEmptyString(principal.status) ||
    !isRecord(session) ||
    !isNonEmptyString(session.sessionId) ||
    !isNonEmptyString(session.authenticatedAt) ||
    !isNonEmptyString(session.expiresAt) ||
    !isNonEmptyString(session.evidenceKind) ||
    !isStringArray(value.clusterPermissions) ||
    !Array.isArray(value.namespaces)
  ) {
    throw new Error('Router returned an invalid Management identity.')
  }

  for (const scope of value.namespaces) {
    if (!isRecord(scope) || !isRecord(scope.namespace)) {
      throw new Error('Router returned an invalid Management namespace scope.')
    }
    const namespace = scope.namespace
    if (
      !isNonEmptyString(namespace.namespaceId) ||
      !isNonEmptyString(namespace.name) ||
      !isNonEmptyString(namespace.status) ||
      !isFiniteNumber(namespace.desiredRevision) ||
      !isFiniteNumber(namespace.appliedRevision) ||
      !isStringArray(scope.permissions) ||
      !Array.isArray(scope.roleBindings) ||
      !Array.isArray(scope.teams) ||
      !isRecord(scope.selfServicePolicy)
    ) {
      throw new Error('Router returned an invalid Management namespace scope.')
    }

    if (scope.user !== undefined) {
      const user = scope.user
      if (
        !isRecord(user) ||
        !isNonEmptyString(user.userId) ||
        !isNonEmptyString(user.email) ||
        !isNonEmptyString(user.displayName) ||
        !isNonEmptyString(user.status)
      ) {
        throw new Error('Router returned an invalid linked Management user.')
      }
    }

    for (const team of scope.teams) {
      if (
        !isRecord(team) ||
        !isNonEmptyString(team.teamId) ||
        !isNonEmptyString(team.name) ||
        (team.role !== 'admin' && team.role !== 'member') ||
        !isNonEmptyString(team.status)
      ) {
        throw new Error('Router returned an invalid Management team membership.')
      }
    }

    const policy = scope.selfServicePolicy
    if (
      !isFiniteNumber(policy.maxKeysPerUser) ||
      !isFiniteNumber(policy.maxDelegatedSessions) ||
      !isFiniteNumber(policy.delegatedSessionTtlSeconds) ||
      typeof policy.allowTeamKeyDelegation !== 'boolean' ||
      typeof policy.automaticFirstKey !== 'boolean' ||
      !isFiniteNumber(policy.revision)
    ) {
      throw new Error('Router returned an invalid Management self-service policy.')
    }
  }
  return value as unknown as ManagementMe
}
