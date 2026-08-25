import { managementOperationRequest, type ManagementRequestOptions } from './managementApiContract'
import {
  MANAGEMENT_API_HEADERS,
  type ManagementApiOperationId,
} from '../generated/managementApiContract'
import type { AccessListParams, AccessPage } from './inferenceAccessTypes'
import type { ManagementPage, MutationReceipt, ResourceDetail } from './routerManagementTypes'

export const idempotencyHeaders = () => ({
  [MANAGEMENT_API_HEADERS.idempotencyKey]: crypto.randomUUID(),
})

export const etag = (kind: string, revision: number) => `"${kind}:${revision}"`

export const query = (values: Record<string, string | number | undefined>) => {
  const params = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value))
  })
  return params
}

export async function request<T>(
  operationId: ManagementApiOperationId,
  options: ManagementRequestOptions & {
    pathParameters?: Record<string, string | number>
  } = {},
): Promise<T> {
  const invoke = managementOperationRequest as (
    id: ManagementApiOperationId,
    requestOptions?: ManagementRequestOptions & {
      pathParameters?: Record<string, string | number>
    },
  ) => Promise<unknown>
  return (await invoke(operationId, options)) as T
}

export const resource = <T>(payload: ResourceDetail<T>) => payload.data

export function viewPage<T, U>(
  page: ManagementPage<T>,
  map: (item: T) => U,
  clientFilter?: string,
): AccessPage<U> {
  let items = page.data.map(map)
  if (clientFilter) {
    const needle = clientFilter.toLocaleLowerCase()
    items = items.filter((item) => JSON.stringify(item).toLocaleLowerCase().includes(needle))
  }
  const exactTotal = page.page.totalCount
    ? Number.parseInt(String(page.page.totalCount), 10)
    : undefined
  return {
    items,
    limit: page.page.pageSize,
    hasMore: page.page.hasMore,
    nextCursor: page.page.nextCursor,
    total:
      exactTotal !== undefined && Number.isSafeInteger(exactTotal) && exactTotal >= 0
        ? exactTotal
        : items.length + (page.page.hasMore ? 1 : 0),
  }
}

export function listQuery(params: AccessListParams) {
  return query({
    cursor: params.cursor,
    pageSize: params.limit,
    status: params.status,
    search: params.q?.trim() || undefined,
    includeTotal: params.includeTotal ? 'true' : undefined,
  })
}

export async function allPages<T>(
  operationId: ManagementApiOperationId,
  pathParameters?: Record<string, string | number>,
  params: AccessListParams = {},
): Promise<T[]> {
  const items: T[] = []
  let cursor = params.cursor
  do {
    const page = await request<ManagementPage<T>>(operationId, {
      pathParameters,
      query: listQuery({ ...params, cursor, limit: Math.min(params.limit ?? 200, 200) }),
    })
    items.push(...page.data)
    cursor = page.page.hasMore ? page.page.nextCursor : undefined
  } while (cursor)
  return items
}

export async function mutateAndRead<T>(
  operationId: ManagementApiOperationId,
  pathParameters: Record<string, string | number> | undefined,
  body: unknown,
  detail: (id: string) => Promise<T>,
  headers: Record<string, string>,
): Promise<T> {
  const receipt = await request<MutationReceipt>(operationId, { pathParameters, body, headers })
  if (!('resource' in receipt) || !receipt.resource.id) {
    throw new Error('Router Management mutation returned no resource.')
  }
  return detail(receipt.resource.id)
}
