import { credentialAccessApi } from './inferenceAccessCredentialApi'
import { identityAccessApi } from './inferenceAccessIdentityApi'
import { policyAccessApi } from './inferenceAccessPolicyApi'
import { selfServiceAccessApi } from './inferenceAccessSelfServiceApi'
import { query, request, viewPage } from './inferenceAccessTransport'
import {
  keyUsage,
  overview,
  requestLog,
  requestLogs,
  teamUsage,
  usage,
  userUsage,
} from './inferenceAccessUsageApi'
import type { ManagementPage } from './routerManagementTypes'
import type { AccessAuditEvent, AccessListParams } from './inferenceAccessTypes'

export type * from './inferenceAccessTypes'

async function auditLogs(filter: AccessListParams = {}) {
  const page = await request<ManagementPage<AccessAuditEvent>>('getAuditEvents', {
    // Audit exposes typed exact filters, not the generic collection search
    // contract. Keep free-text filtering in the product view until the
    // Router publishes a bounded audit-search selector.
    query: query({ cursor: filter.cursor, pageSize: filter.limit }),
  })
  return viewPage(page, (item) => item, filter.q?.trim())
}

export const inferenceAccessApi = {
  overview,
  ...identityAccessApi,
  ...credentialAccessApi,
  ...policyAccessApi,
  usage,
  keyUsage,
  userUsage,
  teamUsage,
  requestLogs,
  requestLog,
  auditLogs,
  ...selfServiceAccessApi,
}
