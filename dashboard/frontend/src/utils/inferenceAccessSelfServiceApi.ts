import {
  assertManagementMe,
  getManagementNamespace,
  managementOperationRequest,
} from './managementApiContract'
import { credentialAccessApi } from './inferenceAccessCredentialApi'
import { identityAccessApi } from './inferenceAccessIdentityApi'
import { listQuery, request, viewPage } from './inferenceAccessTransport'
import { keyUsage, requestLog, requestLogs, usage } from './inferenceAccessUsageApi'
import type { ManagementPage, ResourceDetail, SelfInferenceKey } from './routerManagementTypes'
import type {
  AccessListParams,
  AccessStatus,
  AccessTeam,
  SelfTeamCatalog,
  UsageFilter,
} from './inferenceAccessTypes'

async function selfTeams(): Promise<SelfTeamCatalog> {
  const identity = assertManagementMe(
    await managementOperationRequest('getMe', { namespace: null }),
  )
  const selectedNamespace = getManagementNamespace()
  const scope =
    identity.namespaces.find((item) => item.namespace.namespaceId === selectedNamespace) ??
    identity.namespaces[0]
  if (!scope?.user) {
    return { items: [], members: [], accessGroups: [], budgets: [] }
  }
  const user = scope.user
  const member = {
    id: user.userId,
    email: user.email,
    name: user.displayName,
    status: user.status === 'active' ? ('active' as const) : ('disabled' as const),
  }
  return {
    items: scope.teams.map((team) => ({
      id: team.teamId,
      name: team.name,
      description: '',
      status: team.status === 'active' ? 'active' : 'disabled',
      members: [{ teamId: team.teamId, userId: user.userId, role: team.role }],
      accessGroupIds: [],
      budgetId: '',
    })),
    members: [member],
    accessGroups: [],
    budgets: [],
  }
}

export const selfServiceAccessApi = {
  selfTeams,
  selfTeam: async (id: string) => {
    const catalog = await selfTeams()
    const team = catalog.items.find((item) => item.id === id)
    if (!team) throw new Error('Team is not visible to this user.')
    return team
  },
  saveSelfTeam: (item: Partial<AccessTeam> & { id: string }) => identityAccessApi.saveTeam(item),
  selfKeys: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<SelfInferenceKey>>('getSelfInferenceKeys', {
      query: listQuery({ ...params, limit: Math.min(params.limit ?? 25, 200) }),
    })
    return viewPage(page, (item) => ({
      id: item.keyId,
      name: item.name,
      prefix: '',
      contextTeamId: item.contextTeamId,
      ownerType: item.owner.type,
      ownerId: item.owner.id,
      status: 'active' as const,
      expiresAt: item.expiresAt,
      accessGroupIds: [],
    }))
  },
  selfKey: async (id: string) => {
    const detail = await request<ResourceDetail<SelfInferenceKey>>('getSelfInferenceKeysByKeyId', {
      pathParameters: { keyId: id },
    })
    return {
      id: detail.data.keyId,
      name: detail.data.name,
      prefix: '',
      contextTeamId: detail.data.contextTeamId,
      ownerType: detail.data.owner.type,
      ownerId: detail.data.owner.id,
      status: 'active' as const,
      expiresAt: detail.data.expiresAt,
      accessGroupIds: [],
    }
  },
  selfKeySecret: (id: string) => credentialAccessApi.keySecret(id),
  createSelfKey: (
    name: string,
    ownerType: 'user' | 'team',
    ownerId: string,
    contextTeamId?: string,
  ) =>
    credentialAccessApi.createKey({
      name,
      ownerType,
      ownerId,
      contextTeamId,
      accessGroupIds: [],
      revision: 0,
    }),
  rotateSelfKey: (id: string) => credentialAccessApi.rotateKey(id),
  setSelfKeyStatus: (id: string, status: AccessStatus) =>
    credentialAccessApi.setKeyStatus(id, status),
  deleteSelfKey: (id: string) => credentialAccessApi.deleteKey(id),
  selfUsage: usage,
  selfKeyUsage: keyUsage,
  selfRequestLogs: (filter: UsageFilter = {}) => requestLogs(filter),
  selfRequestLog: (id: string) => requestLog(id),
}
