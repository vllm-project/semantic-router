import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import {
  allPages,
  etag,
  idempotencyHeaders,
  listQuery,
  mutateAndRead,
  request,
  resource,
  viewPage,
} from './inferenceAccessTransport'
import {
  relationshipParams,
  subjectBindings,
  syncSubjectBindings,
} from './inferenceAccessPolicySupport'
import type {
  ManagementMembership,
  ManagementPage,
  ManagementTeam,
  ManagementUser,
  ResourceDetail,
} from './routerManagementTypes'
import type {
  AccessListParams,
  AccessStatus,
  AccessTeam,
  AccessUser,
  TeamMembership,
} from './inferenceAccessTypes'

function mapMembership(item: ManagementMembership): TeamMembership {
  if (item.role !== 'admin' && item.role !== 'member') {
    throw new Error('Router returned an unsupported team membership role.')
  }
  const relation = item as ManagementMembership & {
    teamName?: string
    teamStatus?: AccessStatus
    displayName?: string
    email?: string
    userStatus?: AccessStatus
  }
  return {
    teamId: item.teamId,
    userId: item.userId,
    role: item.role,
    revision: item.revision,
    teamName: relation.teamName,
    teamStatus: relation.teamStatus,
    userName: relation.displayName,
    userEmail: relation.email,
    userStatus: relation.userStatus,
  }
}

function mapUser(
  item: ManagementUser,
  memberships: ManagementMembership[] = [],
  accessGroupIds: string[] = [],
  budgetId?: string,
): AccessUser {
  return {
    id: item.userId,
    email: item.email,
    name: item.displayName,
    status: item.status === 'active' ? 'active' : 'disabled',
    revision: item.revision,
    accessGroupIds,
    budgetId,
    memberships: memberships.map(mapMembership),
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

function mapTeam(
  item: ManagementTeam,
  members: ManagementMembership[] = [],
  accessGroupIds: string[] = [],
  budgetId = '',
): AccessTeam {
  return {
    id: item.teamId,
    name: item.name,
    description: item.description,
    status: item.status === 'active' ? 'active' : 'disabled',
    revision: item.revision,
    members: members.map(mapMembership),
    accessGroupIds,
    budgetId,
    createdAt: item.createdAt,
    updatedAt: item.updatedAt,
  }
}

async function syncTeamMemberships(teamId: string, desired: TeamMembership[] | undefined) {
  if (desired === undefined) return
  const current = await allPages<ManagementMembership>('getTeamsByTeamIdMembers', { teamId })
  const desiredByUser = new Map(desired.map((membership) => [membership.userId, membership]))

  for (const membership of current) {
    const next = desiredByUser.get(membership.userId)
    if (!next) {
      await request('deleteTeamsByTeamIdMembersByUserId', {
        pathParameters: { teamId, userId: membership.userId },
        headers: {
          [MANAGEMENT_API_HEADERS.ifMatch]: etag('membership', membership.revision),
        },
      })
    } else if (next.role !== membership.role) {
      await request('patchTeamsByTeamIdMembersByUserId', {
        pathParameters: { teamId, userId: membership.userId },
        body: { role: next.role },
        headers: {
          [MANAGEMENT_API_HEADERS.ifMatch]: etag('membership', membership.revision),
        },
      })
    }
  }

  const existingUsers = new Set(current.map((membership) => membership.userId))
  for (const membership of desired) {
    if (!existingUsers.has(membership.userId)) {
      await request('putTeamsByTeamIdMembersByUserId', {
        pathParameters: { teamId, userId: membership.userId },
        body: { role: membership.role },
        headers: idempotencyHeaders(),
      })
    }
  }
}

async function user(id: string) {
  const [detail, memberships, bindings] = await Promise.all([
    request<ResourceDetail<ManagementUser>>('getUsersByUserId', {
      pathParameters: { userId: id },
    }),
    request<ManagementPage<ManagementMembership>>('getUsersByUserIdMemberships', {
      pathParameters: { userId: id },
      query: listQuery({ limit: 12, includeTotal: true }),
    })
      .then((page) => page.data)
      .catch(() => []),
    subjectBindings('user', id).catch(() => ({ accessGroupIds: [], budgetId: undefined })),
  ])
  return mapUser(detail.data, memberships, bindings.accessGroupIds, bindings.budgetId)
}

async function team(id: string) {
  const [detail, members, bindings] = await Promise.all([
    request<ResourceDetail<ManagementTeam>>('getTeamsByTeamId', {
      pathParameters: { teamId: id },
    }),
    request<ManagementPage<ManagementMembership>>('getTeamsByTeamIdMembers', {
      pathParameters: { teamId: id },
      query: listQuery({ limit: 12, includeTotal: true }),
    })
      .then((page) => page.data)
      .catch(() => []),
    subjectBindings('team', id).catch(() => ({ accessGroupIds: [], budgetId: undefined })),
  ])
  return mapTeam(detail.data, members, bindings.accessGroupIds, bindings.budgetId)
}

async function teamForEdit(id: string) {
  const [detail, members, bindings] = await Promise.all([
    request<ResourceDetail<ManagementTeam>>('getTeamsByTeamId', {
      pathParameters: { teamId: id },
    }),
    allPages<ManagementMembership>('getTeamsByTeamIdMembers', { teamId: id }),
    subjectBindings('team', id),
  ])
  return mapTeam(detail.data, members, bindings.accessGroupIds, bindings.budgetId)
}

async function saveUser(item: Partial<AccessUser> & { id: string }) {
  await mutateAndRead(
    'patchUsersByUserId',
    { userId: item.id },
    { email: item.email, displayName: item.name, status: item.status },
    user,
    { [MANAGEMENT_API_HEADERS.ifMatch]: etag('user', item.revision ?? 0) },
  )
  await syncSubjectBindings('user', item.id, item.accessGroupIds, item.budgetId)
  return user(item.id)
}

async function saveTeam(item: Partial<AccessTeam>) {
  let savedTeam: AccessTeam
  if (!item.id) {
    savedTeam = await mutateAndRead(
      'postTeams',
      undefined,
      {
        name: item.name,
        description: item.description,
        accessPolicyIds: item.accessGroupIds,
        rateLimitPolicyId: item.budgetId,
      },
      team,
      idempotencyHeaders(),
    )
  } else {
    savedTeam = await mutateAndRead(
      'patchTeamsByTeamId',
      { teamId: item.id },
      { name: item.name, description: item.description, status: item.status },
      team,
      { [MANAGEMENT_API_HEADERS.ifMatch]: etag('team', item.revision ?? 0) },
    )
    await syncSubjectBindings('team', savedTeam.id, item.accessGroupIds, item.budgetId)
  }
  await syncTeamMemberships(savedTeam.id, item.members)
  return team(savedTeam.id)
}

export const identityAccessApi = {
  users: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementUser>>('getUsers', {
      query: listQuery(params),
    })
    return viewPage(page, (item) => mapUser(item))
  },
  user,
  userMemberships: async (id: string, params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementMembership>>(
      'getUsersByUserIdMemberships',
      {
        pathParameters: { userId: id },
        query: listQuery(relationshipParams(params)),
      },
    )
    return viewPage(page, mapMembership)
  },
  userSummary: async (id: string) =>
    mapUser(
      resource(
        await request<ResourceDetail<ManagementUser>>('getUsersByUserId', {
          pathParameters: { userId: id },
        }),
      ),
    ),
  saveUser,
  deleteUser: async (id: string) => {
    const current = await user(id)
    await request('deleteUsersByUserId', {
      pathParameters: { userId: id },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('user', current.revision ?? 0) },
    })
  },
  teams: async (params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementTeam>>('getTeams', {
      query: listQuery(params),
    })
    return viewPage(page, (item) => mapTeam(item))
  },
  team,
  teamMembers: async (id: string, params: AccessListParams = {}) => {
    const page = await request<ManagementPage<ManagementMembership>>('getTeamsByTeamIdMembers', {
      pathParameters: { teamId: id },
      query: listQuery(relationshipParams(params)),
    })
    return viewPage(page, mapMembership)
  },
  teamForEdit,
  teamSummary: async (id: string) =>
    mapTeam(
      resource(
        await request<ResourceDetail<ManagementTeam>>('getTeamsByTeamId', {
          pathParameters: { teamId: id },
        }),
      ),
    ),
  saveTeam,
  deleteTeam: async (id: string) => {
    const current = await team(id)
    await request('deleteTeamsByTeamId', {
      pathParameters: { teamId: id },
      headers: { [MANAGEMENT_API_HEADERS.ifMatch]: etag('team', current.revision ?? 0) },
    })
  },
}
