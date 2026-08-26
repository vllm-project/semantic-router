import {
  inferenceAccessApi,
  type AccessListParams,
  type AccessPage,
  type AccessResourceOption,
} from '../utils/inferenceAccessApi'
import {
  routingManagementApi,
  type RoutingEntrypoint,
  type RoutingListParams,
  type RoutingModel,
  type RoutingPage,
} from '../utils/routingManagementApi'
import type { AccessPickerSource } from './AccessAsyncResourcePicker'
import type {
  AccessBudget,
  AccessAPIKey,
  AccessGroup,
  AccessTeam,
  AccessUser,
} from '../utils/inferenceAccessApi'

export interface AccessControlSelectorSources {
  users: AccessPickerSource<AccessUser>
  teams: AccessPickerSource<AccessTeam>
  groups: AccessPickerSource<AccessGroup>
  budgets: AccessPickerSource<AccessBudget>
  keys: AccessPickerSource<AccessAPIKey>
  models: AccessPickerSource<AccessResourceOption>
  entrypoints: AccessPickerSource<AccessResourceOption>
}

function routingPickerPage<T extends RoutingModel | RoutingEntrypoint>(
  page: RoutingPage<T>,
  resourceType: AccessResourceOption['resourceType'],
): AccessPage<AccessResourceOption> {
  return {
    items: page.data.map((item) => ({
      resourceType,
      resourceId: item.id,
      name: item.name,
      status: item.status,
    })),
    limit: page.page.pageSize,
    hasMore: page.page.hasMore,
    ...(page.page.nextCursor ? { nextCursor: page.page.nextCursor } : {}),
    total: page.data.length + (page.page.hasMore ? 1 : 0),
  }
}

function routingListParams(params: AccessListParams) {
  return {
    search: params.q,
    cursor: params.cursor,
    pageSize: params.limit,
    status:
      params.status === 'active' || params.status === 'draft' || params.status === 'disabled'
        ? params.status
        : undefined,
  } satisfies RoutingListParams
}

const routingSourceIdentity = {
  id: (item: AccessResourceOption) => item.resourceId,
  title: (item: AccessResourceOption) => item.name,
  description: (item: AccessResourceOption) =>
    item.status === 'active' ? 'Available' : item.status === 'draft' ? 'Draft' : 'Disabled',
}

export const accessControlSelectorSources: AccessControlSelectorSources = {
  users: {
    list: inferenceAccessApi.users,
    detail: inferenceAccessApi.userSummary,
    id: (item) => item.id,
    title: (item) => item.name,
    description: (item) => item.email,
  },
  teams: {
    list: inferenceAccessApi.teams,
    detail: inferenceAccessApi.teamSummary,
    id: (item) => item.id,
    title: (item) => item.name,
    description: (item) => item.description || item.id,
  },
  groups: {
    list: inferenceAccessApi.groups,
    detail: inferenceAccessApi.group,
    id: (item) => item.id,
    title: (item) => item.name,
    description: (item) =>
      `${item.resources.length} model grant${item.resources.length === 1 ? '' : 's'}`,
  },
  budgets: {
    list: inferenceAccessApi.budgets,
    detail: inferenceAccessApi.budget,
    id: (item) => item.id,
    title: (item) => item.name,
    description: (item) => `${item.rules.length} limit${item.rules.length === 1 ? '' : 's'}`,
  },
  keys: {
    list: inferenceAccessApi.keys,
    detail: inferenceAccessApi.keySummary,
    id: (item) => item.id,
    title: (item) => item.name,
    description: (item) => item.id,
  },
  models: {
    list: async (params) =>
      routingPickerPage(
        await routingManagementApi.listModelsPage(routingListParams(params)),
        'model',
      ),
    detail: async (id) => {
      const item = await routingManagementApi.getModel(id)
      return {
        resourceType: 'model',
        resourceId: item.id,
        name: item.name,
        status: item.status,
      }
    },
    ...routingSourceIdentity,
  },
  entrypoints: {
    list: async (params) =>
      routingPickerPage(
        await routingManagementApi.listEntrypointsPage(routingListParams(params)),
        'entrypoint',
      ),
    detail: async (id) => {
      const item = await routingManagementApi.getEntrypoint(id)
      return {
        resourceType: 'entrypoint',
        resourceId: item.id,
        name: item.name,
        status: item.status,
      }
    },
    ...routingSourceIdentity,
  },
}
