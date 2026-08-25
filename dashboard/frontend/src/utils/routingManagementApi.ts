import {
  hasOnlyKeys,
  isNonEmptyString,
  isRecord,
  isStringArray,
  managementOperationRequest,
} from './managementApiContract'
import { MANAGEMENT_API_HEADERS } from '../generated/managementApiContract'
import type {
  ManagementOperation,
  RoutingAssignmentModel,
  RoutingAssignmentSet,
  RoutingBulkImportRequest,
  RoutingClaimValue,
  RoutingDecision,
  RoutingEntrypoint,
  RoutingEntrypointRule,
  RoutingEntrypointWrite,
  RoutingFallbackPolicy,
  RoutingListParams,
  RoutingModel,
  RoutingModelCardView,
  RoutingModelControl,
  RoutingModelPatch,
  RoutingModelWrite,
  RoutingMutationReceipt,
  RoutingPage,
  RoutingProbeResponse,
  RoutingRecipe,
  RoutingRecipeProvenance,
  RoutingRecipeWrite,
  RoutingResolveResponse,
  RoutingStatus,
} from './routingManagementTypes'

export type * from './routingManagementTypes'

const isFiniteInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value)

const isStatus = (value: unknown): value is RoutingStatus =>
  value === 'draft' || value === 'active' || value === 'disabled'

function assertDecision(value: unknown): RoutingDecision {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['id', 'name', 'dispatchCardinality']) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    (value.dispatchCardinality !== 'single' && value.dispatchCardinality !== 'multi')
  ) {
    throw new Error('Router returned an invalid Recipe decision.')
  }
  return value as unknown as RoutingDecision
}

function assertAssignmentSet(value: unknown): RoutingAssignmentSet {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['models'], ['fallback']) ||
    !Array.isArray(value.models)
  ) {
    throw new Error('Router returned an invalid decision assignment.')
  }
  const models = value.models.map((model) => {
    if (
      !isRecord(model) ||
      !hasOnlyKeys(
        model,
        ['modelId', 'modelRevision', 'priority', 'weight'],
        ['loraName', 'reasoning'],
      ) ||
      !isNonEmptyString(model.modelId) ||
      !isFiniteInteger(model.modelRevision) ||
      !isFiniteInteger(model.priority) ||
      !isNonEmptyString(model.weight)
    ) {
      throw new Error('Router returned an invalid Model assignment.')
    }
    return model as unknown as RoutingAssignmentModel
  })
  let fallback: RoutingFallbackPolicy | undefined
  if (value.fallback !== undefined) {
    if (
      !isRecord(value.fallback) ||
      !hasOnlyKeys(value.fallback, ['strategy', 'on']) ||
      value.fallback.strategy !== 'priority' ||
      !Array.isArray(value.fallback.on) ||
      !value.fallback.on.every(
        (trigger) => trigger === 'unavailable' || trigger === 'overloaded' || trigger === 'timeout',
      )
    ) {
      throw new Error('Router returned an invalid fallback policy.')
    }
    fallback = value.fallback as unknown as RoutingFallbackPolicy
  }
  return { models, ...(fallback ? { fallback } : {}) }
}

function assertModelControl(value: unknown): RoutingModelControl {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['retry', 'timeout']) ||
    !isRecord(value.retry) ||
    !hasOnlyKeys(value.retry, ['count', 'on']) ||
    !isFiniteInteger(value.retry.count) ||
    value.retry.count < 0 ||
    value.retry.count > 5 ||
    !Array.isArray(value.retry.on) ||
    !value.retry.on.every(
      (trigger) => trigger === 'unavailable' || trigger === 'overloaded' || trigger === 'timeout',
    ) ||
    new Set(value.retry.on).size !== value.retry.on.length ||
    !isRecord(value.timeout) ||
    !hasOnlyKeys(value.timeout, ['request', 'stream']) ||
    !isNonEmptyString(value.timeout.request) ||
    !isNonEmptyString(value.timeout.stream)
  ) {
    throw new Error('Router returned invalid Model control.')
  }
  return value as unknown as RoutingModelControl
}

function assertModel(value: unknown): RoutingModel {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    !isStatus(value.status) ||
    !isFiniteInteger(value.revision) ||
    !isFiniteInteger(value.modelRevision) ||
    !isStringArray(value.aliases) ||
    !Array.isArray(value.capabilities) ||
    !Array.isArray(value.loras) ||
    !isRecord(value.control) ||
    !Array.isArray(value.backends)
  ) {
    throw new Error('Router returned an invalid Model.')
  }
  return { ...(value as unknown as RoutingModel), control: assertModelControl(value.control) }
}

function assertModelCard(value: unknown): RoutingModelCardView {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['id', 'name', 'card']) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    !isRecord(value.card) ||
    !hasOnlyKeys(
      value.card,
      ['aliases', 'capabilities', 'loras', 'tags'],
      ['paramSize', 'contextWindowSize', 'description', 'reasoning', 'qualityScore', 'modality'],
    ) ||
    !isStringArray(value.card.aliases) ||
    !isStringArray(value.card.capabilities) ||
    !isStringArray(value.card.loras) ||
    !isStringArray(value.card.tags) ||
    (value.card.paramSize !== undefined && typeof value.card.paramSize !== 'string') ||
    (value.card.contextWindowSize !== undefined &&
      (!isFiniteInteger(value.card.contextWindowSize) || value.card.contextWindowSize < 0)) ||
    (value.card.description !== undefined && typeof value.card.description !== 'string') ||
    (value.card.qualityScore !== undefined && typeof value.card.qualityScore !== 'number') ||
    (value.card.modality !== undefined && typeof value.card.modality !== 'string')
  ) {
    throw new Error('Router returned an invalid Model Card.')
  }
  if (value.card.reasoning !== undefined) {
    if (
      !isRecord(value.card.reasoning) ||
      !hasOnlyKeys(value.card.reasoning, [], ['type', 'efforts']) ||
      (value.card.reasoning.type !== undefined && typeof value.card.reasoning.type !== 'string') ||
      (value.card.reasoning.efforts !== undefined && !isStringArray(value.card.reasoning.efforts))
    ) {
      throw new Error('Router returned an invalid Model Card reasoning family.')
    }
  }
  return value as unknown as RoutingModelCardView
}

function assertRecipe(value: unknown): RoutingRecipe {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    !isStatus(value.status) ||
    !isFiniteInteger(value.revision) ||
    !isFiniteInteger(value.recipeRevision) ||
    (value.origin !== 'custom' && value.origin !== 'distribution') ||
    typeof value.immutable !== 'boolean' ||
    !Array.isArray(value.decisions) ||
    !isRecord(value.document)
  ) {
    throw new Error('Router returned an invalid Recipe.')
  }
  let provenance: RoutingRecipeProvenance | undefined
  if (value.provenance !== undefined) {
    if (
      !isRecord(value.provenance) ||
      !hasOnlyKeys(value.provenance, [
        'distributionId',
        'distributionVersion',
        'assetDigest',
        'sourceRecipeId',
        'sourceRevision',
        'recipeDigest',
        'installedAt',
      ]) ||
      !isNonEmptyString(value.provenance.distributionId) ||
      !isNonEmptyString(value.provenance.distributionVersion) ||
      !/^sha256:[a-f0-9]{64}$/.test(String(value.provenance.assetDigest)) ||
      !isNonEmptyString(value.provenance.sourceRecipeId) ||
      !isFiniteInteger(value.provenance.sourceRevision) ||
      !/^sha256:[a-f0-9]{64}$/.test(String(value.provenance.recipeDigest)) ||
      !isNonEmptyString(value.provenance.installedAt)
    ) {
      throw new Error('Router returned invalid Recipe provenance.')
    }
    provenance = value.provenance as unknown as RoutingRecipeProvenance
  }
  if (
    (value.origin === 'distribution' && (!value.immutable || !provenance)) ||
    (value.origin === 'custom' && (value.immutable || provenance))
  ) {
    throw new Error('Router returned inconsistent Recipe provenance.')
  }
  return {
    ...(value as unknown as RoutingRecipe),
    ...(provenance ? { provenance } : {}),
    decisions: value.decisions.map(assertDecision),
  }
}

function assertEntrypoint(value: unknown): RoutingEntrypoint {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.id) ||
    !isNonEmptyString(value.name) ||
    !isStatus(value.status) ||
    !isFiniteInteger(value.revision) ||
    !isFiniteInteger(value.entrypointRevision) ||
    !isStringArray(value.aliases) ||
    !isFiniteInteger(value.ruleCount) ||
    value.ruleCount < 0 ||
    !isFiniteInteger(value.assignedModelCount) ||
    value.assignedModelCount < 0
  ) {
    throw new Error('Router returned an invalid Entrypoint.')
  }
  let rules: RoutingEntrypointRule[] | undefined
  if (value.rules !== undefined) {
    if (!Array.isArray(value.rules)) throw new Error('Router returned invalid topology.')
    rules = value.rules.map((rule) => {
      if (
        !isRecord(rule) ||
        !isNonEmptyString(rule.id) ||
        !isNonEmptyString(rule.name) ||
        !isNonEmptyString(rule.recipeId) ||
        !isFiniteInteger(rule.recipeRevision) ||
        !isRecord(rule.assignments)
      ) {
        throw new Error('Router returned invalid topology.')
      }
      return {
        ...(rule as unknown as RoutingEntrypointRule),
        assignments: Object.fromEntries(
          Object.entries(rule.assignments).map(([decisionId, assignment]) => [
            decisionId,
            assertAssignmentSet(assignment),
          ]),
        ),
      }
    })
  }
  return {
    ...(value as unknown as RoutingEntrypoint),
    ...(rules ? { rules } : {}),
  }
}

function assertPage<T>(value: unknown, parseItem: (item: unknown) => T): RoutingPage<T> {
  if (!isRecord(value) || !Array.isArray(value.data) || !isRecord(value.page)) {
    throw new Error('Router returned an invalid routing page.')
  }
  if (typeof value.page.hasMore !== 'boolean' || !isFiniteInteger(value.page.pageSize)) {
    throw new Error('Router returned invalid routing pagination.')
  }
  return {
    data: value.data.map(parseItem),
    page: {
      hasMore: value.page.hasMore,
      pageSize: value.page.pageSize,
      ...(isNonEmptyString(value.page.nextCursor) ? { nextCursor: value.page.nextCursor } : {}),
    },
  }
}

function assertDetail<T>(value: unknown, parseItem: (item: unknown) => T): T {
  if (!isRecord(value) || !hasOnlyKeys(value, ['data'])) {
    throw new Error('Router returned an invalid routing detail.')
  }
  return parseItem(value.data)
}

function assertMutationReceipt(value: unknown): RoutingMutationReceipt {
  if (!isRecord(value) || !hasOnlyKeys(value, [], ['resource', 'operation', 'idempotency'])) {
    throw new Error('Router returned an invalid routing mutation receipt.')
  }
  const hasResource = value.resource !== undefined
  const hasOperation = value.operation !== undefined
  if (hasResource === hasOperation) {
    throw new Error('Router returned an invalid routing mutation receipt.')
  }
  if (hasResource) {
    if (
      !isRecord(value.resource) ||
      !hasOnlyKeys(value.resource, ['kind', 'id', 'revision']) ||
      !isNonEmptyString(value.resource.kind) ||
      !isNonEmptyString(value.resource.id) ||
      !isFiniteInteger(value.resource.revision)
    ) {
      throw new Error('Router returned an invalid routing mutation receipt.')
    }
  }
  if (hasOperation) {
    if (
      !isRecord(value.operation) ||
      !hasOnlyKeys(value.operation, ['operationId'], ['desiredRevision']) ||
      !isNonEmptyString(value.operation.operationId) ||
      (value.operation.desiredRevision !== undefined &&
        !isFiniteInteger(value.operation.desiredRevision))
    ) {
      throw new Error('Router returned an invalid routing mutation receipt.')
    }
  }
  if (value.idempotency !== undefined) {
    if (
      !isRecord(value.idempotency) ||
      !hasOnlyKeys(value.idempotency, ['replayed'], ['originalRequestId']) ||
      typeof value.idempotency.replayed !== 'boolean' ||
      (value.idempotency.originalRequestId !== undefined &&
        !isNonEmptyString(value.idempotency.originalRequestId))
    ) {
      throw new Error('Router returned an invalid routing mutation receipt.')
    }
  }
  return value as unknown as RoutingMutationReceipt
}

function assertProbeResponse(value: unknown): RoutingProbeResponse {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['reachable', 'latencyMilliseconds', 'checkedAt']) ||
    typeof value.reachable !== 'boolean' ||
    !isFiniteInteger(value.latencyMilliseconds) ||
    !isNonEmptyString(value.checkedAt)
  ) {
    throw new Error('Router returned an invalid Model probe.')
  }
  return value as unknown as RoutingProbeResponse
}

function assertOperation(value: unknown): ManagementOperation {
  if (
    !isRecord(value) ||
    !isNonEmptyString(value.operationId) ||
    !isNonEmptyString(value.kind) ||
    !['pending', 'running', 'succeeded', 'partially_succeeded', 'failed', 'cancelled'].includes(
      String(value.state),
    )
  ) {
    throw new Error('Router returned an invalid operation.')
  }
  return value as unknown as ManagementOperation
}

function assertResolveResponse(value: unknown): RoutingResolveResponse {
  if (
    !isRecord(value) ||
    !hasOnlyKeys(value, ['outcome'], ['entrypoint', 'rule', 'recipe']) ||
    (value.outcome !== 'matched' &&
      value.outcome !== 'claimed_no_match' &&
      value.outcome !== 'unclaimed')
  ) {
    throw new Error('Router returned an invalid Entrypoint resolution.')
  }
  if (value.rule !== undefined) {
    const rule = value.rule
    if (
      !isRecord(rule) ||
      !isNonEmptyString(rule.id) ||
      !isNonEmptyString(rule.name) ||
      !isNonEmptyString(rule.recipeId) ||
      !isFiniteInteger(rule.recipeRevision) ||
      !isRecord(rule.assignments)
    ) {
      throw new Error('Router returned an invalid Entrypoint resolution.')
    }
    Object.values(rule.assignments).forEach(assertAssignmentSet)
  }
  if (value.entrypoint !== undefined) {
    const entrypoint = value.entrypoint
    if (
      !isRecord(entrypoint) ||
      !hasOnlyKeys(entrypoint, ['id', 'revision', 'name', 'aliases']) ||
      !isNonEmptyString(entrypoint.id) ||
      !isFiniteInteger(entrypoint.revision) ||
      !isNonEmptyString(entrypoint.name) ||
      !isStringArray(entrypoint.aliases)
    ) {
      throw new Error('Router returned an invalid Entrypoint resolution.')
    }
  }
  if (value.recipe !== undefined) {
    const recipe = value.recipe
    if (
      !isRecord(recipe) ||
      !hasOnlyKeys(recipe, ['id', 'revision', 'name', 'decisions', 'document']) ||
      !isNonEmptyString(recipe.id) ||
      !isFiniteInteger(recipe.revision) ||
      !isNonEmptyString(recipe.name) ||
      !Array.isArray(recipe.decisions) ||
      !isRecord(recipe.document)
    ) {
      throw new Error('Router returned an invalid Entrypoint resolution.')
    }
    recipe.decisions.forEach(assertDecision)
  }
  return value as unknown as RoutingResolveResponse
}

type RoutingListOperation =
  | 'getRoutingModels'
  | 'getRoutingModelCards'
  | 'getRoutingRecipes'
  | 'getRoutingEntrypoints'

async function listPage<T>(
  operationId: RoutingListOperation,
  parseItem: (item: unknown) => T,
  params: RoutingListParams = {},
): Promise<RoutingPage<T>> {
  const query = new URLSearchParams({
    pageSize: String(Math.min(Math.max(params.pageSize ?? 100, 1), 100)),
  })
  if (params.cursor) query.set('cursor', params.cursor)
  if (params.search?.trim()) query.set('search', params.search.trim())
  if (params.status) query.set('status', params.status)
  return assertPage(
    await managementOperationRequest(operationId, { query, signal: params.signal }),
    parseItem,
  )
}

async function listAll<T>(
  operationId: RoutingListOperation,
  parseItem: (item: unknown) => T,
  signal?: AbortSignal,
): Promise<T[]> {
  const result: T[] = []
  let cursor: string | undefined
  const seenCursors = new Set<string>()
  do {
    const page = await listPage(operationId, parseItem, { cursor, pageSize: 100, signal })
    result.push(...page.data)
    cursor = page.page.hasMore ? page.page.nextCursor : undefined
    if (page.page.hasMore && !cursor)
      throw new Error('Router returned an incomplete routing cursor.')
    if (cursor && seenCursors.has(cursor)) {
      throw new Error('Router returned a repeated routing cursor.')
    }
    if (cursor) seenCursors.add(cursor)
  } while (cursor)
  return result
}

const commandHeaders = () => ({ [MANAGEMENT_API_HEADERS.idempotencyKey]: crypto.randomUUID() })
const revisionHeaders = (kind: 'mdl' | 'rcp' | 'ep', revision: number) => ({
  [MANAGEMENT_API_HEADERS.ifMatch]: `"${kind}:${revision}"`,
})

export const routingManagementApi = {
  listModels: () => listAll('getRoutingModels', assertModel),
  listModelCards: (signal?: AbortSignal) =>
    listAll('getRoutingModelCards', assertModelCard, signal),
  listRecipes: () => listAll('getRoutingRecipes', assertRecipe),
  listEntrypoints: () => listAll('getRoutingEntrypoints', assertEntrypoint),
  listModelsPage: (params?: RoutingListParams) => listPage('getRoutingModels', assertModel, params),
  listEntrypointsPage: (params?: RoutingListParams) =>
    listPage('getRoutingEntrypoints', assertEntrypoint, params),
  getModel: async (id: string) =>
    assertDetail(
      await managementOperationRequest('getRoutingModelsByModelId', {
        pathParameters: { modelId: id },
      }),
      assertModel,
    ),
  getRecipe: async (id: string) =>
    assertDetail(
      await managementOperationRequest('getRoutingRecipesByRecipeId', {
        pathParameters: { recipeId: id },
      }),
      assertRecipe,
    ),
  getEntrypoint: async (id: string) =>
    assertDetail(
      await managementOperationRequest('getRoutingEntrypointsByEntrypointId', {
        pathParameters: { entrypointId: id },
      }),
      assertEntrypoint,
    ),
  getEntrypointTopology: async (id: string) => {
    const payload = await managementOperationRequest('getRoutingEntrypointsByEntrypointId', {
      pathParameters: { entrypointId: id },
      query: new URLSearchParams({ includeTopology: 'true' }),
    })
    return assertDetail(payload, assertEntrypoint)
  },
  createModel: (input: RoutingModelWrite) =>
    managementOperationRequest('postRoutingModels', {
      body: input,
      headers: commandHeaders(),
    }).then(assertMutationReceipt),
  updateModel: (id: string, revision: number, input: RoutingModelPatch) =>
    managementOperationRequest('patchRoutingModelsByModelId', {
      pathParameters: { modelId: id },
      body: input,
      headers: revisionHeaders('mdl', revision),
    }).then(assertMutationReceipt),
  deleteModel: async (id: string, revision: number) => {
    await managementOperationRequest('deleteRoutingModelsByModelId', {
      pathParameters: { modelId: id },
      headers: revisionHeaders('mdl', revision),
    })
  },
  createRecipe: (input: RoutingRecipeWrite) =>
    managementOperationRequest('postRoutingRecipes', {
      body: input,
      headers: commandHeaders(),
    }).then(assertMutationReceipt),
  updateRecipe: (id: string, revision: number, input: RoutingRecipeWrite) =>
    managementOperationRequest('patchRoutingRecipesByRecipeId', {
      pathParameters: { recipeId: id },
      body: input,
      headers: revisionHeaders('rcp', revision),
    }).then(assertMutationReceipt),
  deleteRecipe: async (id: string, revision: number) => {
    await managementOperationRequest('deleteRoutingRecipesByRecipeId', {
      pathParameters: { recipeId: id },
      headers: revisionHeaders('rcp', revision),
    })
  },
  createEntrypoint: (input: RoutingEntrypointWrite) =>
    managementOperationRequest('postRoutingEntrypoints', {
      body: input,
      headers: commandHeaders(),
    }).then(assertMutationReceipt),
  updateEntrypoint: (id: string, revision: number, input: RoutingEntrypointWrite) =>
    managementOperationRequest('patchRoutingEntrypointsByEntrypointId', {
      pathParameters: { entrypointId: id },
      body: input,
      headers: revisionHeaders('ep', revision),
    }).then(assertMutationReceipt),
  deleteEntrypoint: async (id: string, revision: number) => {
    await managementOperationRequest('deleteRoutingEntrypointsByEntrypointId', {
      pathParameters: { entrypointId: id },
      headers: revisionHeaders('ep', revision),
    })
  },
  publishEntrypoint: (id: string, revision: number) =>
    managementOperationRequest('postRoutingEntrypointsByEntrypointIdPublish', {
      pathParameters: { entrypointId: id },
      headers: { ...revisionHeaders('ep', revision), ...commandHeaders() },
    }).then(assertMutationReceipt),
  unpublishEntrypoint: (id: string, revision: number) =>
    managementOperationRequest('postRoutingEntrypointsByEntrypointIdUnpublish', {
      pathParameters: { entrypointId: id },
      headers: { ...revisionHeaders('ep', revision), ...commandHeaders() },
    }).then(assertMutationReceipt),
  probeModel: async (id: string) =>
    assertProbeResponse(
      await managementOperationRequest('postRoutingModelsByModelIdProbe', {
        pathParameters: { modelId: id },
      }),
    ),
  resolveEntrypoint: async (
    id: string,
    input: { path?: string; claims?: Record<string, RoutingClaimValue> },
  ) =>
    assertResolveResponse(
      await managementOperationRequest('postRoutingEntrypointsByEntrypointIdResolve', {
        pathParameters: { entrypointId: id },
        body: input,
      }),
    ),
  bulkImportModels: (input: RoutingBulkImportRequest) =>
    managementOperationRequest('postRoutingModelsBulkImport', {
      body: input,
      headers: commandHeaders(),
    }).then(assertMutationReceipt),
  getOperation: async (operationId: string) =>
    assertOperation(
      await managementOperationRequest('getOperationsByOperationId', {
        pathParameters: { operationId },
      }),
    ),
}

export async function waitForRoutingMutation(
  receipt: RoutingMutationReceipt,
  options: { timeoutMilliseconds?: number; pollMilliseconds?: number } = {},
): Promise<void> {
  if (!receipt.operation) return
  const timeout = options.timeoutMilliseconds ?? 30_000
  const poll = options.pollMilliseconds ?? 250
  const deadline = Date.now() + timeout
  while (Date.now() < deadline) {
    const operation = await routingManagementApi.getOperation(receipt.operation.operationId)
    if (operation.state === 'succeeded') return
    if (
      operation.state === 'failed' ||
      operation.state === 'partially_succeeded' ||
      operation.state === 'cancelled'
    ) {
      const reason = operation.itemErrors?.[0]?.reason
      throw new Error(reason || `Model operation ${operation.state.replace('_', ' ')}.`)
    }
    await new Promise<void>((resolve) => window.setTimeout(resolve, poll))
  }
  throw new Error('Model operation is still running. Refresh to see its latest state.')
}
