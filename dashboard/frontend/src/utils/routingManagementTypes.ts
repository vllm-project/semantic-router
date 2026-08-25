import type {
  MutationReceipt,
  Operation,
  RoutingAssignmentReasoning as WireRoutingAssignmentReasoning,
  RoutingAssignmentSetView,
  RoutingAssignmentSetWrite as WireRoutingAssignmentSetWrite,
  RoutingAssignmentView,
  RoutingAssignmentWrite,
  RoutingBulkImportRequest as WireRoutingBulkImportRequest,
  RoutingBulkModelSelection as WireRoutingBulkModelSelection,
  RoutingClaimValue as WireRoutingClaimValue,
  RoutingDecision as WireRoutingDecision,
  RoutingEntrypointPage,
  RoutingEntrypointRuleView,
  RoutingEntrypointView,
  RoutingEntrypointWrite as WireRoutingEntrypointWrite,
  RoutingFallbackPolicy as WireRoutingFallbackPolicy,
  RoutingMatcher as WireRoutingMatcher,
  RoutingModelCard as WireRoutingModelCard,
  RoutingModelCardView as WireRoutingModelCardView,
  RoutingModelControl as WireRoutingModelControl,
  RoutingModelPatch as WireRoutingModelPatch,
  RoutingModelView,
  RoutingModelWrite as WireRoutingModelWrite,
  RoutingPricing as WireRoutingPricing,
  RoutingProbeResponse as WireRoutingProbeResponse,
  RoutingRecipeProvenanceView,
  RoutingRecipeView,
  RoutingRecipeWrite as WireRoutingRecipeWrite,
  RoutingResolveResponse as WireRoutingResolveResponse,
} from '../generated/managementApiContract'

// Wire resources are generated from OpenAPI. The only enriched type below is
// RoutingModel: the UI normalizes Router defaults before rendering controls.
export type RoutingStatus = RoutingModelView['status']
export type DispatchCardinality = WireRoutingDecision['dispatchCardinality']
export type FallbackTrigger = WireRoutingFallbackPolicy['on'][number]

export type RoutingDecision = WireRoutingDecision
export type RoutingAssignmentReasoning = WireRoutingAssignmentReasoning
export type RoutingAssignmentModel = RoutingAssignmentView
export type RoutingFallbackPolicy = WireRoutingFallbackPolicy
export type RoutingAssignmentSet = RoutingAssignmentSetView
export type RoutingAssignmentModelWrite = RoutingAssignmentWrite
export type RoutingAssignmentSetWrite = WireRoutingAssignmentSetWrite
export type RoutingMatcher = WireRoutingMatcher
export type RoutingClaimValue = WireRoutingClaimValue

type RequiredDefined<T> = { [Key in keyof T]-?: Exclude<T[Key], undefined> }
type RoutingModelRetry = RequiredDefined<NonNullable<WireRoutingModelControl['retry']>>
type RoutingModelTimeout = RequiredDefined<NonNullable<WireRoutingModelControl['timeout']>>

/** UI-normalized defaults returned by routingManagementApi. */
export type RoutingModelControl = {
  retry: RoutingModelRetry
  timeout: RoutingModelTimeout
}

/** UI-normalized pricing; unset prices are represented by null. */
export type RoutingPricing = RequiredDefined<WireRoutingPricing>

export type RoutingModel = Omit<RoutingModelView, 'control' | 'pricing'> & {
  control: RoutingModelControl
  pricing: RoutingPricing
}
export type RoutingModelCard = WireRoutingModelCard
export type RoutingModelCardView = WireRoutingModelCardView
export type RoutingModelControlWrite = WireRoutingModelControl
export type RoutingModelBackendWrite = WireRoutingModelWrite['backends'][number]
export type RoutingModelWrite = WireRoutingModelWrite
export type RoutingModelPatch = WireRoutingModelPatch
export type RoutingBulkModelSelection = WireRoutingBulkModelSelection
export type RoutingBulkImportRequest = WireRoutingBulkImportRequest
export type RoutingRecipe = RoutingRecipeView
export type RoutingRecipeProvenance = RoutingRecipeProvenanceView
export type RoutingEntrypointRule = RoutingEntrypointRuleView
export type RoutingEntrypoint = RoutingEntrypointView
export type RoutingEntrypointWrite = WireRoutingEntrypointWrite
export type RoutingRecipeWrite = WireRoutingRecipeWrite
export type RoutingMutationReceipt = MutationReceipt
export type RoutingResolveResponse = WireRoutingResolveResponse
export type RoutingProbeResponse = WireRoutingProbeResponse
export type ManagementOperation = Operation
export type RoutingPage<T> = Omit<RoutingEntrypointPage, 'data'> & { data: T[] }

export interface RoutingListParams {
  search?: string
  cursor?: string
  pageSize?: number
  status?: RoutingStatus
  signal?: AbortSignal
}
