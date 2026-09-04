import type { ModelProviderPreset } from './modelProviderCatalog'
import type { ModelPricing, ProviderReliability, RoutingModelCard } from './configPageSupport'

export interface ConnectedModelInput {
  provider: ModelProviderPreset
  baseUrl: string
  apiKey: string
  modelIds: string[]
  modelNames: Record<string, string>
  catalogModels: Record<string, string>
  reasoningFamily?: string
  metadata: Omit<RoutingModelCard, 'name'>
  pricing?: ModelPricing
  reliability?: ProviderReliability
}
