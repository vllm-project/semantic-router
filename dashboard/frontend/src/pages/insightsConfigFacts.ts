import { createContext, useContext } from 'react'

import type { AgentRuleFacts, AgentRuleIndex } from './insightsAgentIdentity'
import { modelProviderIconAssets } from './modelProviderIcons'

/**
 * What Insights reads from the router configuration: the agent rules of each
 * recipe, so a matched housekeeping rule marks its call, and the provider each
 * configured model is served by, for its logo. Without the configuration, as
 * for a viewer who may not read it, both stay empty and the pages degrade.
 */
export interface InsightsConfigFacts {
  agentRules?: AgentRuleIndex
  modelProviders: Record<string, string>
}

export interface ConfigDocument {
  recipes?: Array<{
    name?: string
    signals?: { agent?: Array<AgentRuleFacts & { name?: string }> }
  }>
  providers?: { models?: Array<{ name?: string; backend_refs?: Array<{ provider?: string }> }> }
}

export const EMPTY_CONFIG_FACTS: InsightsConfigFacts = { modelProviders: {} }

export const InsightsConfigContext = createContext<InsightsConfigFacts>(EMPTY_CONFIG_FACTS)

export function buildAgentRuleIndex(config: ConfigDocument): AgentRuleIndex {
  const index: AgentRuleIndex = new Map()
  for (const recipe of config.recipes ?? []) {
    const rules = new Map<string, AgentRuleFacts>()
    for (const rule of recipe.signals?.agent ?? []) {
      if (rule.name) rules.set(rule.name, rule)
    }
    index.set(recipe.name || 'default', rules)
  }
  return index
}

export function buildModelProviders(config: ConfigDocument): Record<string, string> {
  const providers: Record<string, string> = {}
  for (const model of config.providers?.models ?? []) {
    const provider = model.backend_refs?.find((ref) => ref.provider)?.provider
    if (model.name && provider) providers[model.name] = provider.toLowerCase()
  }
  return providers
}

// Hugging Face organisations that publish under another name than their logo's.
const VENDOR_ALIASES: Record<string, string> = {
  'meta-llama': 'meta',
  mistralai: 'mistral',
  moonshotai: 'moonshot',
  'deepseek-ai': 'deepseek',
  'zai-org': 'zai',
  'x-ai': 'xai',
  qwenlm: 'qwen',
}

/**
 * The provider whose logo a model shows: the organisation its id is published
 * under, as in qwen/qwen3-8b, else the provider the configuration serves it
 * from. A model neither names stays without a logo.
 */
export function modelProviderId(
  model: string | undefined,
  providers: Record<string, string>,
): string | undefined {
  if (!model) return undefined
  const slash = model.indexOf('/')
  if (slash > 0) {
    const vendor = model.slice(0, slash).toLowerCase()
    const id = VENDOR_ALIASES[vendor] ?? vendor
    if (modelProviderIconAssets[id]) return id
  }
  const configured = providers[model]
  return configured && modelProviderIconAssets[configured] ? configured : undefined
}

export function useInsightsConfig(): InsightsConfigFacts {
  return useContext(InsightsConfigContext)
}
