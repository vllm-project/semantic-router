import { createContext, useContext } from 'react'

import type { AgentRuleFacts, AgentRuleIndex } from './insightsAgentIdentity'
import { modelProviderIconAssets } from './modelProviderIcons'

/**
 * What Insights reads from the router configuration: the agent rules of each
 * recipe, so a matched housekeeping rule marks its call, the provider each
 * configured model is served by, for its logo, and the model router_replay's
 * cost_baseline_model names, so a saving reads against it. Without the
 * configuration, as for a viewer who may not read it, they stay empty and the
 * pages degrade.
 */
export interface InsightsConfigFacts {
  agentRules?: AgentRuleIndex
  modelProviders: Record<string, string>
  /** The one model every router_replay plugin that sets cost_baseline_model names. */
  costBaselineModel?: string
}

interface ConfigPlugin {
  type?: string
  configuration?: { cost_baseline_model?: string }
}

export interface ConfigDocument {
  recipes?: Array<{
    name?: string
    signals?: { agent?: Array<AgentRuleFacts & { name?: string }> }
    decisions?: Array<{ plugins?: ConfigPlugin[] }>
  }>
  providers?: { models?: Array<{ name?: string; backend_refs?: Array<{ provider?: string }> }> }
  global?: { plugins?: ConfigPlugin[] }
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

/**
 * The model router_replay compares savings with when the operator names one:
 * global.plugins or a decision's plugins. Several different models leave it
 * unnamed, as does none.
 */
export function buildCostBaselineModel(config: ConfigDocument): string | undefined {
  const plugins = [
    ...(config.global?.plugins ?? []),
    ...(config.recipes ?? []).flatMap((recipe) =>
      (recipe.decisions ?? []).flatMap((decision) => decision.plugins ?? []),
    ),
  ]
  const models = new Set(
    plugins
      .filter((plugin) => plugin.type === 'router_replay')
      .map((plugin) => plugin.configuration?.cost_baseline_model?.trim())
      .filter((model): model is string => Boolean(model)),
  )
  return models.size === 1 ? [...models][0] : undefined
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
