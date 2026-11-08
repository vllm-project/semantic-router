import { fromCanonicalLayout, toCanonicalLayout } from './configPageCanonicalLayout'
import type { ConfigData, RoutingModelCard } from './configPageSupport'

const cloneUnknown = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T

export const cloneConfigData = (value: ConfigData): ConfigData => cloneUnknown(value)

export const ensureRoutingConfig = (cfg: ConfigData) => {
  if (!cfg.routing) {
    cfg.routing = {}
  }
  if (!cfg.routing.modelCards) {
    cfg.routing.modelCards = []
  }
  return cfg.routing
}

export const ensureProvidersConfig = (cfg: ConfigData) => {
  if (!cfg.providers) {
    cfg.providers = { models: [] }
  }
  if (!cfg.providers.models) {
    cfg.providers.models = []
  }
  return cfg.providers
}

export const upsertRoutingModelCard = (
  cfg: ConfigData,
  name: string,
  patch: Partial<RoutingModelCard> = {},
) => {
  const routing = ensureRoutingConfig(cfg)
  const cards = [...(routing.modelCards || [])]
  const index = cards.findIndex((card) => card.name === name)
  if (index >= 0) {
    cards[index] = { ...cards[index], ...patch, name }
  } else {
    cards.push({ name, ...patch })
  }
  routing.modelCards = cards
}

export const removeRoutingModelCard = (cfg: ConfigData, name: string) => {
  const routing = ensureRoutingConfig(cfg)
  routing.modelCards = (routing.modelCards || []).filter((card) => card.name !== name)
}

export const hasRoutingModelCardMetadata = (card: Omit<RoutingModelCard, 'name'>): boolean =>
  Object.values(card).some((value) =>
    Array.isArray(value) ? value.length > 0 : value !== undefined && value !== '',
  )

export const writeRoutingModelCard = (
  cfg: ConfigData,
  name: string,
  card: Omit<RoutingModelCard, 'name'>,
  options: { removeWhenEmpty?: boolean } = {},
) => {
  if (hasRoutingModelCardMetadata(card)) {
    upsertRoutingModelCard(cfg, name, card)
  } else if (options.removeWhenEmpty) {
    removeRoutingModelCard(cfg, name)
  }
}

export const routingModelCardReferenceCount = (cfg: ConfigData, name: string): number =>
  (cfg.providers?.models || []).filter((model) => (model.catalog?.trim() || model.name) === name)
    .length

export const removeModelCardDataIfUnreferenced = (cfg: ConfigData, name: string) => {
  if (routingModelCardReferenceCount(cfg, name) === 0) {
    removeRoutingModelCard(cfg, name)
    if (cfg.evaluation?.records) {
      cfg.evaluation.records = cfg.evaluation.records.filter((record) => record.model !== name)
      if (
        cfg.evaluation.records.length === 0 &&
        !cfg.evaluation.benchmarks?.length &&
        !cfg.evaluation.indices?.length
      ) {
        delete cfg.evaluation
      }
    }
  }
}

const canonicalizeManagerView = (updatedConfig: ConfigData): ConfigData => {
  const next = cloneConfigData(updatedConfig)
  const routing = ensureRoutingConfig(next)
  if (next.signals) {
    routing.signals = cloneUnknown(next.signals)
    delete next.signals
  }
  if (next.projections) {
    routing.projections = cloneUnknown(next.projections)
    delete next.projections
  }
  if (next.decisions) {
    routing.decisions = cloneUnknown(next.decisions)
    delete next.decisions
  }
  return next
}

export const canonicalizeConfigForManagerSave = (updatedConfig: ConfigData): ConfigData =>
  toCanonicalLayout(canonicalizeManagerView(updatedConfig))

export const projectCanonicalConfigForManager = (data: ConfigData): ConfigData => {
  const next = canonicalizeManagerView(fromCanonicalLayout(data))
  if (next.routing?.signals) {
    next.signals = cloneUnknown(next.routing.signals)
  }
  if (next.routing?.projections) {
    next.projections = cloneUnknown(next.routing.projections)
  }
  if (next.routing?.decisions) {
    next.decisions = cloneUnknown(next.routing.decisions)
  }
  return next
}
