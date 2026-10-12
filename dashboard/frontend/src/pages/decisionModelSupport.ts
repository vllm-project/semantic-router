import { DECISION_RUNTIME_CATALOG, DECISION_RUNTIME_CAPABILITIES } from './decisionRuntimeCatalog'

export type DecisionModelName = string
export const DEFAULT_DECISION_MODEL = 'Vela-2.0-0.3B'
export interface DecisionModelOption {
  name: string
  artifact: string
  label: string
  family: string
  provider: string
  hardware: string
  summary: string
  questionTypes: readonly string[]
  revision?: string
}

const DECISION_FAMILY_LABELS: Record<string, string> = {
  decision1: 'Decision 1.0',
  decision2: 'Decision 2.0',
  decision3: 'Decision 3.0',
}
const velaSizes = ['0.3B', '0.8B', '4B', '9B'] as const
export const DECISION_MODEL_OPTIONS: readonly DecisionModelOption[] = [
  ...velaSizes.map(
    (size): DecisionModelOption => ({
      name: `Vela-2.0-${size}`,
      artifact: `vllm-sr/Vela-2.0-${size}`,
      label: `Vela 2.0 ${size}`,
      family: 'Vela 2.0',
      provider: 'vllm-sr',
      hardware:
        size === '0.3B' ? 'CPU or GPU' : size === '0.8B' ? 'GPU recommended' : 'GPU required',
      summary: 'Typed decisions with classification, scoring, label sets and precise spans.',
      questionTypes: ['choice', 'score', 'noul', 'span', 'set'],
    }),
  ),
  ...DECISION_RUNTIME_CATALOG.map(
    (entry): DecisionModelOption => ({
      name: entry.name,
      artifact: entry.id,
      label:
        entry.family === 'decision3'
          ? `Decision 3.0 ${entry.name}`
          : entry.name.replace('Decision-', 'Decision ').replace(/-/g, ' '),
      family: DECISION_FAMILY_LABELS[entry.family] ?? 'Decision 2.0',
      provider: entry.provider,
      hardware: `${entry.minMemoryGiB} GiB minimum memory`,
      summary:
        entry.family === 'decision3'
          ? 'General judgment questions over text, images and videos for classification, scoring and routing decisions.'
          : 'General judgment questions for classification, scoring and routing decisions.',
      questionTypes: DECISION_RUNTIME_CAPABILITIES,
      revision: entry.revision,
    }),
  ),
]
export const DECISION_MODELS = DECISION_MODEL_OPTIONS.map((option) => option.name)
export const DECISION_MODEL_HINT =
  'Choose a deployed decision model. Task support follows its native capabilities and available adapters.'

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}
}
function catalogOf(config: unknown): Record<string, unknown> {
  return record(record(record(config).global).model_catalog)
}
export function configuredDecisionDeployment(config: unknown): string {
  const binding = record(record(catalogOf(config).system).decision_model)
  return typeof binding.deployment === 'string' ? binding.deployment : 'primary'
}
export function configuredDecisionModel(config: unknown): DecisionModelName {
  const catalog = catalogOf(config)
  const deployment = configuredDecisionDeployment(config)
  const artifact = record(record(catalog.deployments)[deployment]).artifact
  if (typeof artifact === 'string')
    return DECISION_MODEL_OPTIONS.find((entry) => entry.artifact === artifact)?.name ?? artifact
  return deployment === 'primary' && !record(record(catalog.system).decision_model).deployment
    ? DEFAULT_DECISION_MODEL
    : deployment
}

// Default selection references a resource; never mutate a previous resource that
// another task may explicitly bind. Callers provide a fresh canonical snapshot.
export function withDecisionModel<T extends Record<string, unknown>>(
  config: T,
  name: DecisionModelName,
): T {
  const option = DECISION_MODEL_OPTIONS.find((entry) => entry.name === name)
  if (!option) throw new Error('Choose a decision model from the catalog.')
  const global = record(config.global)
  const catalog = record(global.model_catalog)
  const deployments = { ...record(catalog.deployments) }
  let deployment = Object.keys(deployments).find(
    (key) => record(deployments[key]).artifact === option.artifact,
  )
  if (!deployment) {
    const stem = option.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')
    deployment = stem
    let suffix = 2
    while (deployments[deployment]) deployment = `${stem}-${suffix++}`
    deployments[deployment] = {
      provider: 'model_runtime',
      artifact: option.artifact,
      ...(option.revision ? { revision: option.revision } : {}),
      device: 'auto',
    }
  }
  return {
    ...config,
    global: {
      ...global,
      model_catalog: {
        ...catalog,
        deployments,
        system: { ...record(catalog.system), decision_model: { deployment } },
      },
    },
  }
}
