import {
  DECISION_RUNTIME_CATALOG,
  type DecisionRuntimeCatalogEntry,
} from './decisionRuntimeCatalog'

export type DecisionModelName = string
export const DEFAULT_DECISION_MODEL = 'vela3-fast'

/** cpu: an encoder that serves well on a CPU; flexible: a small decoder; gpu: a larger decoder. */
export type DecisionDeviceClass = 'cpu' | 'flexible' | 'gpu'

export interface DecisionModelOption {
  name: string
  artifact: string
  label: string
  /** Family label, such as Vela 3.0. */
  family: string
  /** The model runtime's registry family, such as vela3_d3. */
  familyId: string
  provider: string
  /** Null when the backbone and memory floor are not published. */
  device: DecisionDeviceClass | null
  hardware: string
  summary: string
  questionTypes: readonly string[]
  modalities: readonly string[]
  spanHeads: readonly string[]
  parameters: number | null
  minMemoryGiB: number | null
  backbone: string
  revision?: string
  /** False for a release the Router's decision catalog serves no questions with. */
  servable: boolean
  /** The pull request adding runtime support for an announced model. */
  pending?: string
}

export const DECISION_DEVICE_LABELS: Record<DecisionDeviceClass, string> = {
  cpu: 'CPU-friendly',
  flexible: 'CPU or GPU',
  gpu: 'GPU recommended',
}

const ENCODER_BACKBONES = new Set(['modernbert', 'ultrabert', 'ultravbert'])

export function decisionDeviceClass(
  entry: Pick<DecisionRuntimeCatalogEntry, 'backbone' | 'minMemoryGiB'>,
): DecisionDeviceClass | null {
  if (ENCODER_BACKBONES.has(entry.backbone)) return 'cpu'
  if (!entry.backbone || entry.minMemoryGiB === null) return null
  return entry.minMemoryGiB <= 8 ? 'flexible' : 'gpu'
}

/** The family a release belongs to: the version in its name, else its registry family's label. */
export function decisionFamilyLabel(
  entry: Pick<DecisionRuntimeCatalogEntry, 'name' | 'family' | 'familyLabel'>,
): string {
  const vela = /^vela(\d+)(?:\.(\d+))?-/i.exec(entry.name)
  if (vela) return `Vela ${vela[1]}.${vela[2] ?? '0'}`
  const named = /^(Vela|Decision)-(\d+\.\d+)-/.exec(entry.name)
  if (named) return `${named[1]} ${named[2]}`
  return entry.familyLabel || entry.family
}

export function decisionModelLabel(name: string): string {
  const vela = /^vela(\d+)(?:\.(\d+))?-(.+)$/i.exec(name)
  if (vela) {
    const variant = vela[3].replace(/-/g, ' ').replace(/\b(\d+)m\b/gi, '$1M')
    return `Vela ${vela[1]}.${vela[2] ?? '0'} ${variant}`
  }
  if (/^d\d/.test(name)) return `Decision 3.0 ${name}`
  return name.replace(/^(Decision|Vela)-/, '$1 ').replace(/-/g, ' ')
}

export function formatParameters(parameters: number | null): string {
  if (parameters === null || !Number.isFinite(parameters) || parameters <= 0) return 'Not published'
  if (parameters >= 1e9) return `${(parameters / 1e9).toFixed(parameters >= 1e10 ? 0 : 1)}B`
  return `${Math.round(parameters / 1e6)}M`
}

function familyRank(label: string): [number, number, number, string] {
  const match = /^(Vela|Decision) (\d+)\.(\d+)/.exec(label)
  if (!match) return [2, 0, 0, label]
  return [match[1] === 'Vela' ? 0 : 1, -Number(match[2]), -Number(match[3]), label]
}

export function compareDecisionFamilies(left: string, right: string): number {
  const a = familyRank(left)
  const b = familyRank(right)
  for (let index = 0; index < a.length; index++) {
    if (a[index] !== b[index]) return a[index] < b[index] ? -1 : 1
  }
  return 0
}

function catalogOption(entry: DecisionRuntimeCatalogEntry): DecisionModelOption {
  const device = decisionDeviceClass(entry)
  const servable = entry.questionTypes.length > 0 && !entry.pending
  return {
    name: entry.name,
    artifact: entry.id,
    label: decisionModelLabel(entry.name),
    family: decisionFamilyLabel(entry),
    familyId: entry.family,
    provider: entry.provider,
    device,
    hardware: [
      device ? DECISION_DEVICE_LABELS[device] : '',
      entry.minMemoryGiB === null ? '' : `${entry.minMemoryGiB} GiB memory`,
    ]
      .filter(Boolean)
      .join(' · '),
    summary:
      entry.summary ||
      (servable
        ? `${decisionModelLabel(entry.name)} answers ${entry.questionTypes.join(', ')} questions.`
        : 'Its serving path for decision questions is not implemented yet, so it answers none.'),
    questionTypes: entry.questionTypes,
    modalities: entry.modalities,
    spanHeads: entry.spanHeads,
    parameters: entry.parameters,
    minMemoryGiB: entry.minMemoryGiB,
    backbone: entry.backbone,
    revision: entry.revision || undefined,
    servable,
    ...(entry.pending ? { pending: entry.pending } : {}),
  }
}

const options = DECISION_RUNTIME_CATALOG.map(catalogOption)
const servableFamilies = new Set(
  options.filter((option) => option.servable).map((option) => option.family),
)

/** Every listed model, grouped by family: families with servable models first, newest Vela first, then by size. */
export const DECISION_CATALOG_MODELS: readonly DecisionModelOption[] = options.sort(
  (left, right) =>
    Number(servableFamilies.has(right.family)) - Number(servableFamilies.has(left.family)) ||
    compareDecisionFamilies(left.family, right.family) ||
    Number(right.servable) - Number(left.servable) ||
    (left.parameters ?? 0) - (right.parameters ?? 0) ||
    left.name.localeCompare(right.name),
)
/** The releases a deployment can serve decision questions with. */
export const DECISION_MODEL_OPTIONS: readonly DecisionModelOption[] =
  DECISION_CATALOG_MODELS.filter((option) => option.servable)
export const DECISION_MODELS = DECISION_MODEL_OPTIONS.map((option) => option.name)
export const DECISION_MODEL_HINT =
  'Choose a deployed decision model. Task support follows its native capabilities and available adapters.'

export function decisionModelByArtifact(
  artifact: string | undefined,
): DecisionModelOption | undefined {
  return artifact
    ? DECISION_CATALOG_MODELS.find((option) => option.artifact === artifact)
    : undefined
}

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

/** The deployment name the catalog suggests for a model: its name as a slug. */
export function decisionDeploymentStem(name: DecisionModelName): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-')
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
    const stem = decisionDeploymentStem(option.name)
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
