import type {
  ConfigData,
  DecisionCondition,
  NumericPredicate,
  SignalType,
  StructureFeature,
  StructureSource,
  Subject,
} from './configPageSupport'

export const STRUCTURE_FEATURE_TYPES = ['exists', 'count', 'density', 'sequence'] as const
export const STRUCTURE_SOURCE_TYPES = ['regex', 'keyword_set', 'sequence'] as const

export const DEFAULT_STRUCTURE_FEATURE: StructureFeature = {
  type: 'count',
  source: {
    type: 'regex',
    pattern: '[?？]',
  },
}

export const DEFAULT_STRUCTURE_PREDICATE: NumericPredicate = { gte: 3 }

export function readStringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

export function normalizeStringList(value: unknown, label: string, required = false): string[] {
  const source = readStringList(value)
  if (source.some((item) => !item.trim())) {
    throw new Error(`${label} cannot contain empty values.`)
  }

  const normalized = source.map((item) => item.trim())
  if (required && normalized.length === 0) {
    throw new Error(`Please provide at least one ${label.toLowerCase()}.`)
  }

  const seen = new Set<string>()
  for (const item of normalized) {
    const key = item.toLocaleLowerCase()
    if (seen.has(key)) {
      throw new Error(`${label} must not contain duplicate values.`)
    }
    seen.add(key)
  }
  return normalized
}

export function readConditions(value: unknown): DecisionCondition[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return []
    const record = item as Record<string, unknown>
    return typeof record.type === 'string' && typeof record.name === 'string'
      ? [{ type: record.type, name: record.name }]
      : []
  })
}

export function normalizeConditions(value: unknown): DecisionCondition[] {
  const conditions = readConditions(value).map((condition) => ({
    type: condition.type.trim(),
    name: condition.name.trim(),
  }))
  if (conditions.some((condition) => !condition.type || !condition.name)) {
    throw new Error('Every composer condition needs both a type and a signal name.')
  }
  const keys = conditions.map((condition) =>
    `${condition.type}:${condition.name}`.toLocaleLowerCase(),
  )
  if (new Set(keys).size !== keys.length) {
    throw new Error('Composer conditions must be unique.')
  }
  return conditions
}

export function readSubjects(value: unknown): Subject[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return []
    const record = item as Record<string, unknown>
    if ((record.kind !== 'User' && record.kind !== 'Group') || typeof record.name !== 'string') {
      return []
    }
    return [{ kind: record.kind, name: record.name }]
  })
}

export function normalizeSubjects(value: unknown): Subject[] {
  const subjects = readSubjects(value).map((subject) => ({
    kind: subject.kind,
    name: subject.name.trim(),
  }))
  if (subjects.length === 0) {
    throw new Error('At least one subject is required for authz signals.')
  }
  if (subjects.some((subject) => !subject.name)) {
    throw new Error('Every authz subject needs a name.')
  }
  const keys = subjects.map((subject) => `${subject.kind}:${subject.name}`.toLocaleLowerCase())
  if (new Set(keys).size !== keys.length) {
    throw new Error('Authz subjects must be unique.')
  }
  return subjects
}

function readSequenceList(value: unknown): string[][] {
  if (!Array.isArray(value)) return []
  return value.map((sequence) => readStringList(sequence))
}

export function readStructureFeature(value: unknown): StructureFeature {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return structuredClone(DEFAULT_STRUCTURE_FEATURE)
  }
  const record = value as Record<string, unknown>
  const rawSource =
    record.source && typeof record.source === 'object' && !Array.isArray(record.source)
      ? (record.source as Record<string, unknown>)
      : {}
  const source: StructureSource = {
    type: typeof rawSource.type === 'string' ? rawSource.type : 'regex',
  }
  if (typeof rawSource.pattern === 'string') source.pattern = rawSource.pattern
  if (Array.isArray(rawSource.keywords)) source.keywords = readStringList(rawSource.keywords)
  if (typeof rawSource.case_sensitive === 'boolean')
    source.case_sensitive = rawSource.case_sensitive
  if (Array.isArray(rawSource.sequences)) source.sequences = readSequenceList(rawSource.sequences)
  return {
    type: typeof record.type === 'string' ? record.type : 'count',
    source,
  }
}

export function readStructurePredicate(value: unknown): NumericPredicate {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const record = value as Record<string, unknown>
  return Object.fromEntries(
    (['gt', 'gte', 'lt', 'lte'] as const)
      .filter((key) => typeof record[key] === 'number' && Number.isFinite(record[key]))
      .map((key) => [key, record[key] as number]),
  )
}

export function normalizeStructureFeature(value: unknown): StructureFeature {
  const feature = readStructureFeature(value)
  if (!STRUCTURE_FEATURE_TYPES.includes(feature.type as (typeof STRUCTURE_FEATURE_TYPES)[number])) {
    throw new Error(`Unsupported structure feature type "${feature.type}".`)
  }
  if (
    !STRUCTURE_SOURCE_TYPES.includes(feature.source.type as (typeof STRUCTURE_SOURCE_TYPES)[number])
  ) {
    throw new Error(`Unsupported structure source type "${feature.source.type}".`)
  }
  if (feature.type === 'sequence' && feature.source.type !== 'sequence') {
    throw new Error('Sequence features require a sequence source.')
  }
  if (feature.source.type === 'regex') {
    const pattern = feature.source.pattern?.trim()
    if (!pattern) throw new Error('A regex pattern is required for the structure source.')
    return {
      type: feature.type,
      source: {
        type: 'regex',
        pattern,
        ...(typeof feature.source.case_sensitive === 'boolean'
          ? { case_sensitive: feature.source.case_sensitive }
          : {}),
      },
    }
  }
  if (feature.source.type === 'keyword_set') {
    return {
      type: feature.type,
      source: {
        type: 'keyword_set',
        keywords: normalizeStringList(feature.source.keywords, 'Structure keywords', true),
        ...(typeof feature.source.case_sensitive === 'boolean'
          ? { case_sensitive: feature.source.case_sensitive }
          : {}),
      },
    }
  }

  const sequences = readSequenceList(feature.source.sequences)
  if (sequences.length === 0) throw new Error('At least one token sequence is required.')
  const normalizedSequences = sequences.map((sequence, index) =>
    normalizeStringList(sequence, `Sequence ${index + 1} tokens`, true),
  )
  if (normalizedSequences.some((sequence) => sequence.length < 2)) {
    throw new Error('Every structure sequence needs at least two tokens.')
  }
  return {
    type: feature.type,
    source: {
      type: 'sequence',
      sequences: normalizedSequences,
      ...(typeof feature.source.case_sensitive === 'boolean'
        ? { case_sensitive: feature.source.case_sensitive }
        : {}),
    },
  }
}

export function normalizeStructurePredicate(
  feature: StructureFeature,
  value: unknown,
): NumericPredicate | undefined {
  if (feature.type === 'exists') return undefined
  const predicate = readStructurePredicate(value)
  if (Object.keys(predicate).length === 0) return undefined
  if (predicate.gt !== undefined && predicate.gte !== undefined) {
    throw new Error('Structure predicate cannot set both gt and gte.')
  }
  if (predicate.lt !== undefined && predicate.lte !== undefined) {
    throw new Error('Structure predicate cannot set both lt and lte.')
  }
  return predicate
}

const SIGNAL_CONFIG_TYPES: Record<SignalType, string> = {
  Keywords: 'keyword',
  Embeddings: 'embedding',
  Domain: 'domain',
  Preference: 'preference',
  'Fact Check': 'fact_check',
  'User Feedback': 'user_feedback',
  Reask: 'reask',
  Language: 'language',
  Context: 'context',
  Structure: 'structure',
  Complexity: 'complexity',
  Modality: 'modality',
  Authz: 'authz',
  Jailbreak: 'jailbreak',
  PII: 'pii',
  KB: 'kb',
}

function countReferences(value: unknown, type: string, name: string): number {
  if (Array.isArray(value)) {
    return value.reduce((total, item) => total + countReferences(item, type, name), 0)
  }
  if (!value || typeof value !== 'object') return 0
  const record = value as Record<string, unknown>
  const ownMatch = record.type === type && record.name === name ? 1 : 0
  return (
    ownMatch +
    Object.values(record).reduce<number>(
      (total, item) => total + countReferences(item, type, name),
      0,
    )
  )
}

export function getSignalReferenceCount(
  config: ConfigData | null,
  signalType: SignalType,
  signalName: string,
): number {
  if (!config) return 0
  const type = SIGNAL_CONFIG_TYPES[signalType]
  return (
    countReferences(config.decisions, type, signalName) +
    countReferences(config.projections?.scores, type, signalName) +
    countReferences(
      config.signals?.complexity?.map((signal) => signal.composer),
      type,
      signalName,
    ) +
    countReferences(config.routing?.decisions, type, signalName) +
    countReferences(config.routing?.projections?.scores, type, signalName) +
    countReferences(
      config.routing?.signals?.complexity?.map((signal) => signal.composer),
      type,
      signalName,
    ) +
    countReferences(
      config.complexity_rules?.map((signal) => signal.composer),
      type,
      signalName,
    )
  )
}
