import { cloneConfigData } from './configPageCanonicalization'
import {
  assertProjectionCalibration,
  assertProjectionInputs,
  assertProjectionMappingOutputCount,
  assertProjectionMembers,
  assertProjectionOutputs,
  assertProjectionPartitionSettings,
  normalizeProjectionCalibration,
  normalizeProjectionInputs,
  normalizeProjectionMembers,
  normalizeProjectionOutputs,
} from './configPageProjectionFormSupport'
import {
  PROJECTION_COLLECTION,
  ensureProjectionConfig,
  type ProjectionKind,
} from './configPageProjectionTableSupport'
import type {
  ConfigData,
  ProjectionMapping,
  ProjectionPartition,
  ProjectionScore,
} from './configPageSupport'

export type ProjectionEntity = ProjectionPartition | ProjectionScore | ProjectionMapping

export function newProjectionDraft(
  kind: ProjectionKind,
  scoreNames: readonly string[] = [],
): ProjectionEntity {
  if (kind === 'partition') return { name: '', semantics: 'exclusive', members: [], default: '' }
  if (kind === 'score') return { name: '', method: 'weighted_sum', inputs: [] }
  return { name: '', source: scoreNames[0] ?? '', method: 'threshold_bands', outputs: [] }
}

/**
 * The projection a draft saves as. Normalizes members, inputs, calibration and
 * outputs the way the earlier forms did, and keeps every other field as written.
 */
export function projectionFromDraft(
  kind: ProjectionKind,
  draft: ProjectionEntity,
): ProjectionEntity {
  const name = (draft.name || '').trim()
  if (!name) throw new Error('Name is required.')
  if (kind === 'partition') {
    const partition = draft as ProjectionPartition
    const members = normalizeProjectionMembers(partition.members)
    const defaultMember = partition.default?.trim() ?? ''
    assertProjectionMembers(members, defaultMember)
    assertProjectionPartitionSettings(partition.semantics, partition.temperature)
    return { ...partition, name, members, default: defaultMember }
  }
  if (kind === 'score') {
    const score = draft as ProjectionScore
    const inputs = normalizeProjectionInputs(score.inputs)
    assertProjectionInputs(inputs)
    return { ...score, name, inputs }
  }
  const mapping = draft as ProjectionMapping
  if (!mapping.source?.trim()) throw new Error('Source score is required.')
  const calibration = normalizeProjectionCalibration(mapping.calibration)
  const outputs = normalizeProjectionOutputs(mapping.outputs)
  assertProjectionCalibration(calibration)
  assertProjectionOutputs(outputs)
  assertProjectionMappingOutputCount(mapping.method, outputs)
  const next: ProjectionMapping = { ...mapping, name, source: mapping.source.trim(), outputs }
  if (calibration) next.calibration = calibration
  else delete next.calibration
  return next
}

/** Writes a projection into a copy of the profile, in place when it already exists. */
export function upsertProjection(
  config: ConfigData,
  kind: ProjectionKind,
  entity: ProjectionEntity,
  originalName?: string,
): ConfigData {
  const next = cloneConfigData(config)
  const projections = ensureProjectionConfig(next)
  const collection = PROJECTION_COLLECTION[kind]
  const entries = [...((projections[collection] ?? []) as ProjectionEntity[])]
  if (entries.some((entry) => entry.name === entity.name && entry.name !== originalName)) {
    throw new Error(`A ${kind} named “${entity.name}” already exists.`)
  }
  const index =
    originalName === undefined ? -1 : entries.findIndex((entry) => entry.name === originalName)
  if (index >= 0) entries[index] = entity
  else entries.push(entity)
  ;(projections as Record<string, unknown>)[collection] = entries
  return next
}

export function removeProjection(
  config: ConfigData,
  kind: ProjectionKind,
  name: string,
): ConfigData {
  const next = cloneConfigData(config)
  const projections = ensureProjectionConfig(next)
  const collection = PROJECTION_COLLECTION[kind]
  ;(projections as Record<string, unknown>)[collection] = (
    (projections[collection] ?? []) as ProjectionEntity[]
  ).filter((entry) => entry.name !== name)
  return next
}
