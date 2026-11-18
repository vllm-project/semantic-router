import type { FieldSchema } from '@/lib/dslSchemaTypes'
import {
  getAlgorithmFieldSchema,
  getPluginFieldSchema,
  getSignalFieldSchema,
} from '@/lib/dslSchemas'
import { serializeFields } from '@/lib/dslMutations'
import { getProjectionFieldSchema, type ProjectionCollection } from '@/lib/projectionSchemas'
import { routerConfigFieldAtPath } from '@/lib/routerConfigSchema'
import { requiredFieldErrors } from '@/lib/schemaFieldValidation'
import type {
  ASTProjectionMappingDecl,
  ASTProjectionPartitionDecl,
  ASTProjectionScoreDecl,
  DSLFieldObject,
} from '@/types/dsl'

// The Builder's forms read the same field schemas and required-field validator
// as the config pages, so a field the contract adds or drops changes both.

export const signalFieldSchema = getSignalFieldSchema
export const pluginFieldSchema = getPluginFieldSchema
export const algorithmFieldSchema = getAlgorithmFieldSchema
export const fieldProblems = requiredFieldErrors

/** The fields a DSL MODEL block carries; the compiler maps them onto the provider model. */
export const DSL_MODEL_FIELD_KEYS = [
  'description',
  'param_size',
  'context_window_size',
  'max_output_tokens',
  'capabilities',
  'modalities',
  'loras',
  'tags',
] as const

/** The provider model contract, narrowed to the fields a MODEL block declares. */
export function modelFieldSchema(): FieldSchema[] {
  const fields = routerConfigFieldAtPath(['providers', 'models'])?.fields ?? []
  return DSL_MODEL_FIELD_KEYS.map((key) => fields.find((field) => field.key === key)).filter(
    (field): field is FieldSchema => field !== undefined,
  )
}

export type ProjectionDeclKind = 'partition' | 'score' | 'mapping'

export const PROJECTION_COLLECTION: Record<ProjectionDeclKind, ProjectionCollection> = {
  partition: 'partitions',
  score: 'scores',
  mapping: 'mappings',
}

/** A projection's fields as the Projections page edits them; the name lives in the block header. */
export function projectionFieldSchema(kind: ProjectionDeclKind): FieldSchema[] {
  return getProjectionFieldSchema(PROJECTION_COLLECTION[kind]).filter(
    (field) => field.key !== 'name',
  )
}

export function projectionPartitionToFields(partition: ASTProjectionPartitionDecl): DSLFieldObject {
  return {
    ...(partition.semantics ? { semantics: partition.semantics } : {}),
    ...(typeof partition.temperature === 'number' ? { temperature: partition.temperature } : {}),
    members: partition.members ?? [],
    ...(partition.default ? { default: partition.default } : {}),
  }
}

export function projectionScoreToFields(score: ASTProjectionScoreDecl): DSLFieldObject {
  return {
    ...(score.method ? { method: score.method } : {}),
    inputs: (score.inputs ?? []).map((input) => ({
      type: input.signalType,
      ...(input.signalName ? { name: input.signalName } : {}),
      weight: input.weight,
      ...(input.valueSource ? { value_source: input.valueSource } : {}),
      ...(typeof input.match === 'number' ? { match: input.match } : {}),
      ...(typeof input.miss === 'number' ? { miss: input.miss } : {}),
    })),
  }
}

export function projectionMappingToFields(mapping: ASTProjectionMappingDecl): DSLFieldObject {
  return {
    ...(mapping.source ? { source: mapping.source } : {}),
    ...(mapping.method ? { method: mapping.method } : {}),
    ...(mapping.calibration
      ? {
          calibration: {
            ...(mapping.calibration.method ? { method: mapping.calibration.method } : {}),
            ...(typeof mapping.calibration.slope === 'number'
              ? { slope: mapping.calibration.slope }
              : {}),
          },
        }
      : {}),
    outputs: (mapping.outputs ?? []).map((output) => ({
      name: output.name,
      ...(typeof output.lt === 'number' ? { lt: output.lt } : {}),
      ...(typeof output.lte === 'number' ? { lte: output.lte } : {}),
      ...(typeof output.gt === 'number' ? { gt: output.gt } : {}),
      ...(typeof output.gte === 'number' ? { gte: output.gte } : {}),
    })),
  }
}

/** Starting fields of a new projection, valid enough to compile once members or inputs are added. */
export const NEW_PROJECTION_FIELDS: Record<ProjectionDeclKind, DSLFieldObject> = {
  partition: { semantics: 'exclusive', members: [] },
  score: { method: 'weighted_sum', inputs: [] },
  mapping: { method: 'threshold_bands', outputs: [] },
}

export function projectionFields(
  kind: ProjectionDeclKind,
  entity: ASTProjectionPartitionDecl | ASTProjectionScoreDecl | ASTProjectionMappingDecl,
): DSLFieldObject {
  if (kind === 'partition') return projectionPartitionToFields(entity as ASTProjectionPartitionDecl)
  if (kind === 'score') return projectionScoreToFields(entity as ASTProjectionScoreDecl)
  return projectionMappingToFields(entity as ASTProjectionMappingDecl)
}

/** The DSL block a form writes, for its preview. */
export function dslBlockPreview(header: string, fields: DSLFieldObject): string {
  const body = serializeFields(fields)
  return body.trim() ? `${header} {\n${body}\n}` : `${header} {}`
}

/** New names follow the old Builder: surrounding space is trimmed and inner spaces become underscores. */
export function normalizeEntityName(name: string): string {
  return name.trim().replace(/\s+/g, '_')
}
