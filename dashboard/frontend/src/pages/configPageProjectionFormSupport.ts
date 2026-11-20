import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import {
  PROJECTION_CALIBRATION_METHODS,
  PROJECTION_INPUT_TYPES,
  PROJECTION_PARTITION_SEMANTICS,
  projectionValueSources,
} from '../lib/projectionSchemas'
import type {
  ProjectionMappingCalibration,
  ProjectionMappingOutput,
  ProjectionScoreInput,
} from './configPageSupport'

const supportedProjectionInputTypes = new Set<string>(PROJECTION_INPUT_TYPES)

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function listPhrase(values: readonly string[]): string {
  return values.length <= 1
    ? values.join('')
    : `${values.slice(0, -1).join(', ')}${values.length > 2 ? ',' : ''} or ${values[values.length - 1]}`
}

export function normalizeProjectionMembers(value: unknown): string[] {
  return normalizeStringList(value)
}

export function assertProjectionMembers(members: string[], defaultMember: string): void {
  if (members.length === 0) throw new Error('Members must include at least one signal.')
  if (!defaultMember.trim()) throw new Error('Default member is required.')
  if (!members.includes(defaultMember.trim())) {
    throw new Error('Default member must also appear in Members.')
  }
}

export function assertProjectionPartitionSettings(
  semantics: string,
  temperature: number | undefined,
): void {
  if (!(PROJECTION_PARTITION_SEMANTICS as readonly string[]).includes(semantics)) {
    throw new Error(`Partition semantics must be ${listPhrase(PROJECTION_PARTITION_SEMANTICS)}.`)
  }
  if (
    semantics === 'softmax_exclusive' &&
    (temperature === undefined || !Number.isFinite(temperature) || temperature <= 0)
  ) {
    throw new Error('Softmax-exclusive partitions require a temperature greater than zero.')
  }
}

export function normalizeProjectionInputs(value: unknown): ProjectionScoreInput[] {
  if (!Array.isArray(value)) return []

  return value.filter(isRecord).map((entry) => ({
    type: optionalString(entry.type) || '',
    name: optionalString(entry.name),
    weight: typeof entry.weight === 'number' ? entry.weight : Number.NaN,
    value_source: optionalString(entry.value_source),
    match: optionalNumber(entry.match),
    miss: optionalNumber(entry.miss),
  }))
}

export function projectionInputErrors(input: ProjectionScoreInput): string[] {
  const errors: string[] = []
  if (!input.type.trim()) errors.push('Input type is required.')
  else if (!supportedProjectionInputTypes.has(input.type))
    errors.push('Input type is not supported.')
  if (!input.name?.trim()) errors.push('Signal or projection name is required.')
  if (!Number.isFinite(input.weight)) errors.push('Weight must be a number.')
  const sources = projectionValueSources(input.type)
  if (input.value_source && !sources.includes(input.value_source)) {
    errors.push(
      input.type === 'projection'
        ? `Projection inputs support ${listPhrase(sources)} values.`
        : `Signal inputs support ${listPhrase(sources)} values.`,
    )
  }
  return errors
}

export function assertProjectionInputs(inputs: ProjectionScoreInput[]): void {
  if (inputs.length === 0) throw new Error('Inputs must include at least one contribution.')
  inputs.forEach((input, index) => {
    const error = projectionInputErrors(input)[0]
    if (error) throw new Error(`Input ${index + 1}: ${error}`)
  })
}

export function normalizeProjectionCalibration(
  value: unknown,
): ProjectionMappingCalibration | undefined {
  if (!isRecord(value)) return undefined
  return {
    method: optionalString(value.method) || '',
    slope: optionalNumber(value.slope),
  }
}

export function assertProjectionCalibration(
  calibration: ProjectionMappingCalibration | undefined,
): void {
  if (!calibration) return
  if (
    calibration.method &&
    !(PROJECTION_CALIBRATION_METHODS as readonly string[]).includes(calibration.method)
  ) {
    throw new Error(`Calibration method must be ${listPhrase(PROJECTION_CALIBRATION_METHODS)}.`)
  }
  if (calibration.slope !== undefined && !Number.isFinite(calibration.slope)) {
    throw new Error('Calibration slope must be a number.')
  }
}

export function normalizeProjectionOutputs(value: unknown): ProjectionMappingOutput[] {
  if (!Array.isArray(value)) return []

  return value.filter(isRecord).map((entry) => ({
    name: optionalString(entry.name) || '',
    lt: optionalNumber(entry.lt),
    lte: optionalNumber(entry.lte),
    gt: optionalNumber(entry.gt),
    gte: optionalNumber(entry.gte),
  }))
}

export function projectionOutputErrors(output: ProjectionMappingOutput): string[] {
  const errors: string[] = []
  if (!output.name.trim()) errors.push('Output name is required.')
  if (
    output.gt === undefined &&
    output.gte === undefined &&
    output.lt === undefined &&
    output.lte === undefined
  ) {
    errors.push('Add at least one threshold bound.')
  }
  if (output.gt !== undefined && output.gte !== undefined)
    errors.push('Choose either gt or gte, not both.')
  if (output.lt !== undefined && output.lte !== undefined)
    errors.push('Choose either lt or lte, not both.')
  return errors
}

export function assertProjectionOutputs(outputs: ProjectionMappingOutput[]): void {
  if (outputs.length === 0) throw new Error('Outputs must include at least one routing band.')
  const names = new Set<string>()
  outputs.forEach((output, index) => {
    const error = projectionOutputErrors(output)[0]
    if (error) throw new Error(`Output ${index + 1}: ${error}`)
    if (names.has(output.name)) throw new Error(`Output ${index + 1}: output name must be unique.`)
    names.add(output.name)
  })
}

export function assertProjectionMappingOutputCount(
  method: string,
  outputs: ProjectionMappingOutput[],
): void {
  if (method === 'multi_emit' && outputs.length < 2) {
    throw new Error('Multi-emit mappings require at least two outputs.')
  }
}
