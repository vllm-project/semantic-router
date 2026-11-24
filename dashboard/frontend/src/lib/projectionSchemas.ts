import { ROUTER_CONFIG_EXTENSION } from '../generated/routerConfigContract'
import type { FieldSchema } from './dslSchemaTypes'
import { mergeRouterFieldSchemas, projectionFieldsFromRouterSchema } from './routerConfigSchema'

// Projection field schemas shared by the config managers and the Builder. The
// Router schema types these fields as strings; the options below are the
// values its projection validator accepts.

export const PROJECTION_INPUT_TYPES = ROUTER_CONFIG_EXTENSION.projection_input_types
export const PROJECTION_PARTITION_SEMANTICS = ['exclusive', 'softmax_exclusive'] as const
export const PROJECTION_SCORE_METHODS = ['weighted_sum'] as const
export const PROJECTION_MAPPING_METHODS = ['threshold_bands', 'multi_emit'] as const
export const PROJECTION_CALIBRATION_METHODS = ['sigmoid_distance'] as const
export const PROJECTION_SIGNAL_VALUE_SOURCES = ['binary', 'confidence', 'raw'] as const
export const PROJECTION_PROJECTION_VALUE_SOURCES = ['score', 'confidence'] as const
export const PROJECTION_BOUND_KEYS = ['gt', 'gte', 'lt', 'lte'] as const

export type ProjectionCollection = 'partitions' | 'scores' | 'mappings'

const PARTITION_OVERRIDES: FieldSchema[] = [
  { key: 'name', label: 'Name', type: 'string', required: true },
  {
    key: 'semantics',
    label: 'Semantics',
    type: 'select',
    required: true,
    options: [...PROJECTION_PARTITION_SEMANTICS],
    description:
      'exclusive keeps the strongest member; softmax_exclusive also spreads confidence by temperature.',
  },
  {
    key: 'temperature',
    label: 'Temperature',
    type: 'number',
    min: 0,
    description: 'Required by softmax_exclusive; lower is sharper.',
  },
  {
    key: 'members',
    label: 'Members',
    type: 'string[]',
    required: true,
    description: 'Declared domain signals coordinated by this partition.',
  },
  {
    key: 'default',
    label: 'Default',
    type: 'string',
    required: true,
    description: 'Member that wins when no member matches.',
  },
]

const SCORE_OVERRIDES: FieldSchema[] = [
  { key: 'name', label: 'Name', type: 'string', required: true },
  {
    key: 'method',
    label: 'Method',
    type: 'select',
    required: true,
    options: [...PROJECTION_SCORE_METHODS],
  },
  {
    key: 'inputs',
    label: 'Inputs',
    type: 'object[]',
    required: true,
    description: 'Weighted signal, knowledge-base metric, or earlier projection contributions.',
    itemLabel: 'Input',
    itemLabelKey: 'name',
    fields: [
      {
        key: 'type',
        label: 'Type',
        type: 'select',
        required: true,
        options: [...PROJECTION_INPUT_TYPES],
      },
      { key: 'name', label: 'Name', type: 'string', required: true },
      { key: 'weight', label: 'Weight', type: 'number', required: true },
      {
        key: 'value_source',
        label: 'Value Source',
        type: 'select',
        options: [
          '',
          ...new Set([...PROJECTION_SIGNAL_VALUE_SOURCES, ...PROJECTION_PROJECTION_VALUE_SOURCES]),
        ],
        description:
          'Signals: binary (default), confidence or raw. Projections: score or confidence.',
      },
      {
        key: 'match',
        label: 'Match Value',
        type: 'number',
        description: 'Value a matched binary input contributes.',
      },
      {
        key: 'miss',
        label: 'Miss Value',
        type: 'number',
        description: 'Value an unmatched binary input contributes.',
      },
    ],
  },
]

const MAPPING_OVERRIDES: FieldSchema[] = [
  { key: 'name', label: 'Name', type: 'string', required: true },
  {
    key: 'source',
    label: 'Source Score',
    type: 'string',
    required: true,
    description: 'Projection score whose output is mapped into named routing bands.',
  },
  {
    key: 'method',
    label: 'Method',
    type: 'select',
    required: true,
    options: [...PROJECTION_MAPPING_METHODS],
    description:
      'threshold_bands emits the one band the score falls in; multi_emit every band it satisfies.',
  },
  {
    key: 'calibration',
    label: 'Calibration',
    type: 'object',
    description: 'Optional confidence calibration applied to output bands.',
    fields: [
      {
        key: 'method',
        label: 'Method',
        type: 'select',
        options: ['', ...PROJECTION_CALIBRATION_METHODS],
      },
      { key: 'slope', label: 'Slope', type: 'number' },
    ],
  },
  {
    key: 'outputs',
    label: 'Outputs',
    type: 'object[]',
    required: true,
    description: 'Named routing bands and their lower or upper threshold bounds.',
    itemLabel: 'Band',
    itemLabelKey: 'name',
    fields: [
      { key: 'name', label: 'Name', type: 'string', required: true },
      { key: 'gt', label: 'Greater Than', type: 'number' },
      { key: 'gte', label: 'Greater Than or Equal', type: 'number' },
      { key: 'lt', label: 'Less Than', type: 'number' },
      { key: 'lte', label: 'Less Than or Equal', type: 'number' },
    ],
  },
]

const OVERRIDES: Record<ProjectionCollection, FieldSchema[]> = {
  partitions: PARTITION_OVERRIDES,
  scores: SCORE_OVERRIDES,
  mappings: MAPPING_OVERRIDES,
}

/** The fields of one projection collection entry: generated contract plus presentation. */
export function getProjectionFieldSchema(collection: ProjectionCollection): FieldSchema[] {
  return mergeRouterFieldSchemas(
    projectionFieldsFromRouterSchema(collection),
    OVERRIDES[collection],
  )
}

/** The value sources an input of this type accepts. */
export function projectionValueSources(inputType: string): readonly string[] {
  return inputType === 'projection'
    ? PROJECTION_PROJECTION_VALUE_SOURCES
    : PROJECTION_SIGNAL_VALUE_SOURCES
}
