import type { FieldConfig } from '../components/EditModal'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import type { NormalizedModel, RoutingModelCard } from './configPageSupport'
import {
  effectiveModelCardFormData,
  modelReasoningFormData,
  normalizeModelLoras,
} from './configPageModelFormSupport'
import {
  getModelStructuredFormFields,
  modelAPIFormatField,
  modelReasoningFamilyField,
} from './configPageModelFormFields'

export const modelCardPatch = (data: Record<string, unknown>): Omit<RoutingModelCard, 'name'> => {
  const capabilities = normalizeStringList(data.capabilities)
  const tags = normalizeStringList(data.tags)
  const loras = normalizeModelLoras(data.loras)
  const input = normalizeStringList(data.input_modalities)
  const output = normalizeStringList(data.output_modalities)
  return {
    param_size:
      typeof data.param_size === 'string' && data.param_size.trim()
        ? data.param_size.trim()
        : undefined,
    context_window_size: data.context_window_size ? Number(data.context_window_size) : undefined,
    description:
      typeof data.description === 'string' && data.description.trim()
        ? data.description.trim()
        : undefined,
    capabilities: capabilities.length ? capabilities : undefined,
    loras: loras.length ? loras : undefined,
    tags: tags.length ? tags : undefined,
    // A model reads and writes text unless it declares otherwise.
    modalities:
      input.length || output.length
        ? { input: input.length ? input : ['text'], output: output.length ? output : ['text'] }
        : undefined,
  }
}

const addOnlyFields: FieldConfig[] = [
  {
    name: 'model_name',
    label: 'Model Name',
    section: 'Identity',
    type: 'text',
    required: true,
    placeholder: 'e.g., openai/gpt-4',
    description: 'Unique identifier for the model',
  },
]

const hiddenForCatalogModels = (data: Record<string, unknown>) =>
  Boolean(String(data.catalog || '').trim())

const inlineReasoningFields: FieldConfig[] = (
  [
    {
      name: 'reasoning_type',
      label: 'Inline Reasoning Type',
      type: 'select',
      options: [
        '',
        'reasoning_effort',
        'reasoning_mode',
        'chat_template_kwargs',
        'top_level_reasoning_effort',
      ],
      description: 'Use only when no built-in family matches a custom model.',
    },
    {
      name: 'reasoning_parameter',
      label: 'Inline Reasoning Parameter',
      type: 'text',
      placeholder: 'e.g., enable_thinking',
    },
    {
      name: 'reasoning_activation_parameter',
      label: 'Inline Reasoning Activation Parameter',
      type: 'text',
      placeholder: 'e.g., enable_thinking',
      description: 'Optional activation flag used alongside a reasoning-effort parameter.',
    },
    {
      name: 'reasoning_effort_flags',
      label: 'Inline Effort Flags',
      type: 'text',
      placeholder: 'low=low_effort, medium=medium_effort',
      description: 'Advanced: map effort names to boolean chat-template flags.',
    },
    {
      name: 'reasoning_levels',
      label: 'Inline Reasoning Levels',
      type: 'text',
      placeholder: 'low, medium, high',
    },
    {
      name: 'reasoning_default',
      label: 'Inline Reasoning Default',
      type: 'text',
      placeholder: 'medium',
    },
    {
      name: 'reasoning_modes',
      label: 'Supported Reasoning Modes',
      type: 'text',
      placeholder: 'adaptive, disabled',
      description: 'Optional mode controls independent from the effort ladder.',
    },
    {
      name: 'reasoning_default_mode',
      label: 'Default Reasoning Mode',
      type: 'select',
      options: ['', 'enabled', 'adaptive', 'disabled'],
    },
    {
      name: 'reasoning_disabled',
      label: 'Inline Reasoning Disabled Value',
      type: 'text',
      placeholder: 'none or disabled',
      description: 'Optional native level that explicitly disables reasoning.',
    },
  ] satisfies FieldConfig[]
).map((field) => ({ ...field, section: 'Reasoning', shouldHide: hiddenForCatalogModels }))

const reasoningFields = (
  reasoningFamilyNames: string[],
  catalog: BuiltInModelCatalog | null,
): FieldConfig[] => [
  modelReasoningFamilyField(reasoningFamilyNames, catalog),
  ...inlineReasoningFields,
]

const identityFields = (
  mode: 'add' | 'edit',
  catalog: BuiltInModelCatalog | null,
): FieldConfig[] => [
  {
    name: 'catalog',
    label: 'Built-in Catalog Model',
    section: 'Identity',
    type: 'text',
    placeholder: 'e.g., organization/model-id',
    description:
      mode === 'add'
        ? 'Optional. Leave empty for a custom or self-hosted model.'
        : 'Canonical card identity. Leave empty for a custom or self-hosted model.',
  },
  {
    name: 'provider_model_id',
    label: 'Provider Model ID',
    section: 'Identity',
    type: 'text',
    placeholder: 'e.g., openai/gpt-4.1',
    description:
      'Concrete upstream model identifier stored under providers.models[].provider_model_id',
  },
  modelAPIFormatField(catalog),
  {
    name: 'param_size',
    label: 'Parameter Size',
    section: 'Identity',
    type: 'text',
    placeholder: 'e.g., 8B',
  },
  {
    name: 'context_window_size',
    label: 'Context Window Size',
    section: 'Identity',
    type: 'number',
    placeholder: 'e.g., 131072',
  },
  {
    name: 'description',
    label: 'Description',
    section: 'Identity',
    type: 'textarea',
    placeholder: 'Short routing-facing model description',
  },
]

export const modelDialogFields = (
  reasoningFamilyNames: string[],
  mode: 'add' | 'edit',
  catalog: BuiltInModelCatalog | null = null,
): FieldConfig[] => [
  ...(mode === 'add' ? addOnlyFields : []),
  ...identityFields(mode, catalog),
  ...reasoningFields(reasoningFamilyNames, catalog),
  ...getModelStructuredFormFields(),
]

export const newModelFormData = (): Record<string, unknown> => ({
  model_name: '',
  catalog: '',
  ...modelReasoningFormData(),
  provider_model_id: '',
  api_format: '',
  external_model_ids: {},
  param_size: '',
  context_window_size: '',
  description: '',
  capabilities: [],
  input_modalities: [],
  output_modalities: [],
  loras: [],
  tags: [],
  backend_refs: [
    {
      name: 'endpoint-1',
      endpoint: 'localhost:8000',
      protocol: 'http',
      weight: 1,
      provider: 'vllm',
    },
  ],
})

export const editModelFormData = (model: NormalizedModel): Record<string, unknown> => ({
  model_name: model.name,
  catalog: model.catalog || '',
  ...modelReasoningFormData(model.reasoning),
  provider_model_id: model.provider_model_id || '',
  api_format: model.api_format_override || '',
  external_model_ids: model.external_model_ids || {},
  ...effectiveModelCardFormData(model),
  backend_refs: model.backend_refs || [],
  pricing: model.pricing || {},
  reliability: model.reliability || {},
})
