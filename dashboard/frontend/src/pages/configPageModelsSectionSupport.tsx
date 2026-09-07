import type { ReactNode } from 'react'

import type { FieldConfig } from '../components/EditModal'
import { normalizeStringList } from '../components/structuredFieldEditorSupport'
import type { ViewSection } from '../components/ViewModal'
import type { NormalizedModel, RoutingModelCard } from './configPageSupport'
import {
  modelReasoningFormData,
  normalizeModelEvaluations,
  normalizeModelLoras,
} from './configPageModelFormSupport'
import { getModelStructuredFormFields } from './configPageModelFormFields'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  ModelEvaluationsEditor,
  ModelExternalIdsEditor,
  ModelLorasEditor,
  ModelPricingEditor,
  ModelReliabilityEditor,
  ModelTagsEditor,
} from './configPageModelStructuredEditors'

export const modelCardPatch = (data: Record<string, unknown>): Omit<RoutingModelCard, 'name'> => {
  const capabilities = normalizeStringList(data.capabilities)
  const tags = normalizeStringList(data.tags)
  const loras = normalizeModelLoras(data.loras)
  const evaluations = normalizeModelEvaluations(data.evaluations)
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
    evaluations: evaluations.length ? evaluations : undefined,
    modality:
      typeof data.modality === 'string' && data.modality.trim() ? data.modality.trim() : undefined,
  }
}

const addOnlyFields: FieldConfig[] = [
  {
    name: 'model_name',
    label: 'Model Name',
    type: 'text',
    required: true,
    placeholder: 'e.g., openai/gpt-4',
    description: 'Unique identifier for the model',
  },
]

const reasoningFields = (reasoningFamilyNames: string[]): FieldConfig[] => [
  {
    name: 'reasoning_family',
    label: 'Reasoning Family',
    type: 'select',
    options: reasoningFamilyNames,
    description: 'Optional for custom models. Built-in models inherit this automatically.',
  },
  {
    name: 'reasoning_type',
    label: 'Inline Reasoning Type',
    type: 'select',
    options: [
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
    options: ['enabled', 'adaptive', 'disabled'],
  },
  {
    name: 'reasoning_disabled',
    label: 'Inline Reasoning Disabled Value',
    type: 'text',
    placeholder: 'none or disabled',
    description: 'Optional native level that explicitly disables reasoning.',
  },
]

const identityFields = (mode: 'add' | 'edit'): FieldConfig[] => [
  {
    name: 'catalog',
    label: 'Built-in Catalog Model',
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
    type: 'text',
    placeholder: 'e.g., openai/gpt-4.1',
    description:
      'Concrete upstream model identifier stored under providers.models[].provider_model_id',
  },
  {
    name: 'api_format',
    label: 'API Format',
    type: 'text',
    placeholder: 'e.g., openai',
    description: 'Provider-specific wire format stored under providers.models[].api_format',
  },
  { name: 'param_size', label: 'Parameter Size', type: 'text', placeholder: 'e.g., 8B' },
  {
    name: 'context_window_size',
    label: 'Context Window Size',
    type: 'number',
    placeholder: 'e.g., 131072',
  },
  {
    name: 'modality',
    label: 'Modality',
    type: 'text',
    placeholder: 'e.g., text, omni, diffusion',
  },
  {
    name: 'description',
    label: 'Description',
    type: 'textarea',
    placeholder: 'Short routing-facing model description',
  },
]

export const modelDialogFields = (
  reasoningFamilyNames: string[],
  mode: 'add' | 'edit',
): FieldConfig[] => [
  ...(mode === 'add' ? addOnlyFields : []),
  ...identityFields(mode).slice(0, 1),
  ...reasoningFields(reasoningFamilyNames),
  ...identityFields(mode).slice(1),
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
  loras: [],
  tags: [],
  evaluations: [],
  modality: '',
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
  catalog: model.catalog || '',
  ...modelReasoningFormData(model.reasoning),
  provider_model_id: model.provider_model_id || '',
  api_format: model.api_format || '',
  external_model_ids: model.external_model_ids || {},
  param_size: model.card_override?.param_size || '',
  context_window_size: model.card_override?.context_window_size || '',
  description: model.card_override?.description || '',
  capabilities: model.card_override?.capabilities || [],
  loras: model.card_override?.loras || [],
  tags: model.card_override?.tags || [],
  evaluations: model.card_override?.evaluations || [],
  modality: model.card_override?.modality || '',
  backend_refs: model.backend_refs || [],
  pricing: model.pricing || {},
  reliability: model.reliability || {},
})

const baseModelViewSection = (model: NormalizedModel, defaultModel: string): ViewSection => ({
  title: 'Basic Information',
  fields: [
    { label: 'Model Name', value: model.name },
    { label: 'Catalog Model', value: model.catalog || 'Custom' },
    { label: 'Reasoning Family', value: model.reasoning_family || 'N/A' },
    { label: 'Is Default', value: model.name === defaultModel ? 'Yes' : 'No' },
    { label: 'Provider Model ID', value: model.provider_model_id || 'N/A' },
    { label: 'API Format', value: model.api_format || 'N/A' },
    { label: 'Modality', value: model.modality || 'N/A' },
    { label: 'Param Size', value: model.param_size || 'N/A' },
    {
      label: 'Context Window',
      value: model.context_window_size ? `${model.context_window_size}` : 'N/A',
    },
  ],
})

const modelRoutingMetadataSection = (model: NormalizedModel): ViewSection | null => {
  const present =
    model.description ||
    model.capabilities?.length ||
    model.tags?.length ||
    model.loras?.length ||
    model.evaluations?.length
  if (!present) return null
  return {
    title: 'Routing Metadata',
    fields: [
      { label: 'Description', value: model.description || 'N/A', fullWidth: true },
      {
        label: 'Capabilities',
        value: <ModelCapabilitiesEditor value={model.capabilities || []} readOnly />,
        fullWidth: true,
      },
      {
        label: 'Tags',
        value: <ModelTagsEditor value={model.tags || []} readOnly />,
        fullWidth: true,
      },
      {
        label: 'LoRAs',
        value: <ModelLorasEditor value={model.loras || []} readOnly />,
        fullWidth: true,
      },
      ...(model.evaluations?.length
        ? [
            {
              label: 'Operator Evaluations',
              value: <ModelEvaluationsEditor value={model.evaluations} readOnly />,
              fullWidth: true,
            },
          ]
        : []),
    ],
  }
}

const editorViewSection = (title: string, label: string, value: ReactNode): ViewSection => ({
  title,
  fields: [{ label, value, fullWidth: true }],
})

const appendModelInfrastructureSections = (
  sections: ViewSection[],
  model: NormalizedModel,
  isReadonly: boolean,
): void => {
  if (model.external_model_ids && Object.keys(model.external_model_ids).length) {
    sections.push(
      editorViewSection(
        'External Model IDs',
        'Provider IDs',
        <ModelExternalIdsEditor value={model.external_model_ids} readOnly />,
      ),
    )
  }
  if (model.backend_refs?.length) {
    sections.push(
      editorViewSection(
        `Provider Backends (${model.backend_refs.length})`,
        'Configured Backend Refs',
        <ModelBackendRefsEditor value={model.backend_refs} readOnly maskSensitive={isReadonly} />,
      ),
    )
  }
  if (model.pricing) {
    sections.push(
      editorViewSection(
        'Pricing',
        'Token Pricing',
        <ModelPricingEditor value={model.pricing} readOnly />,
      ),
    )
  }
  if (model.reliability) {
    sections.push(
      editorViewSection(
        'Delivery',
        'Policy',
        <ModelReliabilityEditor value={model.reliability} readOnly />,
      ),
    )
  }
}

export const modelViewSections = (
  model: NormalizedModel,
  defaultModel: string,
  isReadonly: boolean,
): ViewSection[] => {
  const sections = [baseModelViewSection(model, defaultModel)]
  const metadata = modelRoutingMetadataSection(model)
  if (metadata) sections.push(metadata)
  appendModelInfrastructureSections(sections, model, isReadonly)
  return sections
}
