import type { FieldConfig } from '../components/EditModal'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  ModelExternalIdsEditor,
  ModelLorasEditor,
  ModelPricingEditor,
  ModelTagsEditor,
} from './configPageModelStructuredEditors'

export function getModelStructuredFormFields(): FieldConfig[] {
  return [
    {
      name: 'capabilities',
      label: 'Capabilities',
      type: 'custom',
      description: 'Capabilities exposed to model selection and routing policies.',
      customRender: (value, onChange) => (
        <ModelCapabilitiesEditor value={value} onChange={onChange} />
      ),
    },
    {
      name: 'tags',
      label: 'Tags',
      type: 'custom',
      description: 'Structured routing labels used by filters, policies, and inventory search.',
      customRender: (value, onChange) => <ModelTagsEditor value={value} onChange={onChange} />,
    },
    {
      name: 'quality_score',
      label: 'Quality Score',
      type: 'number',
      min: 0,
      max: 1,
      step: 0.01,
      placeholder: '0.85',
    },
    {
      name: 'loras',
      label: 'LoRA Adapters',
      type: 'custom',
      customRender: (value, onChange) => <ModelLorasEditor value={value} onChange={onChange} />,
    },
    {
      name: 'backend_refs',
      label: 'Provider Backends',
      type: 'custom',
      description: 'Physical inference targets stored under providers.models[].backend_refs.',
      customRender: (value, onChange) => (
        <ModelBackendRefsEditor value={value} onChange={onChange} />
      ),
    },
    {
      name: 'external_model_ids',
      label: 'External Model IDs',
      type: 'custom',
      description:
        'Provider-to-model ID aliases stored under providers.models[].external_model_ids.',
      customRender: (value, onChange) => (
        <ModelExternalIdsEditor value={value} onChange={onChange} />
      ),
    },
    {
      name: 'pricing',
      label: 'Token Pricing',
      type: 'custom',
      description: 'Per-million-token rates stored under providers.models[].pricing.',
      customRender: (value, onChange) => <ModelPricingEditor value={value} onChange={onChange} />,
    },
  ]
}
