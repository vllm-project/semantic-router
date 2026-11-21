import NativeSelect from '../components/NativeSelect'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { Input } from '../ui'
import { effectiveModelAPIFormat, modelAPIFormats } from './configPageModelCatalogSupport'
import { normalizeModelBackendRefs, normalizeModelStringMap } from './configPageModelFormSupport'
import type { FieldConfig } from '../components/EditModal'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  ModelExternalIdsEditor,
  ModelLorasEditor,
  ModelModalitiesEditor,
  ModelPricingEditor,
  ModelReliabilityEditor,
  ModelTagsEditor,
} from './configPageModelStructuredEditors'

export function getModelStructuredFormFields(): FieldConfig[] {
  return [
    {
      name: 'capabilities',
      label: 'Capabilities',
      section: 'Routing metadata',
      type: 'custom',
      description: 'Capabilities exposed to model selection and routing policies.',
      customRender: (value, onChange) => (
        <ModelCapabilitiesEditor value={value} onChange={onChange} />
      ),
    },
    {
      name: 'input_modalities',
      label: 'Input Modalities',
      section: 'Routing metadata',
      type: 'custom',
      fullWidth: false,
      description:
        'Media the model reads (text, image, audio, video), stored under providers.models[].modalities.input.',
      customRender: (value, onChange) => (
        <ModelModalitiesEditor label="Input modalities" value={value} onChange={onChange} />
      ),
    },
    {
      name: 'output_modalities',
      label: 'Output Modalities',
      section: 'Routing metadata',
      type: 'custom',
      fullWidth: false,
      description:
        "Media the model writes, stored under providers.models[].modalities.output; a decision's output contract needs them.",
      customRender: (value, onChange) => (
        <ModelModalitiesEditor label="Output modalities" value={value} onChange={onChange} />
      ),
    },
    {
      name: 'tags',
      label: 'Tags',
      section: 'Routing metadata',
      type: 'custom',
      description: 'Structured routing labels used by filters, policies, and inventory search.',
      customRender: (value, onChange) => <ModelTagsEditor value={value} onChange={onChange} />,
    },
    {
      name: 'loras',
      label: 'LoRA Adapters',
      section: 'Routing metadata',
      type: 'custom',
      customRender: (value, onChange) => <ModelLorasEditor value={value} onChange={onChange} />,
    },
    {
      name: 'backend_refs',
      label: 'Provider Backends',
      section: 'Backends',
      type: 'custom',
      description: 'Physical inference targets stored under providers.models[].backend_refs.',
      customRender: (value, onChange) => (
        <ModelBackendRefsEditor value={value} onChange={onChange} />
      ),
    },
    {
      name: 'external_model_ids',
      label: 'External Model IDs',
      section: 'Backends',
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
      section: 'Pricing',
      type: 'custom',
      description: 'Per-million-token rates stored under providers.models[].pricing.',
      customRender: (value, onChange) => <ModelPricingEditor value={value} onChange={onChange} />,
    },
    {
      name: 'reliability',
      label: 'Delivery Policy',
      section: 'Delivery',
      type: 'custom',
      description: 'Retry, health check, and load-balancing controls for this model.',
      customRender: (value, onChange) => (
        <ModelReliabilityEditor value={value} onChange={onChange} />
      ),
    },
  ]
}

export function modelAPIFormatField(catalog: BuiltInModelCatalog | null): FieldConfig {
  return {
    name: 'api_format',
    label: 'API Format',
    section: 'Identity',
    type: 'custom',
    fullWidth: false,
    description: 'Use the provider/model default or choose an explicit wire format.',
    customRender: (value, onChange, data = {}) => {
      const model = {
        name:
          typeof data.model_name === 'string' && data.model_name.trim()
            ? data.model_name.trim()
            : 'custom-model',
        catalog: typeof data.catalog === 'string' ? data.catalog.trim() || undefined : undefined,
        provider_model_id:
          typeof data.provider_model_id === 'string'
            ? data.provider_model_id.trim() || undefined
            : undefined,
        external_model_ids: normalizeModelStringMap(data.external_model_ids),
        backend_refs: normalizeModelBackendRefs(data.backend_refs),
      }
      const inherited = effectiveModelAPIFormat(model, catalog)
      const explicit = typeof value === 'string' ? value : ''
      const effective = effectiveModelAPIFormat(
        { ...model, api_format: explicit || undefined },
        catalog,
      )
      return (
        <div className="flex flex-col gap-1.5">
          <NativeSelect
            aria-label="API Format"
            value={explicit}
            onChange={(event) => onChange(event.target.value)}
          >
            <option value="">
              {inherited.format ? `Inherit (${inherited.format})` : 'Inherit (unresolved)'}
            </option>
            {modelAPIFormats.map((format) => (
              <option key={format} value={format}>
                {format}
              </option>
            ))}
          </NativeSelect>
          {effective.error ? (
            <small role="status" className="text-xs text-warning-foreground">
              {effective.error}
            </small>
          ) : null}
        </div>
      )
    },
  }
}

export function modelReasoningFamilyField(
  names: string[],
  catalog: BuiltInModelCatalog | null,
): FieldConfig {
  return {
    name: 'reasoning_family',
    label: 'Reasoning Family',
    section: 'Reasoning',
    type: 'custom',
    fullWidth: false,
    description:
      'Built-in models inherit reasoning from their catalog. Custom models can select a family or use inline settings.',
    customRender: (value, onChange, data = {}) => {
      const catalogID = typeof data.catalog === 'string' ? data.catalog.trim() : ''
      if (catalogID) {
        const builtIn = catalog?.models.find((model) => model.id === catalogID)
        const label = builtIn ? builtIn.reasoning_family || 'None' : 'Unknown catalog model'
        return <Input aria-label="Reasoning Family" readOnly value={`${label} (inherited)`} />
      }
      return (
        <NativeSelect
          aria-label="Reasoning Family"
          value={typeof value === 'string' ? value : ''}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">None / inline settings</option>
          {names.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </NativeSelect>
      )
    },
  }
}
