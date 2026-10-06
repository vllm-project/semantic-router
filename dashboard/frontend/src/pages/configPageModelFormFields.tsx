import styles from '../components/EditModal.module.css'
import type { BuiltInModelCatalog } from '../types/modelCatalog'
import { effectiveModelAPIFormat, modelAPIFormats } from './configPageModelCatalogSupport'
import { normalizeModelBackendRefs, normalizeModelStringMap } from './configPageModelFormSupport'
import type { FieldConfig } from '../components/EditModal'
import {
  ModelBackendRefsEditor,
  ModelCapabilitiesEditor,
  ModelExternalIdsEditor,
  ModelLorasEditor,
  ModelPricingEditor,
  ModelReliabilityEditor,
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
    {
      name: 'reliability',
      label: 'Delivery Policy',
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
        <>
          <select
            aria-label="API Format"
            className={styles.select}
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
          </select>
          {effective.error && <small role="status">{effective.error}</small>}
        </>
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
    type: 'custom',
    fullWidth: false,
    description:
      'Built-in models inherit reasoning from their catalog. Custom models can select a family or use inline settings.',
    customRender: (value, onChange, data = {}) => {
      const catalogID = typeof data.catalog === 'string' ? data.catalog.trim() : ''
      if (catalogID) {
        const builtIn = catalog?.models.find((model) => model.id === catalogID)
        const label = builtIn ? builtIn.reasoning_family || 'None' : 'Unknown catalog model'
        return (
          <input
            aria-label="Reasoning Family"
            className={styles.input}
            readOnly
            value={`${label} (inherited)`}
          />
        )
      }
      return (
        <select
          aria-label="Reasoning Family"
          className={styles.select}
          value={typeof value === 'string' ? value : ''}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">None / inline settings</option>
          {names.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      )
    },
  }
}
