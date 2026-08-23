import type { ProviderConnectionField, ProviderInterface } from '../utils/providerCatalogApi'
import ConfigPageProviderConnectionField from './ConfigPageProviderConnectionField'
import {
  initialProviderFieldValue,
  type EditableConnectionValue,
  type ExecutionFormValues,
  type PricingFormValues,
} from './configPageModelOnboardingSupport'
import styles from './ConfigPageAddModelsDialog.module.css'

interface Props {
  interfaces: ProviderInterface[]
  interfaceId: string
  onInterfaceId: (value: string) => void
  connectionFields: ProviderConnectionField[]
  connectionValues: Record<string, EditableConnectionValue>
  onConnectionValue: (name: string, value: EditableConnectionValue) => void
  namePrefix: string
  onNamePrefix: (value: string) => void
  execution: ExecutionFormValues
  onExecution: (field: keyof ExecutionFormValues, value: string) => void
  pricing: PricingFormValues
  onPricing: (field: keyof PricingFormValues, value: string) => void
}

export default function ConfigPageModelAdvancedOptions({
  interfaces,
  interfaceId,
  onInterfaceId,
  connectionFields,
  connectionValues,
  onConnectionValue,
  namePrefix,
  onNamePrefix,
  execution,
  onExecution,
  pricing,
  onPricing,
}: Props) {
  return (
    <details className={styles.advanced}>
      <summary>
        <span>Advanced</span>
        <small>Connection, execution, and pricing</small>
      </summary>
      <div className={styles.advancedContent}>
        {interfaces.length > 1 || connectionFields.length > 0 ? (
          <section className={styles.advancedSection}>
            <div className={styles.advancedHeading}>
              <strong>Connection</strong>
              <span>Optional settings exposed by this provider.</span>
            </div>
            <div className={styles.advancedGrid}>
              {interfaces.length > 1 ? (
                <label className={styles.field}>
                  <span>API style</span>
                  <select
                    value={interfaceId}
                    onChange={(event) => onInterfaceId(event.target.value)}
                  >
                    {interfaces.map((providerInterface) => (
                      <option key={providerInterface.id} value={providerInterface.id}>
                        {providerInterface.label}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
              {connectionFields.map((field) => (
                <ConfigPageProviderConnectionField
                  key={field.name}
                  field={field}
                  value={connectionValues[field.name] ?? initialProviderFieldValue(field)}
                  onChange={(value) => onConnectionValue(field.name, value)}
                />
              ))}
            </div>
          </section>
        ) : null}
        <section className={styles.advancedSection}>
          <div className={styles.advancedHeading}>
            <strong>Model names</strong>
            <span>Add a namespace to every imported model.</span>
          </div>
          <div className={styles.advancedGrid}>
            <label className={`${styles.field} ${styles.fullField}`}>
              <span>
                Name prefix <small>Optional</small>
              </span>
              <input
                value={namePrefix}
                onChange={(event) => onNamePrefix(event.target.value)}
                placeholder="team or environment"
              />
            </label>
          </div>
        </section>
        <section className={styles.advancedSection}>
          <div className={styles.advancedHeading}>
            <strong>Execution</strong>
            <span>Override only when this model needs it.</span>
          </div>
          <div className={styles.advancedGrid}>
            <label className={styles.field}>
              <span>
                Max retries <small>Optional</small>
              </span>
              <input
                type="number"
                min="0"
                max="5"
                step="1"
                value={execution.maxRetries}
                onChange={(event) => onExecution('maxRetries', event.target.value)}
                placeholder="Default"
              />
            </label>
            <label className={styles.field}>
              <span>
                Request timeout <small>Optional</small>
              </span>
              <input
                value={execution.requestTimeout}
                onChange={(event) => onExecution('requestTimeout', event.target.value)}
                placeholder="5m"
              />
            </label>
            <label className={styles.field}>
              <span>
                Stream timeout <small>Optional</small>
              </span>
              <input
                value={execution.streamTimeout}
                onChange={(event) => onExecution('streamTimeout', event.target.value)}
                placeholder="5m"
              />
            </label>
          </div>
        </section>
        <section className={styles.advancedSection}>
          <div className={styles.advancedHeading}>
            <strong>Pricing</strong>
            <span>Cost per one million tokens.</span>
          </div>
          <div className={`${styles.advancedGrid} ${styles.pricingGrid}`}>
            <PricingInput
              label="Input cost"
              value={pricing.inputCost}
              placeholder="0.25"
              onChange={(value) => onPricing('inputCost', value)}
            />
            <PricingInput
              label="Output cost"
              value={pricing.outputCost}
              placeholder="1.00"
              onChange={(value) => onPricing('outputCost', value)}
            />
            <PricingInput
              label="Cache read cost"
              value={pricing.cacheReadCost}
              placeholder="Defaults to input cost"
              onChange={(value) => onPricing('cacheReadCost', value)}
            />
            <PricingInput
              label="Cache write cost"
              value={pricing.cacheWriteCost}
              placeholder="Defaults to input cost"
              onChange={(value) => onPricing('cacheWriteCost', value)}
            />
          </div>
        </section>
      </div>
    </details>
  )
}

function PricingInput({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string
  value: string
  placeholder: string
  onChange: (value: string) => void
}) {
  return (
    <label className={styles.field}>
      <span>
        {label} <small>Optional</small>
      </span>
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        inputMode="decimal"
      />
    </label>
  )
}
