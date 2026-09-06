import type { ReactNode } from 'react'

import ProductIcon from '../components/ProductIcon'
import type { ConnectModelAdvancedValues } from './configPageConnectModelSupport'
import styles from './ConfigPageConnectModelsDialog.module.css'

interface Props {
  value: ConnectModelAdvancedValues
  reasoningFamilies: string[]
  onChange: (value: ConnectModelAdvancedValues) => void
}

export default function ConfigPageConnectModelAdvanced({
  value,
  reasoningFamilies,
  onChange,
}: Props) {
  const update = <Key extends keyof ConnectModelAdvancedValues>(
    key: Key,
    next: ConnectModelAdvancedValues[Key],
  ) => onChange({ ...value, [key]: next })

  return (
    <details className={styles.advanced}>
      <summary>
        <span className={styles.advancedSummaryIcon} aria-hidden="true">
          <ProductIcon name="settings" />
        </span>
        <span>
          <strong>Advanced settings</strong>
          <small>Identity, routing metadata, cost, and delivery</small>
        </span>
        <ProductIcon className={styles.advancedChevron} name="chevron-down" aria-hidden="true" />
      </summary>

      <div className={styles.advancedContent}>
        <section className={styles.advancedSection}>
          <div className={styles.advancedHeading}>
            <strong>Identity</strong>
            <span>Keep imported models easy to recognize.</span>
          </div>
          <div className={styles.advancedGrid}>
            <Field label="Name prefix" hint="Optional">
              <input
                value={value.namePrefix}
                onChange={(event) => update('namePrefix', event.target.value)}
                placeholder="team or environment"
              />
            </Field>
            <Field label="Reasoning family" hint="Optional">
              <select
                value={value.reasoningFamily}
                onChange={(event) => update('reasoningFamily', event.target.value)}
              >
                <option value="">None</option>
                {reasoningFamilies.map((family) => (
                  <option key={family} value={family}>
                    {family}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Description" hint="Optional" fullWidth>
              <textarea
                value={value.description}
                onChange={(event) => update('description', event.target.value)}
                placeholder="What this model is best at"
              />
            </Field>
          </div>
        </section>

        <section className={styles.advancedSection}>
          <div className={styles.advancedHeading}>
            <strong>Routing metadata</strong>
            <span>Help recipes match work to this model.</span>
          </div>
          <div className={styles.advancedGrid}>
            <Field label="Modality" hint="Optional">
              <select
                value={value.modality}
                onChange={(event) => update('modality', event.target.value)}
              >
                <option value="">Not specified</option>
                <option value="ar">Text / autoregressive</option>
                <option value="omni">Omni</option>
                <option value="diffusion">Diffusion</option>
              </select>
            </Field>
            <Field label="Parameter size" hint="Optional">
              <input
                value={value.parameterSize}
                onChange={(event) => update('parameterSize', event.target.value)}
                placeholder="e.g. 32B"
              />
            </Field>
            <Field label="Context window" hint="Optional">
              <input
                type="number"
                min="1"
                value={value.contextWindow}
                onChange={(event) => update('contextWindow', event.target.value)}
                placeholder="e.g. 131072"
              />
            </Field>
            <Field label="Quality score" hint="0–1">
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={value.qualityScore}
                onChange={(event) => update('qualityScore', event.target.value)}
                placeholder="e.g. 0.9"
              />
            </Field>
            <Field label="Capabilities" hint="Comma separated" fullWidth>
              <input
                value={value.capabilities}
                onChange={(event) => update('capabilities', event.target.value)}
                placeholder="reasoning, coding, tools, vision"
              />
            </Field>
            <Field label="Tags" hint="Comma separated" fullWidth>
              <input
                value={value.tags}
                onChange={(event) => update('tags', event.target.value)}
                placeholder="fast, local, frontier"
              />
            </Field>
          </div>
        </section>

        <section className={styles.advancedSection}>
          <div className={styles.advancedHeading}>
            <strong>Pricing</strong>
            <span>Cost per one million tokens.</span>
          </div>
          <div className={styles.advancedGrid}>
            <CostField
              label="Input cost"
              value={value.inputCost}
              onChange={(next) => update('inputCost', next)}
            />
            <CostField
              label="Output cost"
              value={value.outputCost}
              onChange={(next) => update('outputCost', next)}
            />
            <CostField
              label="Cache read cost"
              value={value.cacheReadCost}
              placeholder="Defaults to input cost"
              onChange={(next) => update('cacheReadCost', next)}
            />
            <CostField
              label="Cache write cost"
              value={value.cacheWriteCost}
              placeholder="Defaults to input cost"
              onChange={(next) => update('cacheWriteCost', next)}
            />
          </div>
        </section>

        <section className={styles.advancedSection}>
          <div className={styles.advancedHeading}>
            <strong>Delivery</strong>
            <span>Override platform defaults only when needed.</span>
          </div>
          <div className={styles.advancedGrid}>
            <Field label="Max retries" hint="0–5">
              <input
                type="number"
                min="0"
                max="5"
                step="1"
                value={value.maxRetries}
                onChange={(event) => update('maxRetries', event.target.value)}
                placeholder="Platform default"
              />
            </Field>
            <Field label="Load balancing" hint="Optional">
              <select
                value={value.loadBalancing}
                onChange={(event) => update('loadBalancing', event.target.value)}
              >
                <option value="">Platform default</option>
                <option value="ROUND_ROBIN">Round robin</option>
                <option value="LEAST_REQUEST">Least requests</option>
                <option value="RING_HASH">Ring hash</option>
                <option value="MAGLEV">Maglev</option>
              </select>
            </Field>
            <Field label="Retry conditions" hint="Optional" fullWidth>
              <input
                value={value.retryOn}
                onChange={(event) => update('retryOn', event.target.value)}
                placeholder="5xx,reset,connect-failure"
              />
            </Field>
            <Field label="Health check path" hint="Optional">
              <input
                value={value.healthCheckPath}
                onChange={(event) => update('healthCheckPath', event.target.value)}
                placeholder="/health"
              />
            </Field>
            <Field label="Check interval" hint="Optional">
              <input
                value={value.healthCheckInterval}
                onChange={(event) => update('healthCheckInterval', event.target.value)}
                placeholder="10s"
              />
            </Field>
            <Field label="Check timeout" hint="Optional">
              <input
                value={value.healthCheckTimeout}
                onChange={(event) => update('healthCheckTimeout', event.target.value)}
                placeholder="2s"
              />
            </Field>
          </div>
        </section>
      </div>
    </details>
  )
}

function Field({
  label,
  hint,
  fullWidth = false,
  children,
}: {
  label: string
  hint?: string
  fullWidth?: boolean
  children: ReactNode
}) {
  return (
    <label className={`${styles.field} ${fullWidth ? styles.fullField : ''}`}>
      <span>
        {label} {hint ? <small>{hint}</small> : null}
      </span>
      {children}
    </label>
  )
}

function CostField({
  label,
  value,
  placeholder = '0.00',
  onChange,
}: {
  label: string
  value: string
  placeholder?: string
  onChange: (value: string) => void
}) {
  return (
    <Field label={label} hint="Optional">
      <div className={styles.costInput}>
        <span>$</span>
        <input
          inputMode="decimal"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
        />
      </div>
    </Field>
  )
}
