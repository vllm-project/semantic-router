import { useState } from 'react'

import ProductIcon from '../ProductIcon'
import { EvaluationActionButton } from './EvaluationPrimitives'
import {
  copyActionLabel,
  copyTextToClipboard,
  type CopyState,
  type TechnicalField,
} from './evaluationTechnicalFields'
import styles from './EvaluationCampaignDecisionTechnicalFields.module.css'

export function CopyableValuePresentation({
  label,
  value,
  displayValue = value,
  copyState,
  onCopy,
}: {
  label: string
  value: string
  displayValue?: string
  copyState: CopyState
  onCopy: () => void
}) {
  const actionLabel = copyActionLabel(label, copyState)
  const statusMessage =
    copyState === 'copied'
      ? `${label} copied.`
      : copyState === 'failed'
        ? `${label} could not be copied.`
        : ''

  return (
    <dd className={styles.copyableValue} title={value}>
      <span>{displayValue}</span>
      <EvaluationActionButton
        type="button"
        compact
        variant="quiet"
        className={styles.copyButton}
        disabled={copyState === 'copying'}
        onClick={onCopy}
        title={actionLabel}
        aria-label={actionLabel}
      >
        <ProductIcon name={copyState === 'copied' ? 'check' : 'copy'} aria-hidden="true" />
      </EvaluationActionButton>
      <span className={styles.srOnly} aria-live="polite">
        {statusMessage}
      </span>
    </dd>
  )
}

export function CopyableValue({
  label,
  value,
  displayValue = value,
}: {
  label: string
  value: string
  displayValue?: string
}) {
  const [copyState, setCopyState] = useState<CopyState>('idle')

  const copy = async () => {
    setCopyState('copying')
    setCopyState(await copyTextToClipboard(value, navigator.clipboard))
  }

  return (
    <CopyableValuePresentation
      label={label}
      value={value}
      displayValue={displayValue}
      copyState={copyState}
      onCopy={() => void copy()}
    />
  )
}

export function TechnicalFieldGrid({ label, fields }: { label: string; fields: TechnicalField[] }) {
  return (
    <dl className={styles.grid} aria-label={label}>
      {fields.map((field) => (
        <div key={field.label}>
          <dt>{field.label}</dt>
          {field.copyable ? (
            <CopyableValue
              label={field.label.toLowerCase()}
              value={String(field.value)}
              displayValue={field.displayValue}
            />
          ) : (
            <dd className={field.mono ? styles.mono : undefined} title={String(field.value)}>
              {field.value}
            </dd>
          )}
        </div>
      ))}
    </dl>
  )
}
