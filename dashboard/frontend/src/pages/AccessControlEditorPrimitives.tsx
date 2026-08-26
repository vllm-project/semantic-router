import { type PropsWithChildren, useId } from 'react'

import ProductIcon from '../components/ProductIcon'
import styles from './AccessControlPage.module.css'

export function Advanced({ children, label = 'Advanced' }: PropsWithChildren<{ label?: string }>) {
  return (
    <details className={styles.advancedSection}>
      <summary>
        <span>{label}</span>
        <small>Optional settings</small>
      </summary>
      <div className={styles.advancedGrid}>{children}</div>
    </details>
  )
}

export function Field({
  label,
  hint,
  wide = false,
  children,
}: PropsWithChildren<{ label: string; hint?: string; wide?: boolean }>) {
  return (
    <label className={`${styles.formField} ${wide ? styles.formFieldWide : ''}`}>
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  )
}

export function PickerField({
  label,
  hint,
  children,
}: PropsWithChildren<{ label: string; hint?: string }>) {
  return (
    <div className={`${styles.formField} ${styles.formFieldWide}`}>
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </div>
  )
}

export function StatusField({
  value,
  onChange,
}: {
  value: string
  onChange: (value: 'active' | 'disabled') => void
}) {
  return (
    <Field label="Status">
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as 'active' | 'disabled')}
      >
        <option value="active">Active</option>
        <option value="disabled">Disabled</option>
      </select>
    </Field>
  )
}

export function CoreSection({
  title,
  detail,
  children,
}: PropsWithChildren<{ title: string; detail: string }>) {
  const titleId = `access-selection-${useId().replace(/:/g, '')}`
  return (
    <section
      className={styles.selectionSection}
      role="group"
      aria-labelledby={titleId}
      data-access-section="core"
    >
      <header className={styles.selectionSectionHeader}>
        <span id={titleId}>{title}</span>
        <small>{detail}</small>
      </header>
      <div className={styles.choiceGrid}>{children}</div>
    </section>
  )
}

export function OwnerChoice({
  active,
  disabled = false,
  title,
  detail,
  onSelect,
}: {
  active: boolean
  disabled?: boolean
  title: string
  detail: string
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      className={`${styles.ownerChoice} ${active ? styles.ownerChoiceActive : ''}`}
      role="radio"
      aria-checked={active}
      disabled={disabled}
      onClick={onSelect}
    >
      <span>{title}</span>
      <small>{detail}</small>
      {active ? (
        <ProductIcon className={styles.choiceCheck} name="check" aria-hidden="true" />
      ) : null}
    </button>
  )
}
