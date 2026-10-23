import React, { memo, useCallback, useId, useMemo, useState } from 'react'

import { DECISION_SIGNAL_TYPES } from '@/generated/routerConfigContract'

import useAccessibleDialog from '../hooks/useAccessibleDialog'
import styles from './ExpressionBuilder.module.css'
import {
  newRuleNodeId,
  PREDICATE_FIELDS,
  predicateDraft,
  predicateFromDraft,
  type PredicateDraft,
  type RuleLeaf,
  type RuleNode,
  type SignalDescriptor,
} from './ExpressionBuilderSupport'
import { OPERATOR_ORDER } from './ExpressionBuilderNodes'

interface EditSignalDialogProps {
  initial: RuleLeaf
  availableSignals: SignalDescriptor[]
  /** Fields set to undefined are cleared on the leaf. */
  onSave: (edit: Partial<RuleLeaf>) => void
  onCancel: () => void
}

// label and on_error only apply to classifier and decision conditions.
const isLabelledType = (signalType: string) => {
  const type = signalType.trim().toLowerCase()
  return type === 'classifier' || type === 'decision'
}

export const EditSignalDialog: React.FC<EditSignalDialogProps> = memo(
  ({ initial, availableSignals, onSave, onCancel }) => {
    const [signalType, setSignalType] = useState(initial.signalType)
    const [signalName, setSignalName] = useState(initial.signalName)
    const [label, setLabel] = useState(initial.label ?? '')
    const [onError, setOnError] = useState<string>(initial.onError ?? '')
    const [predicate, setPredicate] = useState<PredicateDraft>(() => predicateDraft(initial.predicate))
    const [search, setSearch] = useState('')
    const titleId = useId()
    const labelListId = useId()
    const dialogRef = useAccessibleDialog<HTMLDivElement>({
      isOpen: true,
      onClose: onCancel,
    })

    const types = useMemo(() => {
      const known = new Set<string>(DECISION_SIGNAL_TYPES)
      for (const signal of availableSignals) known.add(signal.signalType)
      return Array.from(known).sort()
    }, [availableSignals])

    const filteredSignals = useMemo(() => {
      let list = availableSignals.filter(signal => signal.signalType === signalType)
      if (search.trim()) {
        const query = search.toLowerCase()
        list = list.filter(signal => signal.name.toLowerCase().includes(query))
      }
      return list
    }, [availableSignals, search, signalType])

    const descriptor = useMemo(
      () => availableSignals.find(signal => signal.signalType === signalType && signal.name === signalName),
      [availableSignals, signalName, signalType]
    )
    const labelled = isLabelledType(signalType)
    const typeChanged = signalType !== initial.signalType
    // A type change drops label and on_error only when the new type cannot
    // carry them; an unchanged type keeps whatever the leaf already had.
    const keepsLabelFields = labelled || !typeChanged
    const dropsLabelFields = !keepsLabelFields && Boolean(initial.label || initial.onError)
    const unsupportedPredicateKeys = useMemo(() => {
      const raw = (initial.predicate ?? {}) as Record<string, unknown>
      return Object.keys(raw).filter(
        key => raw[key] !== undefined && !(PREDICATE_FIELDS as readonly string[]).includes(key)
      )
    }, [initial.predicate])
    const parsedPredicate = useMemo(
      () => predicateFromDraft(predicate, initial.predicate),
      [initial.predicate, predicate]
    )
    const canSave = Boolean(signalType.trim() && signalName.trim()) && parsedPredicate.invalid.length === 0

    const handleSubmit = useCallback(
      (event: React.FormEvent) => {
        event.preventDefault()
        // Synthetic submit events bubble through portals to a host form.
        event.stopPropagation()
        if (!canSave) return
        onSave({
          signalType,
          signalName,
          label: keepsLabelFields && label !== '' ? label : undefined,
          predicate: parsedPredicate.predicate,
          onError: keepsLabelFields && onError !== '' ? (onError as RuleLeaf['onError']) : undefined,
        })
      },
      [canSave, keepsLabelFields, label, onError, onSave, parsedPredicate.predicate, signalName, signalType]
    )

    return (
      <div className={styles.editOverlay} role="presentation" onMouseDown={onCancel}>
        <div
          ref={dialogRef}
          className={styles.editDialog}
          role="dialog"
          data-rule-editor-dialog
          aria-modal="true"
          aria-labelledby={titleId}
          tabIndex={-1}
          onMouseDown={event => event.stopPropagation()}
        >
          <div className={styles.editDialogHeader}>
            <span id={titleId}>Edit Signal</span>
            <button
              type="button"
              className={styles.editDialogClose}
              aria-label="Close signal editor"
              onClick={onCancel}
            >
              ×
            </button>
          </div>
          <form onSubmit={handleSubmit} className={styles.editDialogBody}>
            <label className={styles.editLabel}>
              Signal Type
              <select
                className={styles.editSelect}
                value={signalType}
                onChange={event => {
                  setSignalType(event.target.value)
                  setSignalName('')
                }}
              >
                {types.map(type => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
                {!types.includes(signalType) ? <option value={signalType}>{signalType}</option> : null}
              </select>
            </label>
            <label className={styles.editLabel}>
              Signal Name
              <input
                className={styles.editInput}
                value={signalName}
                onChange={event => setSignalName(event.target.value)}
                placeholder="Enter signal name"
                data-dialog-initial-focus
              />
            </label>
            {filteredSignals.length > 0 ? (
              <div className={styles.editSignalList}>
                <input
                  className={styles.editSearchInput}
                  value={search}
                  onChange={event => setSearch(event.target.value)}
                  placeholder="Filter signals..."
                  aria-label="Filter available signals"
                />
                <div className={styles.editSignalOptions}>
                  {filteredSignals.map(signal => (
                    <button
                      type="button"
                      key={signal.name}
                      className={`${styles.editSignalOption} ${signal.name === signalName ? styles.editSignalOptionActive : ''}`}
                      onClick={() => setSignalName(signal.name)}
                    >
                      {signal.name}
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
            {labelled ? (
              <>
                <label className={styles.editLabel}>
                  Label
                  <input
                    className={styles.editInput}
                    value={label}
                    list={descriptor?.labels?.length ? labelListId : undefined}
                    onChange={event => setLabel(event.target.value)}
                    placeholder={descriptor?.labelForbidden ? 'This question takes no label' : 'Declared label'}
                  />
                  {descriptor?.labels?.length ? (
                    <datalist id={labelListId}>
                      {descriptor.labels.map(option => (
                        <option key={option} value={option} />
                      ))}
                    </datalist>
                  ) : null}
                </label>
                <label className={styles.editLabel}>
                  On error
                  <select
                    className={styles.editSelect}
                    value={onError}
                    onChange={event => setOnError(event.target.value)}
                  >
                    <option value="">Not set</option>
                    <option value="no_match">no_match</option>
                    <option value="match">match</option>
                  </select>
                </label>
              </>
            ) : null}
            <fieldset className={styles.editLabel} style={{ border: 'none', margin: 0, padding: 0 }}>
              <legend>Score predicate</legend>
              <div className={styles.editFieldRow}>
                {PREDICATE_FIELDS.map(key => (
                  <label key={key} className={styles.editLabel}>
                    {key}
                    <input
                      className={styles.editInput}
                      type="number"
                      step="any"
                      value={predicate[key]}
                      aria-invalid={parsedPredicate.invalid.includes(key) ? true : undefined}
                      onChange={event => setPredicate(current => ({ ...current, [key]: event.target.value }))}
                    />
                  </label>
                ))}
              </div>
            </fieldset>
            {unsupportedPredicateKeys.length > 0 ? (
              <p className={styles.editNote} role="note">
                Unsupported predicate {unsupportedPredicateKeys.length === 1 ? 'field' : 'fields'}{' '}
                {unsupportedPredicateKeys.join(', ')} {unsupportedPredicateKeys.length === 1 ? 'is' : 'are'} kept
                and reported as an error until removed from the config.
              </p>
            ) : null}
            {dropsLabelFields ? (
              <p className={styles.editNote} role="note">
                {signalType} conditions cannot carry a label or on_error, so saving clears them.
              </p>
            ) : null}
            <div className={styles.editDialogActions}>
              <button type="button" className={styles.editBtnCancel} onClick={onCancel}>
                Cancel
              </button>
              <button
                type="submit"
                className={styles.editBtnSave}
                disabled={!canSave}
              >
                Save
              </button>
            </div>
          </form>
        </div>
      </div>
    )
  }
)
EditSignalDialog.displayName = 'EditSignalDialog'

interface AddChildPickerProps {
  availableSignals: SignalDescriptor[]
  onPick: (node: RuleNode) => void
  onCancel: () => void
}

export const AddChildPicker: React.FC<AddChildPickerProps> = memo(
  ({ availableSignals, onPick, onCancel }) => {
    const [search, setSearch] = useState('')
    const titleId = useId()
    const dialogRef = useAccessibleDialog<HTMLDivElement>({
      isOpen: true,
      onClose: onCancel,
    })

    const groups = useMemo(() => {
      const grouped: Record<string, SignalDescriptor[]> = {}
      for (const signal of availableSignals) {
        const key = signal.signalType.toUpperCase()
        if (!grouped[key]) grouped[key] = []
        grouped[key].push(signal)
      }
      return Object.entries(grouped).sort((left, right) => left[0].localeCompare(right[0]))
    }, [availableSignals])

    const filteredGroups = useMemo(() => {
      if (!search.trim()) return groups
      const query = search.toLowerCase()
      return groups
        .map(
          ([type, signals]) =>
            [
              type,
              signals.filter(
                signal =>
                  signal.name.toLowerCase().includes(query) ||
                  signal.signalType.toLowerCase().includes(query)
              ),
            ] as [string, SignalDescriptor[]]
        )
        .filter(([, signals]) => signals.length > 0)
    }, [groups, search])

    return (
      <div className={styles.editOverlay} role="presentation" onMouseDown={onCancel}>
        <div
          ref={dialogRef}
          className={styles.addPickerDialog}
          role="dialog"
          data-rule-editor-dialog
          aria-modal="true"
          aria-labelledby={titleId}
          tabIndex={-1}
          onMouseDown={event => event.stopPropagation()}
        >
          <div className={styles.editDialogHeader}>
            <span id={titleId}>Add Child Node</span>
            <button
              type="button"
              className={styles.editDialogClose}
              aria-label="Close child node picker"
              onClick={onCancel}
            >
              ×
            </button>
          </div>
          <div className={styles.addPickerBody}>
            <div className={styles.addPickerSection}>
              <div className={styles.addPickerSectionTitle}>Operators</div>
              <div className={styles.addPickerOps}>
                {OPERATOR_ORDER.map(operator => (
                  <button
                    type="button"
                    key={operator}
                    className={styles.addPickerOpBtn}
                    onClick={() =>
                      onPick(
                        operator === 'NOT'
                          ? { id: newRuleNodeId(), operator: 'NOT', conditions: [] as unknown as [RuleNode] }
                          : { id: newRuleNodeId(), operator, conditions: [] }
                      )
                    }
                  >
                    {operator}
                  </button>
                ))}
              </div>
            </div>
            <div className={styles.addPickerSection}>
              <div className={styles.addPickerSectionTitle}>Signals</div>
              <input
                className={styles.editInput}
                value={search}
                onChange={event => setSearch(event.target.value)}
                placeholder="Search signals..."
                aria-label="Search signals"
                data-dialog-initial-focus
              />
              <div className={styles.addPickerSignalList}>
                {filteredGroups.map(([type, signals]) => (
                  <div key={type}>
                    <div className={styles.addPickerGroupTitle}>{type}</div>
                    <div className={styles.addPickerGroupItems}>
                      {signals.map(signal => (
                        <button
                          type="button"
                          key={signal.name}
                          className={styles.addPickerSignalItem}
                          onClick={() =>
                            onPick({ id: newRuleNodeId(), signalType: signal.signalType, signalName: signal.name })
                          }
                        >
                          {signal.name}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
                {filteredGroups.length === 0 ? (
                  <div className={styles.addPickerEmpty}>No matching signals</div>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }
)
AddChildPicker.displayName = 'AddChildPicker'
