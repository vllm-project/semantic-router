import React, { memo, useCallback, useEffect, useMemo, useState } from 'react'

import styles from './ExpressionBuilder.module.css'
import type { RuleNode, SignalDescriptor } from './ExpressionBuilderSupport'
import { OPERATOR_ORDER } from './ExpressionBuilderNodes'

interface EditSignalDialogProps {
  signalType: string
  signalName: string
  availableSignals: SignalDescriptor[]
  onSave: (signalType: string, signalName: string) => void
  onCancel: () => void
}

export const EditSignalDialog: React.FC<EditSignalDialogProps> = memo(
  ({ signalType: initialType, signalName: initialName, availableSignals, onSave, onCancel }) => {
    const [signalType, setSignalType] = useState(initialType)
    const [signalName, setSignalName] = useState(initialName)
    const [search, setSearch] = useState('')

    const types = useMemo(
      () => Array.from(new Set(availableSignals.map(signal => signal.signalType))).sort(),
      [availableSignals]
    )

    const filteredSignals = useMemo(() => {
      let list = availableSignals.filter(signal => signal.signalType === signalType)
      if (search.trim()) {
        const query = search.toLowerCase()
        list = list.filter(signal => signal.name.toLowerCase().includes(query))
      }
      return list
    }, [availableSignals, search, signalType])

    const handleSubmit = useCallback(
      (event: React.FormEvent) => {
        event.preventDefault()
        if (signalType.trim() && signalName.trim()) {
          onSave(signalType, signalName)
        }
      },
      [onSave, signalName, signalType]
    )

    useEffect(() => {
      const handler = (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
          onCancel()
        }
      }
      window.addEventListener('keydown', handler)
      return () => window.removeEventListener('keydown', handler)
    }, [onCancel])

    return (
      <div className={styles.editOverlay} onClick={onCancel}>
        <div className={styles.editDialog} onClick={event => event.stopPropagation()}>
          <div className={styles.editDialogHeader}>
            <span>Edit Signal</span>
            <button className={styles.editDialogClose} onClick={onCancel}>
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
                autoFocus
              />
            </label>
            {filteredSignals.length > 0 ? (
              <div className={styles.editSignalList}>
                <input
                  className={styles.editSearchInput}
                  value={search}
                  onChange={event => setSearch(event.target.value)}
                  placeholder="Filter signals..."
                />
                <div className={styles.editSignalOptions}>
                  {filteredSignals.map(signal => (
                    <div
                      key={signal.name}
                      className={`${styles.editSignalOption} ${signal.name === signalName ? styles.editSignalOptionActive : ''}`}
                      onClick={() => setSignalName(signal.name)}
                    >
                      {signal.name}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            <div className={styles.editDialogActions}>
              <button type="button" className={styles.editBtnCancel} onClick={onCancel}>
                Cancel
              </button>
              <button
                type="submit"
                className={styles.editBtnSave}
                disabled={!signalType.trim() || !signalName.trim()}
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

    useEffect(() => {
      const handler = (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
          onCancel()
        }
      }
      window.addEventListener('keydown', handler)
      return () => window.removeEventListener('keydown', handler)
    }, [onCancel])

    return (
      <div className={styles.editOverlay} onClick={onCancel}>
        <div className={styles.addPickerDialog} onClick={event => event.stopPropagation()}>
          <div className={styles.editDialogHeader}>
            <span>Add Child Node</span>
            <button className={styles.editDialogClose} onClick={onCancel}>
              ×
            </button>
          </div>
          <div className={styles.addPickerBody}>
            <div className={styles.addPickerSection}>
              <div className={styles.addPickerSectionTitle}>Operators</div>
              <div className={styles.addPickerOps}>
                {OPERATOR_ORDER.map(operator => (
                  <button
                    key={operator}
                    className={styles.addPickerOpBtn}
                    onClick={() =>
                      onPick(
                        operator === 'NOT'
                          ? { operator: 'NOT', conditions: [] as unknown as [RuleNode] }
                          : { operator, conditions: [] }
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
                autoFocus
              />
              <div className={styles.addPickerSignalList}>
                {filteredGroups.map(([type, signals]) => (
                  <div key={type}>
                    <div className={styles.addPickerGroupTitle}>{type}</div>
                    <div className={styles.addPickerGroupItems}>
                      {signals.map(signal => (
                        <div
                          key={signal.name}
                          className={styles.addPickerSignalItem}
                          onClick={() =>
                            onPick({ signalType: signal.signalType, signalName: signal.name })
                          }
                        >
                          {signal.name}
                        </div>
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
