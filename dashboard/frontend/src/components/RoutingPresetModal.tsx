import React from 'react'
import styles from './RoutingPresetModal.module.css'
import {
  countSignals,
  getRoutingPreset,
  routingPresets,
  type RoutingPresetId,
} from '../presets/routingPresets'

interface RoutingPresetModalProps {
  isOpen: boolean
  defaultModel: string
  selectedPresetId: RoutingPresetId | null
  conflicts: string[]
  error: string | null
  isApplying: boolean
  onClose: () => void
  onSelectPreset: (presetId: RoutingPresetId) => void
  onApply: () => void
}

const RoutingPresetModal: React.FC<RoutingPresetModalProps> = ({
  isOpen,
  defaultModel,
  selectedPresetId,
  conflicts,
  error,
  isApplying,
  onClose,
  onSelectPreset,
  onApply,
}) => {
  if (!isOpen) {
    return null
  }

  const selectedPreset = selectedPresetId ? getRoutingPreset(selectedPresetId) : null
  const selectedFragment = selectedPreset ? selectedPreset.build(defaultModel) : null

  return (
    <div className={styles.overlay} onClick={onClose}>
      <div className={styles.modal} onClick={(event) => event.stopPropagation()}>
        <div className={styles.header}>
          <div>
            <h3 className={styles.title}>Apply routing preset</h3>
            <p className={styles.subtitle}>
              Presets add reusable signals and decisions on top of the current config. Existing names must stay unique.
            </p>
          </div>
          <button className={styles.closeButton} onClick={onClose} aria-label="Close preset picker">
            ×
          </button>
        </div>

        <div className={styles.grid}>
          {routingPresets.map((preset) => {
            const fragment = preset.build(defaultModel)
            const isSelected = preset.id === selectedPresetId

            return (
              <button
                key={preset.id}
                className={`${styles.card} ${isSelected ? styles.cardSelected : ''}`}
                onClick={() => onSelectPreset(preset.id)}
              >
                <div className={styles.cardHeader}>
                  <h4 className={styles.cardTitle}>{preset.label}</h4>
                  <span className={styles.cardBadge}>Preset</span>
                </div>
                <p className={styles.cardDescription}>{preset.description}</p>
                <div className={styles.cardStats}>
                  <span>{countSignals(fragment.signals)} signals</span>
                  <span>{fragment.decisions.length} decisions</span>
                </div>
              </button>
            )
          })}
        </div>

        <div className={styles.summary}>
          <div className={styles.summaryHeader}>
            <h4 className={styles.summaryTitle}>
              {selectedPreset ? `${selectedPreset.label} summary` : 'Select a preset'}
            </h4>
            <span className={styles.summaryModel}>Target model: {defaultModel || 'Unavailable'}</span>
          </div>

          {selectedPreset && selectedFragment ? (
            <>
              <p className={styles.summaryText}>{selectedPreset.description}</p>
              <div className={styles.summaryStats}>
                <div className={styles.statBlock}>
                  <span className={styles.statLabel}>Signals</span>
                  <span className={styles.statValue}>{countSignals(selectedFragment.signals)}</span>
                </div>
                <div className={styles.statBlock}>
                  <span className={styles.statLabel}>Decisions</span>
                  <span className={styles.statValue}>{selectedFragment.decisions.length}</span>
                </div>
              </div>
            </>
          ) : (
            <p className={styles.summaryText}>Pick one preset to preview the routing fragment before saving it.</p>
          )}

          {conflicts.length > 0 && (
            <div className={styles.warningPanel}>
              <div className={styles.warningTitle}>Conflicting names detected</div>
              <ul className={styles.warningList}>
                {conflicts.map((conflict) => (
                  <li key={conflict}>{conflict}</li>
                ))}
              </ul>
            </div>
          )}

          {error && (
            <div className={styles.errorPanel}>
              <div className={styles.errorTitle}>Preset application failed</div>
              <p className={styles.errorText}>{error}</p>
            </div>
          )}
        </div>

        <div className={styles.footer}>
          <button className={styles.secondaryButton} onClick={onClose}>
            Cancel
          </button>
          <button
            className={styles.primaryButton}
            onClick={onApply}
            disabled={!selectedPreset || conflicts.length > 0 || isApplying}
          >
            {isApplying ? 'Applying…' : 'Apply preset'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default RoutingPresetModal
