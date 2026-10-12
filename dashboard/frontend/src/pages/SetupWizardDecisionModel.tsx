import styles from './SetupWizardPage.module.css'
import { DECISION_MODEL_OPTIONS, type DecisionModelName } from './decisionModelSupport'

interface SetupDecisionModelSectionProps {
  value: DecisionModelName
  onChange: (name: DecisionModelName) => void
}

export function SetupDecisionModelSection({ value, onChange }: SetupDecisionModelSectionProps) {
  return (
    <section className={styles.presetSection} aria-labelledby="setup-decision-model-title">
      <div className={styles.presetSectionHeader}>
        <div>
          <h3 id="setup-decision-model-title" className={styles.presetSectionTitle}>
            Decision model
          </h3>
          <p className={styles.presetSectionDescription}>
            The Vela model that answers the Router&apos;s questions: the built-in signals, and every
            decision question that names no deployment, in one call per request. A decision
            selector that names no deployment asks it too. Each size brings its own calibrated
            thresholds.
          </p>
        </div>
        <span className={styles.presetSummaryBadge}>{value}</span>
      </div>
      <div className={styles.presetGrid} role="radiogroup" aria-label="Decision model">
        {DECISION_MODEL_OPTIONS.map((option) => {
          const active = option.name === value
          return (
            <button
              key={option.name}
              type="button"
              role="radio"
              aria-checked={active}
              className={`${styles.presetCard} ${active ? styles.presetCardActive : ''}`}
              onClick={() => onChange(option.name)}
            >
              <div className={styles.presetCardHeader}>
                <h4 className={styles.presetCardTitle}>{option.label}</h4>
                <span className={styles.presetCardMeta}>{option.hardware}</span>
              </div>
              <p className={styles.presetCardDescription}>{option.summary}</p>
            </button>
          )
        })}
      </div>
    </section>
  )
}
