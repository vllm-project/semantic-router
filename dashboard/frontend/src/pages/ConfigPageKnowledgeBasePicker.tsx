import styles from './ConfigPageTaxonomyClassifiers.module.css'
import type { TaxonomyClassifierRecord } from './configPageTaxonomyClassifierSupport'

interface ConfigPageKnowledgeBasePickerProps {
  knowledgeBases: TaxonomyClassifierRecord[]
  selectedKnowledgeBase: TaxonomyClassifierRecord | null
  onSelect: (name: string) => void
  onRefresh: () => void
}

function describeKnowledgeBase(knowledgeBase: TaxonomyClassifierRecord): string {
  const groupCount = Object.keys(knowledgeBase.groups ?? {}).length
  const metricCount = knowledgeBase.metrics?.length ?? 0
  return `${knowledgeBase.labels.length} labels · ${groupCount} groups · ${metricCount} metrics`
}

export default function ConfigPageKnowledgeBasePicker({
  knowledgeBases,
  selectedKnowledgeBase,
  onSelect,
  onRefresh,
}: ConfigPageKnowledgeBasePickerProps) {
  return (
    <div className={styles.scopePicker}>
      <div className={styles.scopePickerHeader}>
        <div className={styles.scopePickerSummary}>
          <span className={styles.summaryLabel}>Active Base</span>
          <div className={styles.scopePickerTitleRow}>
            <strong className={styles.scopePickerTitle}>
              {selectedKnowledgeBase?.name ?? 'No knowledge base available'}
            </strong>
            {selectedKnowledgeBase ? (
              selectedKnowledgeBase.builtin ? (
                <span className={styles.tableBadge}>Built-in</span>
              ) : (
                <span className={styles.mutedBadge}>Custom</span>
              )
            ) : null}
          </div>
          <span className={styles.summaryHint}>
            Groups and labels stay scoped to one base at a time so large knowledge bases remain readable.
          </span>
          {selectedKnowledgeBase ? (
            <div className={styles.scopePickerMetaRow}>
              <span className={styles.scopePickerMeta}>{describeKnowledgeBase(selectedKnowledgeBase)}</span>
              <span className={styles.scopePickerMeta}>
                Default threshold {selectedKnowledgeBase.threshold}
              </span>
            </div>
          ) : null}
        </div>

        <div className={styles.scopePickerActions}>
          <button type="button" className={styles.secondaryButton} onClick={onRefresh}>
            Refresh
          </button>
        </div>
      </div>

      <div className={styles.scopePickerList} role="list" aria-label="Knowledge base picker">
        {knowledgeBases.length > 0 ? (
          knowledgeBases.map((knowledgeBase) => {
            const active = selectedKnowledgeBase?.name === knowledgeBase.name
            return (
              <button
                key={knowledgeBase.name}
                type="button"
                role="listitem"
                className={active ? styles.scopePickerOptionActive : styles.scopePickerOption}
                onClick={() => onSelect(knowledgeBase.name)}
              >
                <span className={styles.scopePickerOptionTitle}>{knowledgeBase.name}</span>
                <span className={styles.scopePickerOptionMeta}>{describeKnowledgeBase(knowledgeBase)}</span>
              </button>
            )
          })
        ) : (
          <div className={styles.notice}>No knowledge bases available yet.</div>
        )}
      </div>
    </div>
  )
}
