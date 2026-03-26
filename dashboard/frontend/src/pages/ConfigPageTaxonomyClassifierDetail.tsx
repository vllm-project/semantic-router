import pageStyles from './ConfigPage.module.css'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  formatSignalReference,
  type TaxonomyClassifierRecord,
} from './configPageTaxonomyClassifierSupport'

interface ConfigPageTaxonomyClassifierDetailProps {
  selectedClassifier: TaxonomyClassifierRecord | null
}

function formatMetric(metricName: string, selectedClassifier: TaxonomyClassifierRecord): string {
  const metric = (selectedClassifier.metrics ?? []).find((entry) => entry.name === metricName)
  if (!metric) {
    return metricName
  }
  if (metric.type === 'group_margin') {
    return `${metric.name}: ${metric.positive_group ?? 'unknown'} - ${metric.negative_group ?? 'unknown'}`
  }
  return `${metric.name}: ${metric.type}`
}

export default function ConfigPageTaxonomyClassifierDetail({
  selectedClassifier,
}: ConfigPageTaxonomyClassifierDetailProps) {
  const groupEntries = Object.entries(selectedClassifier?.groups ?? {})
  const thresholdEntries = Object.entries(selectedClassifier?.label_thresholds ?? {})

  if (!selectedClassifier) {
    return (
      <div className={pageStyles.sectionTableBlock}>
        <div className={styles.notice}>No knowledge base selected.</div>
      </div>
    )
  }

  return (
    <div className={pageStyles.sectionTableBlock}>
      <div className={styles.detailShell}>
        <aside className={styles.detailSummaryPanel}>
          <div className={styles.detailHeader}>
            <div>
              <span className={styles.summaryLabel}>Knowledge Base</span>
              <h3 className={styles.detailHeroTitle}>{selectedClassifier.name}</h3>
              <p className={styles.detailDescription}>
                Review the base-level routing contract first, then scan bindings and package shape on the right.
              </p>
            </div>
            {selectedClassifier.load_error ? <span className={styles.warningBadge}>Load issue</span> : null}
          </div>

          <div className={styles.detailSummaryList}>
            <div className={styles.detailSummaryRow}>
              <span className={styles.detailListLabel}>Default threshold</span>
              <span className={styles.detailListValue}>{selectedClassifier.threshold}</span>
            </div>
            <div className={styles.detailSummaryRow}>
              <span className={styles.detailListLabel}>Source</span>
              <code className={styles.inlineCode}>
                {selectedClassifier.source.path}
                {selectedClassifier.source.manifest ? `/${selectedClassifier.source.manifest}` : ''}
              </code>
            </div>
            <div className={styles.detailSummaryRow}>
              <span className={styles.detailListLabel}>Labels</span>
              <span className={styles.detailListValue}>{selectedClassifier.labels.length}</span>
            </div>
            <div className={styles.detailSummaryRow}>
              <span className={styles.detailListLabel}>Groups</span>
              <span className={styles.detailListValue}>{groupEntries.length}</span>
            </div>
            <div className={styles.detailSummaryRow}>
              <span className={styles.detailListLabel}>Signal refs</span>
              <span className={styles.detailListValue}>{selectedClassifier.signal_references.length}</span>
            </div>
            <div className={styles.detailSummaryRow}>
              <span className={styles.detailListLabel}>Metrics</span>
              <span className={styles.detailListValue}>{selectedClassifier.metrics?.length ?? 0}</span>
            </div>
            <div className={styles.detailSummaryRow}>
              <span className={styles.detailListLabel}>Threshold overrides</span>
              <span className={styles.detailListValue}>{thresholdEntries.length}</span>
            </div>
          </div>
        </aside>

        <div className={styles.detailPanelsColumn}>
          <section className={styles.detailPanel}>
            <div className={styles.groupTitle}>Routing Bindings</div>
            <div className={styles.detailPanelGrid}>
              <div className={styles.groupBlock}>
                <div className={styles.detailList}>
                  {selectedClassifier.signal_references.length > 0 ? (
                    selectedClassifier.signal_references.map((reference) => (
                      <div key={`${selectedClassifier.name}-signal-${reference.name}`} className={styles.detailListItem}>
                        <span className={styles.detailListLabel}>{reference.name}</span>
                        <span className={styles.detailListValue}>{formatSignalReference(reference)}</span>
                      </div>
                    ))
                  ) : (
                    <div className={styles.detailListEmpty}>No KB signals reference this knowledge base yet.</div>
                  )}
                </div>
              </div>

              <div className={styles.groupBlock}>
                <div className={styles.detailList}>
                  {(selectedClassifier.metrics ?? []).length > 0 ? (
                    (selectedClassifier.metrics ?? []).map((metric) => (
                      <div key={`${selectedClassifier.name}-metric-${metric.name}`} className={styles.detailListItem}>
                        <span className={styles.detailListLabel}>{metric.name}</span>
                        <span className={styles.detailListValue}>{formatMetric(metric.name, selectedClassifier)}</span>
                      </div>
                    ))
                  ) : (
                    <div className={styles.detailListEmpty}>No metrics configured.</div>
                  )}
                </div>
              </div>
            </div>
          </section>

          <section className={styles.detailPanel}>
            <div className={styles.groupTitle}>Package Shape</div>
            <div className={styles.detailPanelGrid}>
              <div className={styles.groupBlock}>
                <div className={styles.detailList}>
                  {groupEntries.length > 0 ? (
                    groupEntries.map(([name, labels]) => (
                      <div key={`${selectedClassifier.name}-group-${name}`} className={styles.detailListItem}>
                        <span className={styles.detailListLabel}>{name}</span>
                        <span className={styles.detailListValue}>
                          {labels.length} labels
                          {labels.length > 0 ? ` · ${labels.slice(0, 3).join(', ')}${labels.length > 3 ? ` +${labels.length - 3}` : ''}` : ''}
                        </span>
                      </div>
                    ))
                  ) : (
                    <div className={styles.detailListEmpty}>No groups configured.</div>
                  )}
                </div>
              </div>

              <div className={styles.groupBlock}>
                <div className={styles.detailList}>
                  {thresholdEntries.length > 0 ? (
                    thresholdEntries.map(([label, threshold]) => (
                      <div key={`${selectedClassifier.name}-threshold-${label}`} className={styles.detailListItem}>
                        <span className={styles.detailListLabel}>{label}</span>
                        <span className={styles.detailListValue}>{threshold}</span>
                      </div>
                    ))
                  ) : (
                    <div className={styles.detailListEmpty}>No label-specific thresholds configured.</div>
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
