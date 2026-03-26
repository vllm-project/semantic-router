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

  return (
    <div className={pageStyles.sectionTableBlock}>
      <div className={styles.detailHeader}>
        <div>
          <h3 className={styles.detailTitle}>
            {selectedClassifier ? `${selectedClassifier.name} Details` : 'Select a knowledge base'}
          </h3>
          <p className={styles.detailDescription}>
            {selectedClassifier
              ? 'Review the essential KB settings and routing bindings without expanding every label and group into one long wall of tags.'
              : 'Pick a knowledge base from the catalog to inspect its routing-facing settings.'}
          </p>
        </div>
        {selectedClassifier?.load_error ? (
          <span className={styles.warningBadge}>Load issue</span>
        ) : null}
      </div>

      {selectedClassifier ? (
        <>
          <div className={styles.detailGrid}>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Labels</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.labels.length}</strong>
              <span className={styles.summaryHint}>Labels currently stored in this base.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Groups</span>
              <strong className={styles.summaryValueSmall}>{groupEntries.length}</strong>
              <span className={styles.summaryHint}>Routing groups defined on top of the label set.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Signal References</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.signal_references.length}</strong>
              <span className={styles.summaryHint}>Signals currently bound to this knowledge base.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Metrics</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.metrics?.length ?? 0}</strong>
              <span className={styles.summaryHint}>Numeric outputs available to projections.</span>
            </article>
          </div>

          <div className={styles.detailMetaGrid}>
            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Core Settings</div>
              <div className={styles.detailList}>
                <div className={styles.detailListItem}>
                  <span className={styles.detailListLabel}>Default threshold</span>
                  <span className={styles.detailListValue}>{selectedClassifier.threshold}</span>
                </div>
                <div className={styles.detailListItem}>
                  <span className={styles.detailListLabel}>Source</span>
                  <code className={styles.inlineCode}>
                    {selectedClassifier.source.path}
                    {selectedClassifier.source.manifest ? `/${selectedClassifier.source.manifest}` : ''}
                  </code>
                </div>
                <div className={styles.detailListItem}>
                  <span className={styles.detailListLabel}>Threshold overrides</span>
                  <span className={styles.detailListValue}>{thresholdEntries.length}</span>
                </div>
                <div className={styles.detailListItem}>
                  <span className={styles.detailListLabel}>Bindable groups</span>
                  <span className={styles.detailListValue}>{selectedClassifier.bind_options.groups.length}</span>
                </div>
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Signal References</div>
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
              <div className={styles.groupTitle}>Metrics</div>
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

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Groups</div>
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
              <div className={styles.groupTitle}>Threshold Overrides</div>
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
        </>
      ) : (
        <div className={styles.notice}>No knowledge base selected.</div>
      )}
    </div>
  )
}
