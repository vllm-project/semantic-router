import pageStyles from './ConfigPage.module.css'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  formatSignalReference,
  type TaxonomyClassifierRecord,
} from './configPageTaxonomyClassifierSupport'

interface ConfigPageTaxonomyClassifierDetailProps {
  selectedClassifier: TaxonomyClassifierRecord | null
}

function summarizeGroup(labels: string[]): string {
  if (labels.length === 0) {
    return '0 labels'
  }
  const preview = labels.slice(0, 3).join(', ')
  const remainder = labels.length > 3 ? ` +${labels.length - 3}` : ''
  return `${labels.length} labels · ${preview}${remainder}`
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
  const sourcePath = selectedClassifier
    ? `${selectedClassifier.source.path}${selectedClassifier.source.manifest ? `/${selectedClassifier.source.manifest}` : ''}`
    : ''

  if (!selectedClassifier) {
    return (
      <div className={pageStyles.sectionTableBlock}>
        <div className={styles.notice}>No knowledge base selected.</div>
      </div>
    )
  }

  return (
    <div className={pageStyles.sectionTableBlock}>
      <div className={styles.detailStack}>
        <section className={styles.detailOverviewPanel}>
          <div className={styles.detailHeader}>
            <div>
              <span className={styles.summaryLabel}>Knowledge</span>
              <h3 className={styles.detailHeroTitle}>{selectedClassifier.name}</h3>
              <p className={styles.detailDescription}>
                Start with the base-level contract here, then scan routing bindings and package structure below.
              </p>
            </div>
            {selectedClassifier.load_error ? <span className={styles.warningBadge}>Load issue</span> : null}
          </div>

          <div className={styles.detailOverviewGrid}>
            <div className={styles.detailOverviewCard}>
              <span className={styles.detailListLabel}>Default threshold</span>
              <span className={styles.detailOverviewValue}>{selectedClassifier.threshold}</span>
            </div>
            <div className={styles.detailOverviewCard}>
              <span className={styles.detailListLabel}>Labels</span>
              <span className={styles.detailOverviewValue}>{selectedClassifier.labels.length}</span>
            </div>
            <div className={styles.detailOverviewCard}>
              <span className={styles.detailListLabel}>Groups</span>
              <span className={styles.detailOverviewValue}>{groupEntries.length}</span>
            </div>
            <div className={styles.detailOverviewCard}>
              <span className={styles.detailListLabel}>Signal refs</span>
              <span className={styles.detailOverviewValue}>{selectedClassifier.signal_references.length}</span>
            </div>
            <div className={styles.detailOverviewCard}>
              <span className={styles.detailListLabel}>Metrics</span>
              <span className={styles.detailOverviewValue}>{selectedClassifier.metrics?.length ?? 0}</span>
            </div>
            <div className={styles.detailOverviewCard}>
              <span className={styles.detailListLabel}>Threshold overrides</span>
              <span className={styles.detailOverviewValue}>{thresholdEntries.length}</span>
            </div>
          </div>

          <div className={styles.detailOverviewMeta}>
            <span className={styles.detailListLabel}>Source</span>
            <code className={styles.inlineCode}>{sourcePath}</code>
          </div>
        </section>

        <div className={styles.detailPanelSplit}>
          <section className={styles.detailPanel}>
            <div className={styles.detailPanelHeader}>
              <div>
                <div className={styles.groupTitle}>Routing Bindings</div>
                <p className={styles.detailPanelDescription}>
                  Signals and metrics that currently depend on this knowledge base.
                </p>
              </div>
            </div>
            <div className={styles.detailPanelSection}>
              <div className={styles.detailSectionHeader}>
                <span className={styles.detailSectionTitle}>Signals</span>
                <span className={styles.detailSectionCount}>{selectedClassifier.signal_references.length}</span>
              </div>
              <div className={styles.detailScrollableList}>
                {selectedClassifier.signal_references.length > 0 ? (
                  selectedClassifier.signal_references.map((reference) => (
                    <div key={`${selectedClassifier.name}-signal-${reference.name}`} className={styles.detailListRow}>
                      <span className={styles.detailRowKey}>{reference.name}</span>
                      <span className={styles.detailRowValue}>{formatSignalReference(reference)}</span>
                    </div>
                  ))
                ) : (
                  <div className={styles.detailListEmpty}>No knowledge signals reference this base yet.</div>
                )}
              </div>
            </div>
            <div className={styles.detailPanelSection}>
              <div className={styles.detailSectionHeader}>
                <span className={styles.detailSectionTitle}>Metrics</span>
                <span className={styles.detailSectionCount}>{selectedClassifier.metrics?.length ?? 0}</span>
              </div>
              <div className={styles.detailScrollableList}>
                {(selectedClassifier.metrics ?? []).length > 0 ? (
                  (selectedClassifier.metrics ?? []).map((metric) => (
                    <div key={`${selectedClassifier.name}-metric-${metric.name}`} className={styles.detailListRow}>
                      <span className={styles.detailRowKey}>{metric.name}</span>
                      <span className={styles.detailRowValue}>{formatMetric(metric.name, selectedClassifier)}</span>
                    </div>
                  ))
                ) : (
                  <div className={styles.detailListEmpty}>No metrics configured.</div>
                )}
              </div>
            </div>
          </section>

          <section className={styles.detailPanel}>
            <div className={styles.detailPanelHeader}>
              <div>
                <div className={styles.groupTitle}>Package Shape</div>
                <p className={styles.detailPanelDescription}>
                  Groups and label-specific threshold overrides defined inside this knowledge base.
                </p>
              </div>
            </div>
            <div className={styles.detailPanelSection}>
              <div className={styles.detailSectionHeader}>
                <span className={styles.detailSectionTitle}>Groups</span>
                <span className={styles.detailSectionCount}>{groupEntries.length}</span>
              </div>
              <div className={styles.detailScrollableList}>
                {groupEntries.length > 0 ? (
                  groupEntries.map(([name, labels]) => (
                    <div key={`${selectedClassifier.name}-group-${name}`} className={styles.detailListRow}>
                      <span className={styles.detailRowKey}>{name}</span>
                      <span className={styles.detailRowValue}>{summarizeGroup(labels)}</span>
                    </div>
                  ))
                ) : (
                  <div className={styles.detailListEmpty}>No groups configured.</div>
                )}
              </div>
            </div>
            <div className={styles.detailPanelSection}>
              <div className={styles.detailSectionHeader}>
                <span className={styles.detailSectionTitle}>Threshold overrides</span>
                <span className={styles.detailSectionCount}>{thresholdEntries.length}</span>
              </div>
              <div className={styles.detailScrollableList}>
                {thresholdEntries.length > 0 ? (
                  thresholdEntries.map(([label, threshold]) => (
                    <div key={`${selectedClassifier.name}-threshold-${label}`} className={styles.detailListRow}>
                      <span className={styles.detailRowKey}>{label}</span>
                      <span className={styles.detailRowValue}>{threshold}</span>
                    </div>
                  ))
                ) : (
                  <div className={styles.detailListEmpty}>No label-specific thresholds configured.</div>
                )}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
