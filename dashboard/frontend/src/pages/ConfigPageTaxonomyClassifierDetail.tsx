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
  return (
    <div className={pageStyles.sectionTableBlock}>
      <div className={styles.detailHeader}>
        <div>
          <h3 className={styles.detailTitle}>
            {selectedClassifier ? `${selectedClassifier.name} Details` : 'Select a knowledge base'}
          </h3>
          <p className={styles.detailDescription}>
            {selectedClassifier
              ? 'Inspect the active knowledge base package, its labels, groups, metrics, and signal bindings before editing.'
              : 'Pick a knowledge base from the catalog to inspect its package, bindings, and metric outputs.'}
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
              <span className={styles.summaryLabel}>Threshold</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.threshold}</strong>
              <span className={styles.summaryHint}>Default threshold for label threshold matching.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Source</span>
              <strong className={styles.summaryValueSmall}>
                {selectedClassifier.source.path}
                {selectedClassifier.source.manifest ? `/${selectedClassifier.source.manifest}` : ''}
              </strong>
              <span className={styles.summaryHint}>Knowledge base asset package on disk.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Signal References</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.signal_references.length}</strong>
              <span className={styles.summaryHint}>KB signals currently bound to this knowledge base.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Metrics</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.metrics?.length ?? 0}</strong>
              <span className={styles.summaryHint}>Numeric outputs available to projections.</span>
            </article>
          </div>

          <div className={styles.detailMetaGrid}>
            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Bindable Labels</div>
              <div className={styles.tagList}>
                {selectedClassifier.bind_options.labels.length > 0 ? (
                  selectedClassifier.bind_options.labels.map((label) => (
                    <span key={`${selectedClassifier.name}-label-${label}`} className={styles.tag}>
                      {label}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No bindable labels discovered.</span>
                )}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Bindable Groups</div>
              <div className={styles.tagList}>
                {selectedClassifier.bind_options.groups.length > 0 ? (
                  selectedClassifier.bind_options.groups.map((group) => (
                    <span key={`${selectedClassifier.name}-group-${group}`} className={styles.tag}>
                      {group}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No bindable groups discovered.</span>
                )}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Bindable Metrics</div>
              <div className={styles.tagList}>
                {selectedClassifier.bind_options.metrics.length > 0 ? (
                  selectedClassifier.bind_options.metrics.map((metricName) => (
                    <span key={`${selectedClassifier.name}-metric-${metricName}`} className={styles.tag}>
                      {formatMetric(metricName, selectedClassifier)}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No metrics configured.</span>
                )}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Signal References</div>
              <div className={styles.tagList}>
                {selectedClassifier.signal_references.length > 0 ? (
                  selectedClassifier.signal_references.map((reference) => (
                    <span key={`${selectedClassifier.name}-signal-${reference.name}`} className={styles.tag}>
                      {formatSignalReference(reference)}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No KB signals reference this knowledge base yet.</span>
                )}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Groups</div>
              <div className={styles.tagList}>
                {Object.entries(selectedClassifier.groups ?? {}).length > 0 ? (
                  Object.entries(selectedClassifier.groups ?? {}).map(([name, labels]) => (
                    <span key={`${selectedClassifier.name}-configured-group-${name}`} className={styles.tag}>
                      {name}: {labels.join(', ')}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No groups configured.</span>
                )}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Label Threshold Overrides</div>
              <div className={styles.tagList}>
                {Object.entries(selectedClassifier.label_thresholds ?? {}).length > 0 ? (
                  Object.entries(selectedClassifier.label_thresholds ?? {}).map(([label, threshold]) => (
                    <span key={`${selectedClassifier.name}-threshold-${label}`} className={styles.tag}>
                      {label}: {threshold}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No label-specific thresholds configured.</span>
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
