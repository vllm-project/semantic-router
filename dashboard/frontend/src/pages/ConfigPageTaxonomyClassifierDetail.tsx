import pageStyles from './ConfigPage.module.css'
import styles from './ConfigPageTaxonomyClassifiers.module.css'
import {
  formatSignalReference,
  type TaxonomyClassifierRecord,
} from './configPageTaxonomyClassifierSupport'

interface ConfigPageTaxonomyClassifierDetailProps {
  selectedClassifier: TaxonomyClassifierRecord | null
}

export default function ConfigPageTaxonomyClassifierDetail({
  selectedClassifier,
}: ConfigPageTaxonomyClassifierDetailProps) {
  return (
    <div className={pageStyles.sectionTableBlock}>
      <div className={styles.detailHeader}>
        <div>
          <h3 className={styles.detailTitle}>
            {selectedClassifier ? `${selectedClassifier.name} Details` : 'Select a classifier'}
          </h3>
          <p className={styles.detailDescription}>
            {selectedClassifier
              ? 'Inspect the active classifier package, its taxonomy bindings, and any signal references before editing tiers or categories.'
              : 'Pick a classifier from the catalog to inspect its taxonomy.'}
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
              <span className={styles.summaryLabel}>Thresholds</span>
              <strong className={styles.summaryValueSmall}>
                {selectedClassifier.threshold} / {selectedClassifier.security_threshold ?? selectedClassifier.threshold}
              </strong>
              <span className={styles.summaryHint}>Base threshold and security override.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Source</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.source.path}</strong>
              <span className={styles.summaryHint}>Classifier package root on disk.</span>
            </article>
            <article className={styles.detailCard}>
              <span className={styles.summaryLabel}>Signal References</span>
              <strong className={styles.summaryValueSmall}>{selectedClassifier.signal_references.length}</strong>
              <span className={styles.summaryHint}>Taxonomy signals currently bound to this classifier.</span>
            </article>
          </div>

          <div className={styles.detailMetaGrid}>
            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Bindable Tiers</div>
              <div className={styles.tagList}>
                {selectedClassifier.bind_options.tiers.length > 0 ? (
                  selectedClassifier.bind_options.tiers.map((tier) => (
                    <span key={`${selectedClassifier.name}-tier-${tier}`} className={styles.tag}>
                      {tier}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No bindable tiers discovered.</span>
                )}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Tier Groups</div>
              <div className={styles.tagList}>
                {Object.entries(selectedClassifier.tier_groups ?? {}).length > 0 ? (
                  Object.entries(selectedClassifier.tier_groups ?? {}).map(([name, categories]) => (
                    <span key={`${selectedClassifier.name}-group-${name}`} className={styles.tag}>
                      {name}: {categories.join(', ')}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No tier groups configured.</span>
                )}
              </div>
            </div>

            <div className={styles.groupBlock}>
              <div className={styles.groupTitle}>Taxonomy Signal References</div>
              <div className={styles.tagList}>
                {selectedClassifier.signal_references.length > 0 ? (
                  selectedClassifier.signal_references.map((reference) => (
                    <span key={`${selectedClassifier.name}-signal-${reference.name}`} className={styles.tag}>
                      {formatSignalReference(reference)}
                    </span>
                  ))
                ) : (
                  <span className={styles.emptyTag}>No taxonomy signals reference this classifier yet.</span>
                )}
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className={styles.notice}>No classifier selected.</div>
      )}
    </div>
  )
}
