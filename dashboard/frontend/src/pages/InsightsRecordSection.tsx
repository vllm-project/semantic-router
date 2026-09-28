import { useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import type { ViewSection } from '../components/ViewPanel'

import styles from './InsightsPage.module.css'
import { getInsightsRecordSectionPresentation } from './insightsRecordSectionSupport'

interface InsightsRecordSectionProps {
  section: ViewSection
  sectionIndex: number
}

export default function InsightsRecordSection({
  section,
  sectionIndex,
}: InsightsRecordSectionProps) {
  const presentation = getInsightsRecordSectionPresentation(section.title)
  const [expanded, setExpanded] = useState(presentation.defaultExpanded ?? true)
  const sectionTitle = section.title || 'Details'
  const sectionId = `insight-section-${sectionIndex}`
  const noteFields = section.fields.filter((field) =>
    presentation.noteFields?.includes(field.label),
  )
  const secondaryFields = section.fields.filter((field) =>
    presentation.secondaryFields?.includes(field.label),
  )
  const primaryFields = section.fields.filter(
    (field) =>
      !presentation.noteFields?.includes(field.label) &&
      !presentation.secondaryFields?.includes(field.label),
  )

  return (
    <section
      className={`${styles.recordSection} ${styles[`recordSection${presentation.size.charAt(0).toUpperCase()}${presentation.size.slice(1)}`]}`}
      aria-labelledby={`${sectionId}-title`}
    >
      <div className={styles.recordSectionHeader}>
        <div>
          <h2 id={`${sectionId}-title`}>{sectionTitle}</h2>
          {presentation.size === 'wide' ? (
            <span className={styles.recordSectionCount}>
              {presentation.description ??
                `${section.fields.length} ${section.fields.length === 1 ? 'group' : 'groups'}`}
            </span>
          ) : null}
        </div>
        {presentation.collapsible ? (
          <button
            type="button"
            className={styles.recordSectionToggle}
            aria-expanded={expanded}
            aria-label={`${expanded ? 'Collapse' : 'Expand'} ${sectionTitle}`}
            aria-controls={`${sectionId}-content`}
            onClick={() => setExpanded((current) => !current)}
          >
            <span aria-hidden="true">{expanded ? 'Collapse' : 'Expand'}</span>
            <ProductIcon
              name="chevron-down"
              width={15}
              height={15}
              className={expanded ? styles.recordSectionChevronOpen : undefined}
            />
          </button>
        ) : null}
      </div>

      {expanded ? (
        <div
          id={`${sectionId}-content`}
          className={
            presentation.structured
              ? styles.recordStructured
              : `${styles.recordFields} ${presentation.size === 'wide' ? styles.recordFieldsWide : ''} ${presentation.metricColumns === 4 ? styles.recordMetrics : ''} ${presentation.metricColumns === 3 ? styles.recordFieldsThree : ''}`
          }
        >
          {primaryFields.map((field, fieldIndex) => (
            <div
              key={`${field.label}-${fieldIndex}`}
              className={
                presentation.structured
                  ? undefined
                  : `${styles.recordField} ${field.fullWidth ? styles.recordFieldWide : ''}`
              }
            >
              {!presentation.structured ? <span>{field.label}</span> : null}
              <div>{field.value}</div>
            </div>
          ))}
          {noteFields.map((field) => (
            <p key={field.label} className={styles.recordSectionNote}>
              <strong>{field.label}: </strong>
              {field.value}
            </p>
          ))}
          {secondaryFields.length > 0 ? (
            <details className={styles.recordExplanation}>
              <summary>
                <span>{presentation.secondaryTitle || 'More details'}</span>
                <ProductIcon name="chevron-down" width={15} height={15} />
              </summary>
              <dl>
                {secondaryFields.map((field) => (
                  <div key={field.label}>
                    <dt>{field.label}</dt>
                    <dd>{field.value}</dd>
                  </div>
                ))}
              </dl>
            </details>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
