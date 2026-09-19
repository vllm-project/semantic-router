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
              : `${styles.recordFields} ${presentation.size === 'wide' ? styles.recordFieldsWide : ''}`
          }
        >
          {section.fields.map((field, fieldIndex) => (
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
        </div>
      ) : null}
    </section>
  )
}
