import { useState } from 'react'

import ProductIcon from '../components/ProductIcon'
import type { ViewSection } from '../components/ViewPanel'

import styles from './InsightsPage.module.css'

type RecordSectionSize = 'compact' | 'feature' | 'wide'

const SECTION_PRESENTATION: Record<
  string,
  { size: RecordSectionSize; collapsible?: boolean; defaultExpanded?: boolean }
> = {
  Lifecycle: { size: 'compact' },
  'Decision Information': { size: 'compact' },
  'Model Selection': { size: 'compact' },
  'Usage & Cost': { size: 'compact' },
  Signals: { size: 'compact' },
  'Plugin Status': { size: 'compact' },
  'Routing Metadata': { size: 'wide', collapsible: true, defaultExpanded: false },
  'Projection Trace': { size: 'wide', collapsible: true, defaultExpanded: true },
  'Tool Trace': { size: 'wide', collapsible: true, defaultExpanded: false },
  'Request / Response': { size: 'wide', collapsible: true, defaultExpanded: false },
}

export function getInsightsRecordSectionPresentation(title?: string) {
  return SECTION_PRESENTATION[title || ''] || { size: 'compact' as const }
}

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
              {section.fields.length} {section.fields.length === 1 ? 'group' : 'groups'}
            </span>
          ) : null}
        </div>
        {presentation.collapsible ? (
          <button
            type="button"
            className={styles.recordSectionToggle}
            aria-expanded={expanded}
            aria-controls={`${sectionId}-content`}
            onClick={() => setExpanded((current) => !current)}
          >
            <span>{expanded ? 'Collapse' : 'Expand'}</span>
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
          className={`${styles.recordFields} ${presentation.size === 'wide' ? styles.recordFieldsWide : ''}`}
        >
          {section.fields.map((field, fieldIndex) => (
            <div
              key={`${field.label}-${fieldIndex}`}
              className={`${styles.recordField} ${field.fullWidth ? styles.recordFieldWide : ''}`}
            >
              <span>{field.label}</span>
              <div>{field.value}</div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  )
}
