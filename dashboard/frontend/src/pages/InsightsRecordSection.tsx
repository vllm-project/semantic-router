import { useState } from 'react'
import { ChevronDown } from 'lucide-react'

import type { ViewSection } from '../components/ViewPanel'
import { cn } from '../ui'

import { CHEVRON, DETAIL } from './insightsEvidenceStyles'
import { getInsightsRecordSectionPresentation } from './insightsRecordSectionSupport'

interface InsightsRecordSectionProps {
  section: ViewSection
  sectionIndex: number
  /** Drawn inside another card, as a step's evidence, rather than as a card of its own. */
  embedded?: boolean
  /** Open a collapsible section on first render, whatever its presentation says. */
  initiallyExpanded?: boolean
  className?: string
}

const FIELD =
  'flex min-w-0 flex-col gap-1 [&>span]:text-xs [&>span]:text-subtle-foreground [&>div]:min-w-0 [&>div]:break-words [&>div]:text-[13px] [&>div]:text-foreground'

const COLUMNS: Record<number, string> = {
  3: 'sm:grid-cols-3',
  4: 'grid-cols-2 lg:grid-cols-4',
}

export default function InsightsRecordSection({
  section,
  sectionIndex,
  embedded = false,
  initiallyExpanded,
  className,
}: InsightsRecordSectionProps) {
  const presentation = getInsightsRecordSectionPresentation(section.title)
  const [expanded, setExpanded] = useState(
    initiallyExpanded ?? presentation.defaultExpanded ?? true,
  )
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
  const description =
    presentation.description ??
    (presentation.size === 'wide'
      ? `${section.fields.length} ${section.fields.length === 1 ? 'group' : 'groups'}`
      : undefined)

  return (
    <section
      className={cn(
        'flex min-w-0 flex-col',
        embedded
          ? 'rounded-lg border border-border bg-surface-subtle/40'
          : 'rounded-xl border border-border bg-card shadow-xs',
        className,
      )}
      aria-labelledby={`${sectionId}-title`}
    >
      <div
        className={cn(
          'flex items-start justify-between gap-3',
          embedded ? 'px-3.5 py-2.5' : 'px-5 pb-3 pt-4',
        )}
      >
        <div className="flex min-w-0 flex-col gap-0.5">
          <h2
            id={`${sectionId}-title`}
            className={cn('font-semibold tracking-[-0.01em]', embedded ? 'text-[13px]' : 'text-sm')}
          >
            {sectionTitle}
          </h2>
          {description ? (
            <span className="text-xs text-subtle-foreground">{description}</span>
          ) : null}
        </div>
        {presentation.collapsible ? (
          <button
            type="button"
            className="inline-flex h-7 shrink-0 items-center gap-1 rounded-md px-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40"
            aria-expanded={expanded}
            aria-label={`${expanded ? 'Collapse' : 'Expand'} ${sectionTitle}`}
            aria-controls={`${sectionId}-content`}
            onClick={() => setExpanded((current) => !current)}
          >
            <span aria-hidden="true">{expanded ? 'Collapse' : 'Expand'}</span>
            <ChevronDown
              className={cn('size-3.5 transition-transform', expanded && 'rotate-180')}
            />
          </button>
        ) : null}
      </div>

      {expanded ? (
        <div
          id={`${sectionId}-content`}
          className={cn(
            'flex min-w-0 flex-col gap-3 border-t border-border',
            embedded ? 'px-3.5 py-3' : 'px-5 py-4',
          )}
        >
          {presentation.structured ? (
            primaryFields.map((field, fieldIndex) => (
              <div key={`${field.label}-${fieldIndex}`} className="min-w-0">
                {field.value}
              </div>
            ))
          ) : (
            <div
              className={cn(
                'grid grid-cols-1 gap-x-6 gap-y-3.5',
                presentation.metricColumns
                  ? COLUMNS[presentation.metricColumns]
                  : presentation.size === 'wide'
                    ? 'sm:grid-cols-2 xl:grid-cols-3'
                    : 'sm:grid-cols-2',
              )}
            >
              {primaryFields.map((field, fieldIndex) => (
                <div
                  key={`${field.label}-${fieldIndex}`}
                  className={cn(FIELD, field.fullWidth && 'col-span-full')}
                >
                  <span>{field.label}</span>
                  <div>{field.value}</div>
                </div>
              ))}
            </div>
          )}
          {noteFields.map((field) => (
            <p key={field.label} className="text-xs leading-relaxed text-muted-foreground">
              <strong className="font-medium text-foreground/85">{field.label}: </strong>
              {field.value}
            </p>
          ))}
          {secondaryFields.length > 0 ? (
            <details className={DETAIL}>
              <summary>
                <span>{presentation.secondaryTitle || 'More details'}</span>
                <ChevronDown className={CHEVRON} />
              </summary>
              <dl className="flex flex-col gap-2.5 border-t border-border px-3 py-3 text-xs leading-relaxed">
                {secondaryFields.map((field) => (
                  <div key={field.label}>
                    <dt className="font-medium text-foreground/85">{field.label}</dt>
                    <dd className="text-muted-foreground">{field.value}</dd>
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
