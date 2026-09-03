import type { DetailsHTMLAttributes, ReactNode } from 'react'
import styles from './EvaluationDisclosure.module.css'

type EvaluationDisclosureIndicator = 'chevron' | 'label'
type EvaluationDisclosureFocus = 'inset' | 'outside'

const INDICATOR_CLASSES: Record<EvaluationDisclosureIndicator, string> = {
  chevron: styles.indicatorChevron,
  label: styles.indicatorLabel,
}

const FOCUS_CLASSES: Record<EvaluationDisclosureFocus, string> = {
  inset: styles.focusInset,
  outside: styles.focusOutside,
}

interface EvaluationDisclosureProps
  extends Omit<DetailsHTMLAttributes<HTMLDetailsElement>, 'children'> {
  children: ReactNode
  summary: ReactNode
  summaryClassName?: string
  indicator?: EvaluationDisclosureIndicator
  focus?: EvaluationDisclosureFocus
}

type EvaluationTechnicalDisclosureProps = Omit<EvaluationDisclosureProps, 'focus' | 'open'>

export default function EvaluationDisclosure({
  children,
  className,
  focus = 'inset',
  indicator = 'chevron',
  summary,
  summaryClassName,
  ...detailsProps
}: EvaluationDisclosureProps) {
  return (
    <details
      {...detailsProps}
      className={[styles.disclosure, className].filter(Boolean).join(' ')}
    >
      <summary
        className={[
          styles.summary,
          INDICATOR_CLASSES[indicator],
          FOCUS_CLASSES[focus],
          summaryClassName,
        ]
          .filter(Boolean)
          .join(' ')}
      >
        {summary}
      </summary>
      {children}
    </details>
  )
}

/**
 * Shared boundary for raw service responses and internal evaluation
 * identities presented as technical details. Technical content is always
 * collapsed initially and uses one focus treatment, so callers cannot
 * accidentally surface it as product copy or opt it open by default.
 */
export function EvaluationTechnicalDisclosure(props: EvaluationTechnicalDisclosureProps) {
  return (
    <EvaluationDisclosure {...props} data-evaluation-technical-details="true" focus="outside" />
  )
}
