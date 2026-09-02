import type { ButtonHTMLAttributes, ReactNode } from 'react'

import type {
  EvaluationRunStatus,
  EvaluationTrackId,
  EvaluationTrackStatus,
  GateVerdict,
} from '../../types/evaluationPlane'
import type { EvaluationCoverage, EvaluationGate } from '../../types/evaluationReport'
import {
  clampFraction,
  gateVerdictPresentation,
  RUN_STATUS_LABELS,
  TRACK_STATUS_LABELS,
} from './evaluationPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import styles from './EvaluationPlane.module.css'

interface EvaluationActionButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'quiet' | 'danger'
  compact?: boolean
}

interface EvaluationTagProps {
  children: ReactNode
  tone?: 'neutral' | 'info' | 'positive' | 'warning' | 'negative'
  mono?: boolean
  title?: string
}

export function EvaluationTag({
  children,
  tone = 'neutral',
  mono = false,
  title,
}: EvaluationTagProps) {
  const toneClass = tone === 'neutral' ? '' : styles[`tag_${tone}`]
  return (
    <span
      className={`${styles.tag} ${toneClass} ${mono ? styles.tagMono : ''}`.trim()}
      data-evaluation-tag="true"
      data-tone={tone}
      title={title}
    >
      {children}
    </span>
  )
}

export function EvaluationActionButton({
  variant = 'secondary',
  compact = false,
  className = '',
  ...props
}: EvaluationActionButtonProps) {
  const variantClass = {
    primary: styles.primaryButton,
    secondary: styles.secondaryButton,
    quiet: styles.quietButton,
    danger: styles.dangerButton,
  }[variant]
  return (
    <button
      {...props}
      data-density={compact ? 'compact' : 'regular'}
      data-evaluation-action="true"
      className={`${variantClass} ${compact ? styles.compactButton : ''} ${className}`.trim()}
    />
  )
}

export function RunStatusBadge({
  status,
}: {
  status: EvaluationRunStatus | EvaluationTrackStatus
}) {
  const label =
    status in RUN_STATUS_LABELS
      ? RUN_STATUS_LABELS[status as EvaluationRunStatus]
      : TRACK_STATUS_LABELS[status]
  return (
    <span
      className={`${styles.badge} ${styles[`status_${status}`]}`}
      data-evaluation-tag="true"
      data-tone={status}
    >
      {label}
    </span>
  )
}

export function GateVerdictBadge({
  verdict,
  disposition = 'advisory',
}: {
  verdict: GateVerdict
  disposition?: EvaluationGate['disposition']
}) {
  const presentation = gateVerdictPresentation({ verdict, disposition })
  return (
    <span
      className={`${styles.badge} ${styles[`gate_${verdict}`]}`}
      data-evaluation-tag="true"
      data-tone={verdict}
      title={presentation.explanation}
    >
      {presentation.label}
    </span>
  )
}

export function TrackChips({ trackIDs }: { trackIDs: EvaluationTrackId[] }) {
  return (
    <div className={styles.chips} aria-label="Evaluation areas">
      {trackIDs.map((trackID) => (
        <EvaluationTag key={trackID} title={TRACK_PRESENTATION[trackID].description}>
          {TRACK_PRESENTATION[trackID].label}
        </EvaluationTag>
      ))}
    </div>
  )
}

export function CoverageBar({ coverage }: { coverage: EvaluationCoverage }) {
  const fraction = clampFraction(coverage.fraction)
  return (
    <div className={styles.coverage}>
      <div className={styles.coverageCopy}>
        <span>Coverage</span>
        <strong>{(fraction * 100).toFixed(1)}%</strong>
      </div>
      <div
        className={styles.progressTrack}
        role="progressbar"
        aria-label="Evaluation coverage"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(fraction * 100)}
      >
        <span style={{ width: `${fraction * 100}%` }} />
      </div>
      <small>
        {coverage.evaluated} of {coverage.total} benchmark checks
        {coverage.unavailable ? ` · ${coverage.unavailable} not measured` : ''}
      </small>
    </div>
  )
}
