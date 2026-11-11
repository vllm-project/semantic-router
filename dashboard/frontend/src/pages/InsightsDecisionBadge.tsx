import { decisionTone, type Tone } from './insightsDecisionTone'
import styles from './InsightsPage.module.css'

const ICONS: Record<Tone, string> = {
  guard: 'M12 3l7 3v5c0 4.5-3 8.3-7 9.5C8 19.3 5 15.5 5 11V6l7-3z',
  privacy: 'M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z',
  deep: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z',
  fast: 'M13 2L4 14h7l-1 8 9-12h-7l1-8z',
  route: 'M5 6h8a4 4 0 0 1 0 8H9a4 4 0 0 0 0 8h10M17 3l3 3-3 3',
}

// A decision as a tinted pill with an icon; `label` is the name to show.
export default function InsightsDecisionBadge({
  decision,
  label,
}: {
  decision: string
  label: string
}) {
  const tone = decisionTone(decision)
  return (
    <span className={styles.decisionBadge} data-tone={tone} title={decision}>
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d={ICONS[tone]} />
      </svg>
      {label}
    </span>
  )
}
