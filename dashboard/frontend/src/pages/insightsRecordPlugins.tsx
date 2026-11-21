import { ChevronDown } from 'lucide-react'

import type { ViewField } from '../components/ViewPanel'
import { CHEVRON, DETAIL } from './insightsEvidenceStyles'
import type { InsightsOutcome, InsightsRecord } from './insightsPageTypes'

const STACK = 'flex min-w-0 flex-col gap-1'
const SUBTLE = 'text-xs text-subtle-foreground'
const DANGER = 'text-[13px] font-medium text-destructive-foreground'
const WARN = 'text-[13px] font-medium text-warning-foreground'
const INFO = 'text-[13px] font-medium text-info-foreground'

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

function riskScore(value: number | null | undefined, available?: boolean) {
  return available && finite(value) ? `${(value * 100).toFixed(1)}%` : 'Score unavailable'
}

function responseOutcomes(record: InsightsRecord, signal: string) {
  return (record.outcomes ?? []).filter(
    (outcome) =>
      outcome.source === 'router' &&
      outcome.metadata?.stage === 'response' &&
      outcome.metadata?.signal === signal,
  )
}

const verdictLabels: Record<string, string> = {
  detected: 'Detected',
  not_detected: 'Not detected',
  unavailable: 'Unavailable',
  not_applicable: 'Not applicable',
}

// How a streamed answer reached the client: held until the guard read it, or
// as it streamed, before the guard read it.
const streamLabels: Record<string, string> = {
  buffered: 'Stream held for the guard',
  unbuffered: 'Streamed before the check',
}

function outcomeEvidence(outcomes: InsightsOutcome[]) {
  const counts = new Map<string, number>()
  for (const outcome of outcomes) {
    const label = verdictLabels[outcome.verdict] ?? outcome.verdict.replace(/_/g, ' ')
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  return (
    <details className={DETAIL}>
      <summary>
        {Array.from(counts, ([label, count]) => `${count} ${label.toLowerCase()}`).join(' · ')}
        <span className="ml-auto font-normal text-subtle-foreground">Inspect recorded checks</span>
        <ChevronDown className={CHEVRON} />
      </summary>
      {outcomes.map((outcome, index) => (
        <div
          className={`${STACK} border-t border-border px-3 py-2`}
          key={`${outcome.target}-${index}`}
        >
          <strong>{outcome.target}</strong>
          <span>{verdictLabels[outcome.verdict] ?? outcome.verdict}</span>
          {outcome.reason ? <span className={SUBTLE}>{outcome.reason}</span> : null}
          <span className={SUBTLE}>
            {outcome.metadata?.score_available === 'true'
              ? `Score: ${finite(outcome.score) ? outcome.score.toFixed(4) : outcome.score === undefined ? '0.0000' : 'Not recorded'}${outcome.metadata.score_kind ? ` (${outcome.metadata.score_kind})` : ''}`
              : 'Score unavailable'}
          </span>
          {outcome.metadata?.action ? (
            <span className={SUBTLE}>Action: {outcome.metadata.action}</span>
          ) : null}
          {outcome.metadata?.stream ? (
            <span className={SUBTLE}>
              {streamLabels[outcome.metadata.stream] ?? outcome.metadata.stream}
            </span>
          ) : null}
        </div>
      ))}
    </details>
  )
}

function guardrails(record: InsightsRecord) {
  const responses = responseOutcomes(record, 'jailbreak')
  const requestObserved = record.jailbreak_detected || record.jailbreak_score_available
  if (
    !record.guardrails_enabled &&
    !record.jailbreak_enabled &&
    !record.pii_enabled &&
    !requestObserved &&
    !record.pii_detected &&
    !responses.length
  )
    return 'Disabled'

  return (
    <div className={STACK}>
      {requestObserved ? (
        <span className={record.jailbreak_detected ? DANGER : SUBTLE}>
          Request jailbreak:{' '}
          {record.jailbreak_detected ? record.jailbreak_type || 'detected' : 'Not detected'} (
          {riskScore(record.jailbreak_confidence, record.jailbreak_score_available)})
        </span>
      ) : record.jailbreak_enabled ? (
        <span>No request jailbreak result recorded</span>
      ) : null}
      {responses.length ? (
        <div className={STACK}>
          <span>Response jailbreak</span>
          {outcomeEvidence(responses)}
        </div>
      ) : null}
      {record.pii_detected ? (
        <span className={record.pii_blocked ? DANGER : WARN}>
          {record.pii_blocked ? 'PII blocked' : 'PII found'}:{' '}
          {record.pii_entities?.join(', ') || 'detected'}
        </span>
      ) : record.pii_enabled ? (
        <span>No PII detection recorded</span>
      ) : null}
      {record.guardrails_enabled &&
      !record.jailbreak_enabled &&
      !record.pii_enabled &&
      !requestObserved &&
      !record.pii_detected &&
      !responses.length ? (
        <span>Enabled; no detection result recorded</span>
      ) : null}
    </div>
  )
}

function hallucination(record: InsightsRecord) {
  const outcomes = responseOutcomes(record, 'hallucination')
  const spans = record.hallucination_spans?.length ? (
    <details className={DETAIL}>
      <summary>
        Inspect {record.hallucination_spans.length} unsupported spans
        <ChevronDown className={CHEVRON} />
      </summary>
      <ul className="flex list-disc flex-col gap-1 border-t border-border py-2 pl-7 pr-3 text-[13px]">
        {record.hallucination_spans.map((span, index) => (
          <li key={index}>{span}</li>
        ))}
      </ul>
    </details>
  ) : null
  if (outcomes.length) {
    return (
      <div className={STACK}>
        {outcomeEvidence(outcomes)}
        {spans}
      </div>
    )
  }
  if (
    !record.hallucination_enabled &&
    !record.hallucination_detected &&
    !record.hallucination_score_available &&
    !spans
  )
    return 'Disabled'
  // Unlike the jailbreak JSON wire fields, this Go float uses omitempty.
  // A positive availability marker with an omitted value encodes a measured zero.
  const score = record.hallucination_confidence === undefined ? 0 : record.hallucination_confidence
  const available = record.hallucination_score_available === true && finite(score)
  return (
    <div className={STACK}>
      <span className={record.hallucination_detected ? DANGER : SUBTLE}>
        {record.hallucination_detected
          ? 'Detected'
          : available
            ? 'Not detected'
            : 'No detection result recorded'}
      </span>
      <span className={SUBTLE}>
        {available
          ? `Score: ${score.toFixed(4)}${record.hallucination_score_kind ? ` (${record.hallucination_score_kind})` : ''}`
          : 'Score unavailable'}
      </span>
      {spans}
    </div>
  )
}

export function buildInsightsPluginFields(record: InsightsRecord): ViewField[] {
  return [
    { label: 'Cache', value: record.from_cache ? 'Hit' : 'Miss' },
    {
      label: 'Cache similarity',
      value: finite(record.cache_similarity) ? record.cache_similarity.toFixed(3) : 'Not recorded',
    },
    { label: 'Streaming', value: record.streaming ? 'On' : 'Off' },
    { label: 'Guardrails', value: guardrails(record) },
    {
      label: 'RAG',
      value: record.rag_enabled ? (
        <div className={STACK}>
          <span className={INFO}>Context retrieved</span>
          <span className={SUBTLE}>
            Backend: {record.rag_backend || 'Not recorded'} · Length:{' '}
            {record.rag_context_length ?? 0} chars
            {' · '}Score:{' '}
            {finite(record.rag_similarity_score)
              ? record.rag_similarity_score.toFixed(3)
              : 'Not recorded'}
          </span>
        </div>
      ) : (
        'Not used'
      ),
    },
    { label: 'Hallucination Detection', value: hallucination(record) },
  ]
}
