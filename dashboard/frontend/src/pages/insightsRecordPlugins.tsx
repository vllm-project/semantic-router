import type { ViewField } from '../components/ViewPanel'
import ProductIcon from '../components/ProductIcon'
import type { InsightsOutcome, InsightsRecord } from './insightsPageTypes'
import styles from './InsightsPage.module.css'

const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

function riskScore(value: number | null | undefined, available?: boolean) {
  return available && finite(value) ? `${(value * 100).toFixed(1)}%` : 'Score unavailable'
}

function responseOutcomes(record: InsightsRecord, signal: string) {
  return (record.outcomes ?? []).filter(
    (outcome) =>
      outcome.source === 'router' &&
      outcome.metadata?.direction === 'response' &&
      outcome.metadata?.signal === signal,
  )
}

const verdictLabels: Record<string, string> = {
  detected: 'Detected',
  not_detected: 'Not detected',
  unavailable: 'Unavailable',
  not_applicable: 'Not applicable',
}

function outcomeEvidence(outcomes: InsightsOutcome[]) {
  const counts = new Map<string, number>()
  for (const outcome of outcomes) {
    const label = verdictLabels[outcome.verdict] ?? outcome.verdict.replace(/_/g, ' ')
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  return (
    <details className={styles.pluginEvidence}>
      <summary>
        {Array.from(counts, ([label, count]) => `${count} ${label.toLowerCase()}`).join(' · ')}
        <span className={styles.costSubtle}>Inspect recorded checks</span>
        <ProductIcon name="chevron-down" width={14} height={14} />
      </summary>
      {outcomes.map((outcome, index) => (
        <div className={styles.pluginStack} key={`${outcome.target}-${index}`}>
          <strong>{outcome.target}</strong>
          <span>{verdictLabels[outcome.verdict] ?? outcome.verdict}</span>
          {outcome.reason ? <span className={styles.costSubtle}>{outcome.reason}</span> : null}
          <span className={styles.costSubtle}>
            {outcome.metadata?.score_available === 'true'
              ? `Score: ${finite(outcome.score) ? outcome.score.toFixed(4) : outcome.score === undefined ? '0.0000' : 'Not recorded'}${outcome.metadata.score_kind ? ` (${outcome.metadata.score_kind})` : ''}`
              : 'Score unavailable'}
          </span>
          {outcome.metadata?.enforcement === 'not_enforced_streaming' ? (
            <span className={styles.costSubtle}>Not enforced for streaming</span>
          ) : outcome.metadata?.action ? (
            <span className={styles.costSubtle}>Action: {outcome.metadata.action}</span>
          ) : null}
        </div>
      ))}
    </details>
  )
}

function guardrails(record: InsightsRecord) {
  const responses = responseOutcomes(record, 'jailbreak')
  const requestObserved = record.jailbreak_detected || record.jailbreak_score_available
  const responseObserved =
    record.response_jailbreak_detected || record.response_jailbreak_score_available
  if (
    !record.guardrails_enabled &&
    !record.jailbreak_enabled &&
    !record.pii_enabled &&
    !requestObserved &&
    !responseObserved &&
    !record.pii_detected &&
    !responses.length
  )
    return 'Disabled'

  return (
    <div className={styles.pluginStack}>
      {requestObserved ? (
        <span className={record.jailbreak_detected ? styles.alertDanger : styles.costSubtle}>
          Request jailbreak:{' '}
          {record.jailbreak_detected ? record.jailbreak_type || 'detected' : 'Not detected'} (
          {riskScore(record.jailbreak_confidence, record.jailbreak_score_available)})
        </span>
      ) : record.jailbreak_enabled ? (
        <span>No request jailbreak result recorded</span>
      ) : null}
      {responses.length ? (
        <div className={styles.pluginStack}>
          <span>Response jailbreak</span>
          {outcomeEvidence(responses)}
        </div>
      ) : responseObserved ? (
        <span
          className={record.response_jailbreak_detected ? styles.alertDanger : styles.costSubtle}
        >
          Response jailbreak:{' '}
          {record.response_jailbreak_detected
            ? record.response_jailbreak_type || 'detected'
            : 'Not detected'}{' '}
          (
          {riskScore(
            record.response_jailbreak_confidence,
            record.response_jailbreak_score_available,
          )}
          )
        </span>
      ) : null}
      {record.pii_detected ? (
        <span className={record.pii_blocked ? styles.alertDanger : styles.alertWarn}>
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
      !responseObserved &&
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
    <details className={styles.pluginEvidence}>
      <summary>
        Inspect {record.hallucination_spans.length} unsupported spans
        <ProductIcon name="chevron-down" width={14} height={14} />
      </summary>
      <ul>
        {record.hallucination_spans.map((span, index) => (
          <li key={index}>{span}</li>
        ))}
      </ul>
    </details>
  ) : null
  if (outcomes.length) {
    return (
      <div className={styles.pluginStack}>
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
    <div className={styles.pluginStack}>
      <span className={record.hallucination_detected ? styles.alertDanger : styles.costSubtle}>
        {record.hallucination_detected
          ? 'Detected'
          : available
            ? 'Not detected'
            : 'No detection result recorded'}
      </span>
      <span className={styles.costSubtle}>
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
        <div className={styles.pluginStack}>
          <span className={styles.alertInfo}>Context retrieved</span>
          <span className={styles.costSubtle}>
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
