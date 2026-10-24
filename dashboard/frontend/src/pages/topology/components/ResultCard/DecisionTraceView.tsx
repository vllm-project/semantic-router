import styles from './DecisionTraceView.module.css'

// Read display fields from the original payload, retaining the Router's JSON
// names. Unknown extensions are never stripped by the transport layer.
type TraceRecord = Record<string, unknown>

function records(value: unknown): TraceRecord[] {
  return Array.isArray(value)
    ? value.filter(
        (node): node is TraceRecord =>
          typeof node === 'object' && node !== null && !Array.isArray(node),
      )
    : []
}

function text(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function stateLabel(node: TraceRecord): string {
  if (node.state === 'unknown') return 'Unknown'
  if (node.signal_error) return 'Error'
  if (node.state === 'true') return 'Match'
  if (node.state === 'false') return 'No match'
  return 'State unavailable'
}

function TraceNode({ node }: { node: TraceRecord }) {
  const label =
    node.node_type === 'leaf'
      ? [node.signal_type, node.signal_name, node.label].map(text).filter(Boolean).join(':')
      : text(node.node_type)
  const children = records(node.children)
  const scored =
    node.state !== 'unknown' &&
    !node.signal_error &&
    node.confidence_scored === true &&
    typeof node.confidence === 'number' &&
    Number.isFinite(node.confidence)
  return (
    <li>
      <div className={styles.node}>
        <span>{label || 'Rule'}</span>
        <strong data-state={text(node.state)}>{stateLabel(node)}</strong>
        {scored && <span>Score {Math.round((node.confidence as number) * 100)}%</span>}
      </div>
      {Boolean(node.signal_error) && (
        <p className={styles.error}>Error: {text(node.signal_error)}</p>
      )}
      {children.length > 0 && (
        <ul>
          {children.map((child, index) => (
            <TraceNode key={index} node={child} />
          ))}
        </ul>
      )}
    </li>
  )
}

export function DecisionTraceView({
  trace,
  selectedDecision,
}: {
  trace?: TraceRecord[]
  selectedDecision: string | null
}) {
  const decisions = records(trace)
  if (decisions.length === 0) return null
  return (
    <section className={styles.trace} aria-label="Decision trace">
      <h3>Decision trace</h3>
      {decisions.map((decision, index) => {
        const name = text(decision.decision_name)
        const root = records([decision.root_trace])[0]
        return (
          <details
            key={`${name}-${index}`}
            open={name === selectedDecision || decision.state === 'unknown'}
          >
            <summary>
              {name || 'Decision'}: {stateLabel(decision)}
              {name === selectedDecision && <span> · Selected</span>}
            </summary>
            {Boolean(decision.on_unknown) && <p>Unknown policy: {text(decision.on_unknown)}</p>}
            {root && (
              <ul>
                <TraceNode node={root} />
              </ul>
            )}
          </details>
        )
      })}
    </section>
  )
}
