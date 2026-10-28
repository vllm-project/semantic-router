import EvidenceValue from './EvidenceValue'
import RecordedConversation, { RecordedToolCalls } from './RecordedConversation'
import { money, number, seconds } from './model'
import type { CallRecord } from './types'
import styles from './CallDetail.module.css'
import shared from './SrBench.module.css'

function tokenCount(value: number | null | undefined) {
  return value === null || value === undefined ? 'Not recorded' : number(value)
}

function settingLabel(key: string) {
  const labels: Record<string, string> = {
    max_tokens: 'Requested output tokens',
    reasoning_effort: 'Reasoning effort',
    top_p: 'Top P',
    top_k: 'Top K',
    min_p: 'Min P',
    chat_template_kwargs: 'Chat template',
  }
  return labels[key] ?? key.replace(/_/g, ' ').replace(/^./, (character) => character.toUpperCase())
}

export default function CallDetail({ call }: { call: CallRecord }) {
  const body = call.request?.effective_body
  const native = call.native_output
  const excluded = new Set(['messages', 'model', 'tools', 'stream', 'stream_options'])
  // Native dispatch derives the final allowance after rendering the input. The
  // original request body can still contain a different pre-render max_tokens.
  if (native) excluded.add('max_tokens')
  const settings = Object.entries(body ?? {}).filter(([key]) => !excluded.has(key))
  const tokens = [
    ['Input tokens', call.usage?.input_tokens],
    ['Cached input tokens', call.usage?.cached_input_tokens],
    ['Cache write tokens', call.usage?.cache_write_tokens],
    ['Output tokens', call.usage?.output_tokens],
  ] as const
  return (
    <div className={styles.detail}>
      <div className={shared.metricGrid}>
        <div>
          <span>Status</span>
          <strong>{call.status}</strong>
        </div>
        <div>
          <span>Selected model</span>
          <strong>{call.selected_model ?? call.model ?? 'Not recorded'}</strong>
        </div>
        <div>
          <span>Role</span>
          <strong>{call.role}</strong>
        </div>
        <div>
          <span>Model cost</span>
          <strong>{money(call.cost_usd)}</strong>
        </div>
        <div>
          <span>Latency</span>
          <strong>{seconds(call.latency_s)}</strong>
        </div>
        <div>
          <span>Time to first token</span>
          <strong>{seconds(call.ttft_s)}</strong>
        </div>
      </div>
      <section aria-label="Visible response" className={styles.section}>
        <div className={shared.sectionHeading}>
          <h4>Visible response</h4>
          <span className={shared.muted}>Finish · {call.finish_reason ?? 'Not recorded'}</span>
        </div>
        {call.final ? (
          <pre className={styles.text}>{call.final}</pre>
        ) : (
          <p className={shared.muted}>No visible final text recorded.</p>
        )}
      </section>
      <div className={styles.budgets}>
        <section aria-label="Token usage" className={styles.section}>
          <h4>Token usage</h4>
          <dl className={styles.fields}>
            {tokens.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{tokenCount(value)}</dd>
              </div>
            ))}
          </dl>
          <p className={shared.muted}>Recorded billing buckets. Missing usage is not zero.</p>
        </section>
        {native && (
          <section aria-label="Native output budget" className={styles.section}>
            <h4>Native output budget</h4>
            <dl className={styles.fields}>
              <div>
                <dt>Input tokens</dt>
                <dd>{tokenCount(native.input_tokens)}</dd>
              </div>
              <div>
                <dt>Context window</dt>
                <dd>{tokenCount(native.context_window)}</dd>
              </div>
              <div>
                <dt>Available output tokens</dt>
                <dd>{tokenCount(native.max_output_tokens)}</dd>
              </div>
              <div>
                <dt>Registered output maximum</dt>
                <dd>{tokenCount(native.configured_max_output_tokens)}</dd>
              </div>
            </dl>
            <p className={shared.muted}>
              The allowance recorded at dispatch, after input rendering. Actual output is shown in
              token usage.
            </p>
          </section>
        )}
      </div>
      <section aria-label="Recorded request settings" className={styles.section}>
        <h4>Recorded request settings</h4>
        {settings.length ? (
          <dl className={styles.fields}>
            {settings.map(([key, value]) => (
              <div key={key}>
                <dt>{settingLabel(key)}</dt>
                <dd>
                  <EvidenceValue value={value} />
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className={shared.muted}>No request settings recorded.</p>
        )}
        <p className={shared.muted}>
          Settings sent with this call; downstream model defaults are not inferred.
        </p>
      </section>
      {!!call.tool_calls?.length && (
        <section aria-label="Tool calls" className={styles.section}>
          <h4>Tool calls</h4>
          <RecordedToolCalls calls={call.tool_calls} />
        </section>
      )}
      <details className={shared.details}>
        <summary>Conversation sent with this call</summary>
        {body?.messages ? (
          <RecordedConversation messages={body.messages} />
        ) : (
          <p className={shared.muted}>Request conversation not recorded.</p>
        )}
      </details>
    </div>
  )
}
