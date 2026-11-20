import EvidenceValue from './EvidenceValue'
import RecordedConversation, { RecordedToolCalls } from './RecordedConversation'
import { money, number, seconds } from './model'
import type { CallRecord } from './types'
import { BenchBlock, Disclosure, Metric, MetricGrid } from './benchUi'
import { benchText, evidenceFieldsClass, evidenceTextClass } from './benchStyles'

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
    <div className="grid min-w-0 gap-4">
      <MetricGrid className="my-0 md:grid-cols-3 xl:grid-cols-6">
        <Metric label="Status" value={call.status} />
        <Metric
          label="Selected model"
          value={
            <span className="text-sm font-medium tracking-normal">
              {call.selected_model ?? call.model ?? 'Not recorded'}
            </span>
          }
        />
        <Metric label="Role" value={call.role} />
        <Metric label="Model cost" value={money(call.cost_usd)} />
        <Metric label="Latency" value={seconds(call.latency_s)} />
        <Metric label="Time to first token" value={seconds(call.ttft_s)} />
      </MetricGrid>
      <BenchBlock aria-label="Visible response">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h4 className={benchText.h4}>Visible response</h4>
          <span className={benchText.muted}>Finish · {call.finish_reason ?? 'Not recorded'}</span>
        </div>
        {call.final ? (
          <pre className={evidenceTextClass}>{call.final}</pre>
        ) : (
          <p className={benchText.muted}>No visible final text recorded.</p>
        )}
      </BenchBlock>
      <div className="grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(280px,100%),1fr))] gap-4">
        <BenchBlock aria-label="Token usage">
          <h4 className={`${benchText.h4} mb-3`}>Token usage</h4>
          <dl className={evidenceFieldsClass}>
            {tokens.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{tokenCount(value)}</dd>
              </div>
            ))}
          </dl>
          <p className={`${benchText.muted} mt-3`}>
            Recorded billing buckets. Missing usage is not zero.
          </p>
        </BenchBlock>
        {native && (
          <BenchBlock aria-label="Native output budget">
            <h4 className={`${benchText.h4} mb-3`}>Native output budget</h4>
            <dl className={evidenceFieldsClass}>
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
            <p className={`${benchText.muted} mt-3`}>
              The allowance recorded at dispatch, after input rendering. Actual output is shown in
              token usage.
            </p>
          </BenchBlock>
        )}
      </div>
      <BenchBlock aria-label="Recorded request settings">
        <h4 className={`${benchText.h4} mb-3`}>Recorded request settings</h4>
        {settings.length ? (
          <dl className={evidenceFieldsClass}>
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
          <p className={benchText.muted}>No request settings recorded.</p>
        )}
        <p className={`${benchText.muted} mt-3`}>
          Settings sent with this call; downstream model defaults are not inferred.
        </p>
      </BenchBlock>
      {!!call.tool_calls?.length && (
        <BenchBlock aria-label="Tool calls">
          <h4 className={`${benchText.h4} mb-3`}>Tool calls</h4>
          <RecordedToolCalls calls={call.tool_calls} />
        </BenchBlock>
      )}
      <Disclosure summary="Conversation sent with this call" className="my-0">
        {body?.messages ? (
          <RecordedConversation messages={body.messages} />
        ) : (
          <p className={benchText.muted}>Request conversation not recorded.</p>
        )}
      </Disclosure>
    </div>
  )
}
