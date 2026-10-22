import { effectiveRequestProfile, number } from './model'
import { targetLabel } from './targetPresentation'
import type { Manifest, Target } from './types'
import styles from './SrBench.module.css'
import controls from './BenchControls.module.css'

function settingLabel(key: string) {
  const labels: Record<string, string> = {
    top_k: 'Top K',
    min_p: 'Min P',
    n: 'Responses per request',
    chat_template_kwargs: 'Chat template',
    enable_thinking: 'Thinking',
    ignore_eos: 'Ignore end-of-sequence',
    stop: 'Stop sequences',
  }
  return labels[key] ?? key.replace(/_/g, ' ').replace(/^./, (character) => character.toUpperCase())
}

function FixedValue({ value }: { value: unknown }) {
  if (Array.isArray(value))
    return (
      <ul>
        {value.map((item, index) => (
          <li key={index}>
            <FixedValue value={item} />
          </li>
        ))}
      </ul>
    )
  if (value !== null && typeof value === 'object')
    return (
      <dl>
        {Object.entries(value).map(([key, item]) => (
          <div key={key}>
            <dt>{settingLabel(key)}</dt>
            <dd>
              <FixedValue value={item} />
            </dd>
          </div>
        ))}
      </dl>
    )
  return (
    <>
      {typeof value === 'boolean'
        ? value
          ? 'Enabled'
          : 'Disabled'
        : value === null
          ? 'Not set'
          : String(value)}
    </>
  )
}

export default function TargetRequestProfile({
  target,
  sampling,
  outputPolicy,
}: {
  target: Target
  sampling: Manifest['sampling']
  outputPolicy?: Manifest['output_policy']
}) {
  const effective = effectiveRequestProfile(target, sampling)
  const fields = [
    ['max_tokens', 'Output tokens'],
    ['temperature', 'Temperature'],
    ['top_p', 'Top P'],
    ['seed', 'Sampling seed'],
    ['reasoning_effort', 'Reasoning effort'],
  ] as const
  const shown = new Set<string>(fields.map(([key]) => key))
  const other = Object.entries(target.request_params ?? {}).filter(([key]) => !shown.has(key))
  return (
    <section aria-label={`${targetLabel(target)} request profile`}>
      <h4>Effective request profile</h4>
      {outputPolicy === 'native' && (
        <p className={styles.muted}>Native capacity · no shared output-token cap.</p>
      )}
      <dl className={controls.profileGrid}>
        {fields.map(([key, label]) => {
          if (outputPolicy === 'native' && key === 'max_tokens') return null
          const value = effective[key]
          if (value === undefined || value === null) return null
          const fixed = target.request_params?.[key] !== undefined
          return (
            <div key={key}>
              <dt>{label}</dt>
              <dd>
                {typeof value === 'number' ? number(value, 3) : String(value)}
                {fixed && <span className={controls.fixedBadge}>Fixed</span>}
              </dd>
            </div>
          )
        })}
      </dl>
      {outputPolicy === 'native' && target.native_limits && (
        <details className={controls.fixedSettings}>
          <summary>Registered model capacities</summary>
          <p className={styles.muted}>
            Model maxima, not the output budget of a particular request. Actual input uses part of
            the context window.
          </p>
          <dl>
            {Object.entries(target.native_limits).map(([model, limits]) => (
              <div key={model}>
                <dt>{model}</dt>
                <dd>
                  {number(limits.context_window)} context tokens ·{' '}
                  {number(limits.max_output_tokens)} maximum output tokens
                </dd>
              </div>
            ))}
          </dl>
        </details>
      )}
      {other.length > 0 && (
        <details className={controls.fixedSettings}>
          <summary>Other fixed settings</summary>
          <dl>
            {other.map(([key, value]) => (
              <div key={key}>
                <dt>{settingLabel(key)}</dt>
                <dd>
                  <FixedValue value={value} />
                </dd>
              </div>
            ))}
          </dl>
        </details>
      )}
      {target.request_params && Object.keys(target.request_params).length > 0 && (
        <p className={styles.muted}>
          Fixed settings belong to this target. Choose another target to use a different profile.
        </p>
      )}
    </section>
  )
}
