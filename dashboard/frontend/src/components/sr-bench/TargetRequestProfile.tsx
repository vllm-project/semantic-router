import { Badge } from '../../ui'
import { effectiveRequestProfile, number } from './model'
import { targetLabel } from './targetPresentation'
import type { Manifest, Target } from './types'
import { Disclosure } from './benchUi'
import { benchText } from './benchStyles'

const fixedSettingsClass =
  'my-2 grid gap-2 text-xs [&>div]:grid [&>div]:grid-cols-[minmax(100px,160px)_minmax(0,1fr)] [&>div]:gap-2 [&_dt]:text-muted-foreground [&_dd]:min-w-0 [&_dd]:break-words [&_dd]:text-foreground [&_ul]:list-disc [&_ul]:pl-4'

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
      <dl className="m-0 grid gap-1">
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
    <section aria-label={`${targetLabel(target)} request profile`} className="mt-3 min-w-0">
      <h4 className={`${benchText.h4} text-xs uppercase tracking-[0.04em] text-muted-foreground`}>
        Effective request profile
      </h4>
      {outputPolicy === 'native' && (
        <p className={benchText.muted}>Native capacity · no shared output-token cap.</p>
      )}
      <dl className="my-2 grid grid-cols-[repeat(auto-fit,minmax(110px,1fr))] gap-3 [&_dt]:text-[11px] [&_dt]:text-muted-foreground [&_dd]:mt-0.5 [&_dd]:flex [&_dd]:flex-wrap [&_dd]:items-center [&_dd]:gap-1.5 [&_dd]:text-[13px] [&_dd]:font-medium [&_dd]:text-foreground [&_dd]:tabular-nums">
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
                {fixed && (
                  <Badge tone="warning" size="sm">
                    Fixed
                  </Badge>
                )}
              </dd>
            </div>
          )
        })}
      </dl>
      {outputPolicy === 'native' && target.native_limits && (
        <Disclosure summary="Registered model capacities" bare className="my-1">
          <p className={benchText.muted}>
            Model maxima, not the output budget of a particular request. Actual input uses part of
            the context window.
          </p>
          <dl className={fixedSettingsClass}>
            {Object.entries(target.native_limits).map(([model, limits]) => (
              <div key={model}>
                <dt className="font-mono">{model}</dt>
                <dd>
                  {number(limits.context_window)} context tokens ·{' '}
                  {number(limits.max_output_tokens)} maximum output tokens
                </dd>
              </div>
            ))}
          </dl>
        </Disclosure>
      )}
      {other.length > 0 && (
        <Disclosure summary="Other fixed settings" bare className="my-1">
          <dl className={fixedSettingsClass}>
            {other.map(([key, value]) => (
              <div key={key}>
                <dt>{settingLabel(key)}</dt>
                <dd>
                  <FixedValue value={value} />
                </dd>
              </div>
            ))}
          </dl>
        </Disclosure>
      )}
      {target.request_params && Object.keys(target.request_params).length > 0 && (
        <p className={benchText.muted}>
          Fixed settings belong to this target. Choose another target to use a different profile.
        </p>
      )}
    </section>
  )
}
