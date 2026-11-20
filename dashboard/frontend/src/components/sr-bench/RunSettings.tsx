import { useId } from 'react'
import { Settings2 } from 'lucide-react'
import { Input } from '../../ui'
import BenchSelect from './BenchSelect'
import type { Manifest, Target } from './types'
import { Disclosure, Notice } from './benchUi'
import { benchText } from './benchStyles'

export type SamplingSettings = Pick<Manifest['sampling'], 'temperature' | 'top_p' | 'seed'>

const formGrid = 'my-3 grid min-w-0 grid-cols-1 gap-4 sm:grid-cols-2'

interface NumberFieldProps {
  label: string
  value: number | ''
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  help?: string
  disabled?: boolean
  placeholder?: string
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  help,
  disabled,
  placeholder,
}: NumberFieldProps) {
  const id = useId()
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <label htmlFor={id} className="text-xs font-medium text-muted-foreground">
        {label}
      </label>
      <Input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        className="tabular-nums"
        value={typeof value === 'number' && !Number.isFinite(value) ? '' : value}
        disabled={disabled}
        placeholder={placeholder}
        aria-describedby={help ? `${id}-help` : undefined}
        onChange={(event) => onChange(event.target.valueAsNumber)}
      />
      {help && (
        <small id={`${id}-help`} className="text-[11.5px] leading-snug text-subtle-foreground">
          {help}
        </small>
      )}
    </div>
  )
}

interface Props {
  limits: Manifest['limits']
  onLimitsChange: (limits: Manifest['limits']) => void
  sampling: SamplingSettings
  onSamplingChange: (sampling: SamplingSettings) => void
  seed: number
  targets: Target[]
  costPolicy: Manifest['cost_policy']
  outputPolicy: 'bounded' | 'native'
  onOutputPolicyChange: (policy: 'bounded' | 'native') => void
  nativeAvailable: boolean
  onCostPolicyChange: (policy: 'require_priced' | 'capability_only') => void
  mode: Manifest['mode']
  previewContext: NonNullable<Manifest['preview_context']>
  onPreviewContextChange: (context: NonNullable<Manifest['preview_context']>) => void
}

export default function RunSettings({
  limits,
  onLimitsChange,
  sampling,
  onSamplingChange,
  seed,
  targets,
  costPolicy,
  outputPolicy,
  onOutputPolicyChange,
  nativeAvailable,
  onCostPolicyChange,
  mode,
  previewContext,
  onPreviewContextChange,
}: Props) {
  const changeLimit = (key: keyof Manifest['limits']) => (value: number) =>
    onLimitsChange({ ...limits, [key]: value })
  function samplingField(key: keyof SamplingSettings, fallback: number) {
    const fixed = targets.filter((target) => typeof target.request_params?.[key] === 'number')
    const allFixed = targets.length > 0 && fixed.length === targets.length
    const values = new Set(fixed.map((target) => target.request_params?.[key] as number))
    return {
      value: allFixed
        ? values.size === 1
          ? [...values][0]
          : ('' as const)
        : (sampling[key] ?? fallback),
      disabled: allFixed,
      placeholder: 'Varies by target',
      help: allFixed
        ? 'Fixed by the selected targets. Effective values are shown above.'
        : fixed.length
          ? `Applies to ${targets.length - fixed.length} of ${targets.length} targets; fixed profiles keep their own values.`
          : 'Default for selected targets.',
      onChange: (value: number) => onSamplingChange({ ...sampling, [key]: value }),
    }
  }
  return (
    <>
      <div className="max-w-md">
        <BenchSelect
          label="Output policy"
          value={outputPolicy}
          onChange={(value) => onOutputPolicyChange(value as 'bounded' | 'native')}
          options={[
            {
              value: 'bounded',
              label: 'Bounded output',
              description: 'Set a shared output-token limit.',
            },
            ...(nativeAvailable || outputPolicy === 'native'
              ? [
                  {
                    value: 'native',
                    label: 'Native capacity',
                    description: 'Use available model capacity without a shared token cap.',
                  },
                ]
              : []),
          ]}
        />
      </div>
      <p className={`${benchText.muted} mt-1.5 max-w-3xl`}>
        {outputPolicy === 'native'
          ? 'Use each selected model’s available context for reasoning and its answer. No fixed shared token cap is sent; remaining output capacity depends on the actual input.'
          : !nativeAvailable
            ? 'Native capacity requires operator-registered model limits and a compatible target profile.'
            : 'A shared cap bounds each response; registered target overrides still apply.'}
      </p>
      <div className={formGrid}>
        <NumberField
          label="Budget (USD)"
          value={costPolicy === 'capability_only' ? '' : limits.max_cost_usd}
          onChange={changeLimit('max_cost_usd')}
          min={0.01}
          step={0.01}
          disabled={costPolicy === 'capability_only'}
          placeholder={costPolicy === 'capability_only' ? 'Not applied' : undefined}
          help={
            costPolicy === 'capability_only'
              ? 'A USD budget requires complete cost accounting.'
              : 'Spend threshold; in-flight requests may exceed it.'
          }
        />
        <NumberField
          label="Run deadline (seconds)"
          value={limits.max_run_seconds}
          onChange={changeLimit('max_run_seconds')}
          min={1}
        />
        {outputPolicy === 'bounded' && (
          <NumberField
            label="Max output tokens"
            value={limits.max_output_tokens ?? ''}
            onChange={changeLimit('max_output_tokens')}
            min={1}
            help="Per response. Must accommodate any fixed target profile."
          />
        )}
        <NumberField
          label="Concurrency"
          value={limits.concurrency}
          onChange={changeLimit('concurrency')}
          min={1}
          max={32}
          help="Cases processed in parallel."
        />
      </div>
      <Disclosure
        summary={
          <>
            <Settings2 aria-hidden="true" className="size-4 text-muted-foreground" />
            Sampling and advanced limits
          </>
        }
        bodyClassName="flex flex-col gap-1 [&_h4]:mt-3 [&_h4:first-child]:mt-0"
      >
        {mode === 'preview' && (
          <>
            <h4 className={benchText.h4}>Preview context</h4>
            <p className={benchText.muted}>
              Optionally inspect routing for an existing session. Preview reads the current learning
              state without updating it.
            </p>
            <div className={formGrid}>
              <label className={benchText.field}>
                Session ID
                <Input
                  value={previewContext.session_id ?? ''}
                  placeholder="Optional"
                  onChange={(event) =>
                    onPreviewContextChange({ ...previewContext, session_id: event.target.value })
                  }
                />
              </label>
              <label className={benchText.field}>
                Conversation ID
                <Input
                  value={previewContext.conversation_id ?? ''}
                  placeholder="Optional"
                  onChange={(event) =>
                    onPreviewContextChange({
                      ...previewContext,
                      conversation_id: event.target.value,
                    })
                  }
                />
              </label>
              <NumberField
                label="Preview sampling seed"
                value={previewContext.sampling_seed ?? seed}
                onChange={(value) =>
                  onPreviewContextChange({ ...previewContext, sampling_seed: value })
                }
                help="Reproduces preview sampling. Later live routing can change with the learning state."
              />
            </div>
          </>
        )}
        {mode === 'live' && (
          <>
            <h4 className={benchText.h4}>Cost accounting</h4>
            <div className="my-2 max-w-md">
              <BenchSelect
                label="Cost accounting"
                value={costPolicy ?? 'require_priced'}
                onChange={(value) =>
                  onCostPolicyChange(value as 'require_priced' | 'capability_only')
                }
                options={[
                  {
                    value: 'require_priced',
                    label: 'Quality and cost',
                    description: 'Require complete prices and a model-cost budget.',
                  },
                  {
                    value: 'capability_only',
                    label: 'Quality only',
                    description: 'Allow incomplete prices. No cost-saving claims.',
                  },
                ]}
              />
            </div>
            {costPolicy === 'capability_only' && (
              <Notice tone="info" as="p">
                No USD budget is applied. Time, output and call limits still apply. Known costs are
                recorded, but this run cannot prove cost savings.
              </Notice>
            )}
          </>
        )}
        <h4 className={benchText.h4}>Sampling defaults</h4>
        <p className={benchText.muted}>
          Fixed target profiles take precedence. Their effective settings remain visible above.
        </p>
        <div className={`${formGrid} lg:grid-cols-3`}>
          <NumberField
            label="Temperature"
            min={0}
            max={2}
            step={0.1}
            {...samplingField('temperature', 0)}
          />
          <NumberField label="Top P" min={0} max={1} step={0.05} {...samplingField('top_p', 1)} />
          <NumberField label="Sampling seed" {...samplingField('seed', seed)} />
        </div>
        <h4 className={benchText.h4}>Request and case limits</h4>
        <div className={formGrid}>
          <NumberField
            label="Request deadline (seconds)"
            value={limits.total_timeout_s}
            onChange={changeLimit('total_timeout_s')}
            min={1}
            help="Maximum duration of one model request."
          />
          <NumberField
            label="Idle timeout (seconds)"
            value={limits.idle_timeout_s}
            onChange={changeLimit('idle_timeout_s')}
            min={1}
            help="Stop a request if no output arrives in this interval."
          />
          <NumberField
            label="Case deadline (seconds)"
            value={limits.case_timeout_s ?? 600}
            onChange={changeLimit('case_timeout_s')}
            min={1}
            help="Maximum duration of a case, including multiple turns."
          />
          <NumberField
            label="Max calls per case"
            value={limits.max_calls_per_case}
            onChange={changeLimit('max_calls_per_case')}
            min={1}
            help="Bounds model calls in multi-turn and agent benchmarks."
          />
        </div>
      </Disclosure>
    </>
  )
}
