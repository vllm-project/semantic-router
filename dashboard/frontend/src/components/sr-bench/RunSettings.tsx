import { useId } from 'react'
import ProductIcon from '../ProductIcon'
import type { Manifest, Target } from './types'
import styles from './SrBench.module.css'
import controls from './BenchControls.module.css'

export type SamplingSettings = Pick<Manifest['sampling'], 'temperature' | 'top_p' | 'seed'>

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
    <div className={controls.selectField}>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        value={typeof value === 'number' && !Number.isFinite(value) ? '' : value}
        disabled={disabled}
        placeholder={placeholder}
        aria-describedby={help ? `${id}-help` : undefined}
        onChange={(event) => onChange(event.target.valueAsNumber)}
      />
      {help && (
        <small id={`${id}-help`} className={controls.fieldHelp}>
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
      <div className={styles.formGrid}>
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
        <NumberField
          label="Max output tokens"
          value={limits.max_output_tokens}
          onChange={changeLimit('max_output_tokens')}
          min={1}
          help="Per response. Must accommodate any fixed target profile."
        />
        <NumberField
          label="Concurrency"
          value={limits.concurrency}
          onChange={changeLimit('concurrency')}
          min={1}
          max={32}
          help="Cases processed in parallel."
        />
      </div>
      <details className={controls.settingsBlock}>
        <summary>
          <ProductIcon name="settings" />
          Sampling and advanced limits
        </summary>
        {mode === 'preview' && (
          <>
            <h4>Preview context</h4>
            <p className={styles.muted}>
              Optionally inspect routing for an existing session. Preview reads the current learning
              state without updating it.
            </p>
            <div className={styles.formGrid}>
              <label>
                Session ID
                <input
                  value={previewContext.session_id ?? ''}
                  placeholder="Optional"
                  onChange={(event) =>
                    onPreviewContextChange({ ...previewContext, session_id: event.target.value })
                  }
                />
              </label>
              <label>
                Conversation ID
                <input
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
        <h4>Sampling defaults</h4>
        <p className={styles.muted}>
          Fixed target profiles take precedence. Their effective settings remain visible above.
        </p>
        <div className={styles.formGrid}>
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
        <h4>Request and case limits</h4>
        <div className={styles.formGrid}>
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
      </details>
    </>
  )
}
