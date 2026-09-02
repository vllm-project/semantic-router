import { EvaluationActionButton, EvaluationTag } from './EvaluationPrimitives'
import styles from './EvaluationCapacitySLO.module.css'
import EvaluationExperimentSectionHeading from './EvaluationExperimentSectionHeading'
import sectionStyles from './EvaluationExperimentSection.module.css'
import type { EvaluationExperimentFormModel } from './useEvaluationExperimentForm'
import type { EvaluationCapacitySLOInput } from './useEvaluationExperimentFormState'

interface CapacitySLOPreset {
  id: string
  label: string
  description: string
  values: Omit<EvaluationCapacitySLOInput, 'requiredConcurrency'>
}

const CAPACITY_SLO_PRESETS: CapacitySLOPreset[] = [
  {
    id: 'latency',
    label: 'Latency guardrail',
    description: '250 ms p95 · 1% errors · 5 req/s · 60% scaling',
    values: {
      maxLatencyP95MS: '250',
      maxErrorRate: '0.01',
      minThroughputRPS: '5',
      minThroughputScalingEfficiency: '0.6',
    },
  },
  {
    id: 'balanced',
    label: 'Balanced service',
    description: '750 ms p95 · 2% errors · 10 req/s · 70% scaling',
    values: {
      maxLatencyP95MS: '750',
      maxErrorRate: '0.02',
      minThroughputRPS: '10',
      minThroughputScalingEfficiency: '0.7',
    },
  },
  {
    id: 'throughput',
    label: 'Throughput guardrail',
    description: '1500 ms p95 · 5% errors · 25 req/s · 80% scaling',
    values: {
      maxLatencyP95MS: '1500',
      maxErrorRate: '0.05',
      minThroughputRPS: '25',
      minThroughputScalingEfficiency: '0.8',
    },
  },
]

function CapacityProtocolSummary({ form }: { form: EvaluationExperimentFormModel }) {
  if (!form.capacityLoadProtocol) return null
  const protocol = form.capacityLoadProtocol
  return (
    <dl className={styles.protocolSummary} aria-label="Recorded capacity load plan">
      <div>
        <dt>Concurrency ladder</dt>
        <dd>{protocol.concurrency_levels.join(' → ')} concurrent requests</dd>
      </div>
      <div>
        <dt>Warmup</dt>
        <dd>{protocol.warmup_request_multiplier} × concurrency requests</dd>
      </div>
      <div>
        <dt>Measurement</dt>
        <dd>
          {protocol.measurement_requests_per_repetition} requests ×{' '}
          {protocol.repetitions_per_level} independent windows (minimum{' '}
          {protocol.minimum_measurement_clusters_per_level})
        </dd>
      </div>
      <div>
        <dt>Confidence / stability</dt>
        <dd>
          {(protocol.confidence_level * 100).toFixed(0)}% worst-window error bound · error spread
          ≤ {(protocol.max_error_rate_cluster_range * 100).toFixed(0)}% · throughput and p95
          variation ≤{' '}
          {(protocol.max_throughput_cv * 100).toFixed(0)}%
        </dd>
      </div>
    </dl>
  )
}

function CapacityPresets({ form }: { form: EvaluationExperimentFormModel }) {
  if (form.baselineLocked) return null
  return (
    <div className={styles.sloPresets} role="group" aria-label="Capacity starting points">
      <div>
        <span>Optional starting points</span>
        <small>Choose explicitly, then review every value against your service objective.</small>
      </div>
      {CAPACITY_SLO_PRESETS.map((preset) => (
        <EvaluationActionButton
          key={preset.id}
          type="button"
          compact
          variant="quiet"
          title={preset.description}
          aria-label={`${preset.label}: ${preset.description}`}
          onClick={() =>
            form.applyCapacitySLOPreset({
              requiredConcurrency: String(form.concurrency),
              ...preset.values,
            })
          }
        >
          {preset.label}
        </EvaluationActionButton>
      ))}
    </div>
  )
}

function RequiredConcurrencyField({ form }: { form: EvaluationExperimentFormModel }) {
  return (
    <label>
      <span>Required concurrency</span>
      <input
        type="number"
        min={1}
        max={form.concurrency}
        step={1}
        required
        value={form.capacitySLOInput.requiredConcurrency}
        disabled={form.baselineLocked}
        onChange={(event) => form.setCapacitySLOField('requiredConcurrency', event.target.value)}
      />
      <small>The qualified operating range must reach at least this concurrency.</small>
    </label>
  )
}

type CapacityUnitFieldKey = Exclude<keyof EvaluationCapacitySLOInput, 'requiredConcurrency'>

interface CapacityUnitFieldProps {
  form: EvaluationExperimentFormModel
  field: CapacityUnitFieldKey
  label: string
  min: string | number
  max?: string | number
  step: string | number
  unit: string
  help: string
}

function CapacityUnitField({
  form,
  field,
  label,
  min,
  max,
  step,
  unit,
  help,
}: CapacityUnitFieldProps) {
  return (
    <label>
      <span>{label}</span>
      <div className={styles.unitInput}>
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          required
          value={form.capacitySLOInput[field]}
          disabled={form.baselineLocked}
          onChange={(event) => form.setCapacitySLOField(field, event.target.value)}
        />
        <span>{unit}</span>
      </div>
      <small>{help}</small>
    </label>
  )
}

function CapacitySLOFields({ form }: { form: EvaluationExperimentFormModel }) {
  return (
    <div className={styles.sloGrid}>
      <RequiredConcurrencyField form={form} />
      <CapacityUnitField
        form={form}
        field="maxLatencyP95MS"
        label="Maximum p95 latency"
        min="0.1"
        step="0.1"
        unit="ms"
        help="Measured independently at every concurrency level."
      />
      <CapacityUnitField
        form={form}
        field="maxErrorRate"
        label="Maximum error rate"
        min={0}
        max="0.999999"
        step="0.001"
        unit="ratio"
        help="Use 0.01 for a one-percent request error budget."
      />
      <CapacityUnitField
        form={form}
        field="minThroughputRPS"
        label="Minimum throughput"
        min="0.1"
        step="0.1"
        unit="req/s"
        help="Applies at and above the required concurrency."
      />
      <CapacityUnitField
        form={form}
        field="minThroughputScalingEfficiency"
        label="Minimum scaling efficiency"
        min="0.01"
        max={1}
        step="0.01"
        unit="ratio"
        help="The selected load plan defines how scaling efficiency is measured."
      />
    </div>
  )
}

export default function EvaluationExperimentCapacitySLO({
  form,
}: {
  form: EvaluationExperimentFormModel
}) {
  if (!form.capacitySLOActive) return null
  return (
    <section className={`${sectionStyles.formSection} ${styles.sloSection}`}>
      <div className={styles.sloHeadingRow}>
        <EvaluationExperimentSectionHeading
          index="05"
          title="Capacity service objective"
          description="Define the live capacity thresholds that this deployment must satisfy under a controlled load test."
        />
        <EvaluationTag tone="info">Required for live capacity</EvaluationTag>
      </div>
      <div className={styles.sloExplanation}>
        <strong>No inferred pass criteria</strong>
        <span>
          Every service objective is evaluated under the recorded load plan. Missing or unstable
          measurements cannot qualify the operating point.
        </span>
      </div>
      <CapacityProtocolSummary form={form} />
      <CapacityPresets form={form} />
      <CapacitySLOFields form={form} />
    </section>
  )
}
