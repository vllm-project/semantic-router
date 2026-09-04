import type {
  EvaluationCampaignFidelityEvidence,
  EvaluationCampaignPairedLiveEvidence,
} from '../../types/evaluationCampaign'
import { TechnicalFieldGrid } from './EvaluationCampaignDecisionTechnicalFields'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import { copyField, digestField, type TechnicalField } from './evaluationTechnicalFields'
import styles from './EvaluationCampaignDecisionTechnicalDetails.module.css'

const CONTROLLED_PAIR_PROTOCOL_LABELS = {
  'abba-interleaved.v1': 'Order-balanced paired execution',
} satisfies Record<EvaluationCampaignPairedLiveEvidence['controlled_pair_protocol'], string>

function pairedIdentityFields(evidence: EvaluationCampaignPairedLiveEvidence): TechnicalField[] {
  return [
    copyField('Baseline deployment', evidence.baseline_target_id),
    copyField('Candidate deployment', evidence.candidate_target_id),
    copyField('Baseline run', evidence.baseline_run_id),
    copyField('Candidate run', evidence.candidate_run_id),
    { label: 'Mixture', value: evidence.recipe_name },
    copyField('Mixture ID', evidence.mixture_id),
    {
      label: 'Evaluation areas',
      value: evidence.track_ids.map((trackID) => TRACK_PRESENTATION[trackID].label).join(' · '),
    },
    {
      label: 'Sampling plan',
      value: `${evidence.bootstrap_samples.toLocaleString()} samples · ${Math.round(evidence.confidence_level * 100)}% confidence`,
    },
    { label: 'Random seed', value: evidence.seed },
    {
      label: 'Execution method',
      value: CONTROLLED_PAIR_PROTOCOL_LABELS[evidence.controlled_pair_protocol],
    },
    { label: 'Execution protocol', value: evidence.controlled_pair_protocol, mono: true },
    copyField('Session ID', evidence.controlled_pair_session_id),
    ...evidence.model_pool_arm_reliability.map((statistic, index) =>
      copyField(`Model reliability binding ${index + 1}`, statistic.arm_id),
    ),
    digestField('Candidate configuration', evidence.candidate_subject_digest),
    digestField('Comparison receipt', evidence.digest),
  ]
}

function pairedSnapshotFields(evidence: EvaluationCampaignPairedLiveEvidence): TechnicalField[] {
  const pairedSnapshots: Array<readonly [string, string, string]> = [
    ['manifest', evidence.baseline_manifest_digest, evidence.candidate_manifest_digest],
    [
      'server execution receipt',
      evidence.baseline_execution_attestation_digest,
      evidence.candidate_execution_attestation_digest,
    ],
    [
      'policy snapshot',
      evidence.baseline_policy_snapshot_digest,
      evidence.candidate_policy_snapshot_digest,
    ],
    [
      'policy binding',
      evidence.baseline_binding_snapshot_digest,
      evidence.candidate_binding_snapshot_digest,
    ],
    [
      'model pool snapshot',
      evidence.baseline_pool_snapshot_digest,
      evidence.candidate_pool_snapshot_digest,
    ],
    [
      'environment snapshot',
      evidence.baseline_environment_snapshot_digest,
      evidence.candidate_environment_snapshot_digest,
    ],
    [
      'backend topology',
      evidence.baseline_backend_topology_digest,
      evidence.candidate_backend_topology_digest,
    ],
  ]

  return [
    digestField('Workload snapshot', evidence.workload_snapshot_digest),
    ...Object.entries(evidence.benchmark_revisions).map(([benchmark, revision]) =>
      digestField(`Benchmark · ${benchmark}`, revision),
    ),
    ...pairedSnapshots.flatMap(([label, baseline, candidate]) => [
      digestField(`Baseline ${label}`, baseline),
      digestField(`Candidate ${label}`, candidate),
    ]),
    copyField('Baseline code revision', evidence.baseline_code_revision),
    copyField('Candidate code revision', evidence.candidate_code_revision),
  ]
}

function pairedMeasurementFields(evidence: EvaluationCampaignPairedLiveEvidence): TechnicalField[] {
  return [
    ...evidence.promotion_statistics.flatMap((statistic, index) => [
      { label: `Release measure ${index + 1} ID`, value: statistic.id, mono: true },
      {
        label: `Release measure ${index + 1} reported threshold`,
        value: `${statistic.threshold.operator} ${statistic.threshold.value} ${statistic.threshold.unit}`,
        mono: true,
      },
    ]),
    ...evidence.statistics.flatMap((statistic, index) => [
      { label: `Diagnostic measure ${index + 1} ID`, value: statistic.id, mono: true },
      {
        label: `Diagnostic measure ${index + 1} analysis unit`,
        value: statistic.analysis_unit,
        mono: true,
      },
    ]),
  ]
}

function fidelityFields(evidence: EvaluationCampaignFidelityEvidence): TechnicalField[] {
  return [
    copyField('Reference run', evidence.reference_run_id),
    copyField('Fresh live run', evidence.live_run_id),
    digestField('Candidate configuration', evidence.candidate_subject_digest),
    { label: 'Evaluation area', value: TRACK_PRESENTATION[evidence.track_id].label },
    { label: 'Benchmark suites', value: evidence.suite_ids.join(' · ') },
    digestField('Consistency receipt', evidence.digest),
    digestField('Reference manifest', evidence.reference_manifest_digest),
    digestField('Fresh live manifest', evidence.live_manifest_digest),
    digestField('Fresh live server execution receipt', evidence.live_execution_attestation_digest),
    digestField('Workload snapshot', evidence.workload_snapshot_digest),
    ...Object.entries(evidence.benchmark_revisions).map(([benchmark, revision]) =>
      digestField(`Benchmark · ${benchmark}`, revision),
    ),
  ]
}

export function ControlledComparisonDetails({
  evidence,
}: {
  evidence: EvaluationCampaignPairedLiveEvidence
}) {
  return (
    <section
      className={styles.technicalGroup}
      aria-labelledby="campaign-technical-comparison-title"
    >
      <div className={styles.technicalGroupHeader}>
        <h4 id="campaign-technical-comparison-title">Controlled comparison</h4>
        <p>Execution identity and frozen inputs for reproducing the matched comparison.</p>
      </div>
      <TechnicalFieldGrid
        label="Controlled comparison identity"
        fields={pairedIdentityFields(evidence)}
      />
      <div className={styles.technicalSubgroup}>
        <h5>Reported measurement definitions</h5>
        <TechnicalFieldGrid
          label="Reported measurement definitions"
          fields={pairedMeasurementFields(evidence)}
        />
      </div>
      <div className={styles.technicalSubgroup}>
        <h5>Reproducibility snapshots</h5>
        <TechnicalFieldGrid
          label="Controlled comparison reproducibility snapshots"
          fields={pairedSnapshotFields(evidence)}
        />
      </div>
    </section>
  )
}

export function FidelityDetails({ evidence }: { evidence: EvaluationCampaignFidelityEvidence }) {
  return (
    <section className={styles.technicalGroup} aria-labelledby="campaign-technical-fidelity-title">
      <div className={styles.technicalGroupHeader}>
        <h4 id="campaign-technical-fidelity-title">Live consistency comparison</h4>
        <p>Run identities and frozen inputs for reproducing the live consistency check.</p>
      </div>
      <TechnicalFieldGrid
        label="Live consistency identity and receipts"
        fields={fidelityFields(evidence)}
      />
    </section>
  )
}
