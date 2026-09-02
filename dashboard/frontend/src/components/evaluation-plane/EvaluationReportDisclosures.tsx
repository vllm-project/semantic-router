import type { EvaluationReport } from '../../types/evaluationReport'
import { formatDateTime } from '../../utils/dateTime'
import {
  getEvaluationArtifactURL,
  isDownloadableEvaluationArtifact,
} from '../../utils/evaluationPlaneApi'
import EvaluationDisclosure, { EvaluationTechnicalDisclosure } from './EvaluationDisclosure'
import EvaluationGateList from './EvaluationGateList'
import {
  evaluationGateFollowUpGuidance,
  formatMetric,
} from './evaluationPresentation'
import reportStyles from './EvaluationReportLayout.module.css'
import styles from './EvaluationReportDisclosures.module.css'

function presentCount(value: number | undefined, suffix: string): string {
  return typeof value === 'number'
    ? `${new Intl.NumberFormat().format(value)} ${suffix}`
    : 'Not recorded'
}

function reportProductFindings(report: EvaluationReport): string[] {
  const productFindings = report.gates
    .filter(
      (gate) =>
        gate.disposition === 'required' &&
        (gate.verdict === 'fail' || gate.verdict === 'unavailable'),
    )
    .map(
      (gate) => `${gate.name}: ${evaluationGateFollowUpGuidance(gate.id)}`,
    )
  if (report.run.evidence_level === 'E0') {
    productFindings.unshift(
      'Use this diagnostic result to verify the evaluation setup; collect controlled or live results before making a release decision.',
    )
  }
  return productFindings
}

function ReleaseChecksDisclosure({ report }: { report: EvaluationReport }) {
  return (
    <EvaluationDisclosure
      className={styles.disclosure}
      data-evaluation-report-disclosure="true"
      summary={
        <>
          All release checks <span>{report.gates.length}</span>
        </>
      }
      summaryClassName={styles.disclosureSummary}
    >
      <div className={styles.disclosureBody}>
        <EvaluationGateList gates={report.gates} />
      </div>
    </EvaluationDisclosure>
  )
}

function CostRecord({
  name,
  ledger,
}: {
  name: string
  ledger: EvaluationReport['costs'][keyof EvaluationReport['costs']]
}) {
  return (
    <article>
      <span>{name.replace(/_/g, ' ')}</span>
      <strong>{formatMetric({ value: ledger.amount, unit: ledger.currency.toLowerCase() })}</strong>
      <small>
        {presentCount(ledger.input_tokens, 'input tokens')} ·{' '}
        {presentCount(ledger.output_tokens, 'output tokens')}
        {typeof ledger.gpu_seconds === 'number'
          ? ` · ${ledger.gpu_seconds.toFixed(1)} GPU seconds`
          : ''}
        {typeof ledger.energy_kwh === 'number' ? ` · ${ledger.energy_kwh.toFixed(2)} kWh` : ''}
      </small>
    </article>
  )
}

function RecordedCostsDisclosure({ report }: { report: EvaluationReport }) {
  return (
    <EvaluationDisclosure
      className={styles.disclosure}
      data-evaluation-report-disclosure="true"
      summary={
        <>
          Recorded costs <span>3 categories</span>
        </>
      }
      summaryClassName={styles.disclosureSummary}
    >
      <div className={styles.disclosureBody}>
        <p className={reportStyles.scopeCopy}>Costs verified from recorded usage.</p>
        <div className={styles.ledgerGrid}>
          {Object.entries(report.costs).map(([name, ledger]) => (
            <CostRecord key={name} name={name} ledger={ledger} />
          ))}
        </div>
      </div>
    </EvaluationDisclosure>
  )
}

function NextStepsDisclosure({ report }: { report: EvaluationReport }) {
  const productFindings = reportProductFindings(report)
  return (
    <EvaluationDisclosure
      className={styles.disclosure}
      data-evaluation-report-disclosure="true"
      summary={
        <>
          Next evaluation steps <span>{productFindings.length}</span>
        </>
      }
      summaryClassName={styles.disclosureSummary}
    >
      <div className={styles.disclosureBody}>
        <p className={reportStyles.scopeCopy}>
          These steps follow from the measured scope and incomplete release checks. They do not turn
          a diagnostic result into a release decision.
        </p>
        {productFindings.length ? (
          <ol className={styles.recommendations}>
            {productFindings.map((item, index) => (
              <li key={`${index}-${item}`}>{item}</li>
            ))}
          </ol>
        ) : (
          <p className={reportStyles.empty}>No required follow-up was identified.</p>
        )}
        {report.recommendations.length ? (
          <EvaluationTechnicalDisclosure
            className={styles.technicalNotes}
            summary={`Technical details · ${report.recommendations.length}`}
            summaryClassName={styles.technicalNotesSummary}
          >
            <p>Recorded service notes are retained verbatim for debugging and reproducibility.</p>
            <ol className={styles.recommendations}>
              {report.recommendations.map((item, index) => (
                <li key={`${index}-${item}`}>{item}</li>
              ))}
            </ol>
          </EvaluationTechnicalDisclosure>
        ) : null}
      </div>
    </EvaluationDisclosure>
  )
}

function ProvenanceField({
  label,
  value,
  code = true,
}: {
  label: string
  value: string | number | undefined
  code?: boolean
}) {
  const displayed = value === undefined || value === null || value === '' ? 'Not recorded' : value
  return (
    <div>
      <dt>{label}</dt>
      <dd>{code ? <code>{displayed}</code> : displayed}</dd>
    </div>
  )
}

function BenchmarkRevisions({ revisions }: { revisions: Record<string, string> | undefined }) {
  const entries = Object.entries(revisions || {})
  return (
    <div className={styles.provenanceWide}>
      <dt>Benchmark revisions</dt>
      <dd>
        {entries.length
          ? entries.map(([name, revision]) => (
              <span key={name}>
                {name}: <code>{revision}</code>
              </span>
            ))
          : 'Not recorded'}
      </dd>
    </div>
  )
}

function ReproducibilityDisclosure({ report }: { report: EvaluationReport }) {
  const provenance = report.provenance
  return (
    <EvaluationTechnicalDisclosure
      className={styles.disclosure}
      data-evaluation-report-disclosure="true"
      summary="Reproducibility details"
      summaryClassName={styles.disclosureSummary}
    >
      <div className={styles.disclosureBody}>
        <dl className={styles.provenance}>
          <ProvenanceField
            label="Generated"
            value={formatDateTime(provenance.generated_at)}
            code={false}
          />
          <ProvenanceField label="Target" value={provenance.target_id} />
          <ProvenanceField label="Seed" value={provenance.seed} code={false} />
          <ProvenanceField
            label="Result verification"
            value="Verified by the evaluation service"
            code={false}
          />
          <ProvenanceField label="Code revision" value={provenance.code_revision} />
          <ProvenanceField label="Workload snapshot" value={provenance.workload_snapshot_digest} />
          <ProvenanceField label="Policy snapshot" value={provenance.policy_snapshot_digest} />
          <ProvenanceField label="Policy binding" value={provenance.binding_snapshot_digest} />
          <ProvenanceField label="Pool snapshot" value={provenance.pool_snapshot_digest} />
          <ProvenanceField label="Environment" value={provenance.environment_snapshot_digest} />
          <ProvenanceField label="Redaction" value={provenance.redaction_policy} code={false} />
          <BenchmarkRevisions revisions={provenance.benchmark_revisions} />
        </dl>
      </div>
    </EvaluationTechnicalDisclosure>
  )
}

function ArtifactRecord({
  artifact,
  runID,
}: {
  artifact: EvaluationReport['artifacts'][number]
  runID: string
}) {
  const artifactType = (() => {
    if (artifact.media_type === 'application/pdf') return 'PDF document'
    if (artifact.media_type?.startsWith('image/')) return 'Image'
    if (artifact.kind === 'json') return 'Structured data'
    if (artifact.kind === 'jsonl') return 'Event records'
    if (artifact.kind === 'csv') return 'Data table'
    if (artifact.kind === 'sha256') return 'Verification checksums'
    if (artifact.kind === 'text' || artifact.kind === 'txt') return 'Text file'
    return 'Supporting file'
  })()
  return (
    <article>
      <div className={styles.artifactSummary}>
        <strong>{artifact.name}</strong>
        <span>{artifactType}</span>
      </div>
      <div className={styles.artifactAction}>
        {isDownloadableEvaluationArtifact(artifact) ? (
          <a
            href={getEvaluationArtifactURL(runID, artifact.id)}
            aria-label={`Download ${artifact.name}`}
          >
            Download
          </a>
        ) : (
          <span>Download unavailable</span>
        )}
      </div>
      <EvaluationTechnicalDisclosure
        className={styles.artifactTechnical}
        summary="Technical details"
        summaryClassName={styles.artifactTechnicalSummary}
      >
        <dl className={styles.artifactTechnicalGrid}>
          <div>
            <dt>File identifier</dt>
            <dd>
              <code>{artifact.id}</code>
            </dd>
          </div>
          <div>
            <dt>Stored format</dt>
            <dd>
              <code>{artifact.kind || 'Not recorded'}</code>
            </dd>
          </div>
          <div>
            <dt>Media type</dt>
            <dd>
              <code>{artifact.media_type || 'Not recorded'}</code>
            </dd>
          </div>
          {artifact.digest ? (
            <div>
              <dt>Digest</dt>
              <dd>
                <code>{artifact.digest}</code>
              </dd>
            </div>
          ) : null}
        </dl>
      </EvaluationTechnicalDisclosure>
    </article>
  )
}

function SupportingFilesDisclosure({ report }: { report: EvaluationReport }) {
  return (
    <EvaluationDisclosure
      className={styles.disclosure}
      data-evaluation-report-disclosure="true"
      summary={
        <>
          Supporting files <span>{report.artifacts.length}</span>
        </>
      }
      summaryClassName={styles.disclosureSummary}
    >
      <div className={styles.disclosureBody}>
        {report.artifacts.length ? (
          <div className={styles.artifactList}>
            {report.artifacts.map((artifact) => (
              <ArtifactRecord key={artifact.id} artifact={artifact} runID={report.run.id} />
            ))}
          </div>
        ) : (
          <p className={reportStyles.empty}>No report artifacts were recorded.</p>
        )}
      </div>
    </EvaluationDisclosure>
  )
}

export default function EvaluationReportDisclosures({ report }: { report: EvaluationReport }) {
  return (
    <>
      <ReleaseChecksDisclosure report={report} />
      <RecordedCostsDisclosure report={report} />
      <NextStepsDisclosure report={report} />
      <ReproducibilityDisclosure report={report} />
      <SupportingFilesDisclosure report={report} />
    </>
  )
}
