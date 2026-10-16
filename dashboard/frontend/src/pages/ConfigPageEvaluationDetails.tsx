import { useMemo, useState } from 'react'

import { DataTable, type Column } from '../components/DataTable'
import TableHeader from '../components/TableHeader'
import type {
  EffectiveEvaluationGroup,
  EffectiveEvaluationRecord,
} from './configPageEffectiveEvaluations'
import styles from './ConfigPageEvaluationDetails.module.css'

const sourceLabel = (record: EffectiveEvaluationRecord): string =>
  record.origin === 'built_in' ? 'Built-in catalog' : 'Configured record'

function SourceReference({ value }: { value: string }) {
  return /^https?:\/\//i.test(value) ? (
    <a href={value} target="_blank" rel="noreferrer">
      {value}
    </a>
  ) : (
    <span>{value}</span>
  )
}

function EvaluationResult({ record }: { record: EffectiveEvaluationRecord }) {
  const { evaluation, benchmark, issues } = record
  return (
    <div className={styles.cell}>
      <strong>{benchmark?.display_name || evaluation.benchmark}</strong>
      {benchmark?.display_name ? (
        <span className={styles.secondary}>{evaluation.benchmark}</span>
      ) : null}
      <span>Effort: {evaluation.reasoning_effort}</span>
      <span>Profile: {evaluation.benchmark_profile || 'Not resolved'}</span>
      <span>Status: {evaluation.status.replace(/_/g, ' ')}</span>
      {Object.entries(evaluation.metrics).map(([name, value]) => {
        const metric = benchmark?.metrics.find((definition) => definition.id === name)
        return (
          <span key={name}>
            {name}: <strong>{value}</strong>
            {metric
              ? ` · ${metric.unit} (${metric.range[0]}–${metric.range[1]})`
              : ' · unit not defined'}
          </span>
        )
      })}
      {Object.keys(evaluation.metrics).length === 0 ? <span>No metric values recorded</span> : null}
      {issues.map((issue) => (
        <span className={styles.issue} key={issue}>
          {issue}
        </span>
      ))}
    </div>
  )
}

function EvaluationSource({ record }: { record: EffectiveEvaluationRecord }) {
  const { evaluation } = record
  const { evidence } = evaluation
  return (
    <div className={styles.cell}>
      <strong>{sourceLabel(record)}</strong>
      <span>Provenance: {evidence.provenance.replace(/_/g, ' ')}</span>
      <span>Verification: {evidence.verification}</span>
      {evidence.source ? (
        <SourceReference value={evidence.source} />
      ) : (
        <span>Source not recorded</span>
      )}
      {evidence.artifact ? (
        <span>
          Artifact: <SourceReference value={evidence.artifact} />
        </span>
      ) : null}
      {evaluation.measured_at ? <span>Measured: {evaluation.measured_at}</span> : null}
      {evaluation.observed_at ? <span>Observed: {evaluation.observed_at}</span> : null}
      {Object.keys(evaluation.subject).length > 0 ? (
        <details>
          <summary>Measurement details</summary>
          <pre className={styles.metadata}>{JSON.stringify(evaluation.subject, null, 2)}</pre>
        </details>
      ) : null}
    </div>
  )
}

export function filterEvaluationDetails(records: EffectiveEvaluationRecord[], query: string) {
  const search = query.trim().toLowerCase()
  if (!search) return records
  return records.filter((record) =>
    [
      record.evaluation.benchmark,
      record.benchmark?.display_name,
      record.evaluation.benchmark_profile,
      record.evaluation.reasoning_effort,
      record.evaluation.status,
      record.evaluation.evidence.source,
      record.evaluation.evidence.provenance,
      record.evaluation.evidence.verification,
      sourceLabel(record),
      ...Object.keys(record.evaluation.metrics),
    ].some((value) => value?.toLowerCase().includes(search)),
  )
}

const columns: Column<EffectiveEvaluationRecord>[] = [
  {
    key: 'result',
    header: 'Benchmark result',
    render: (record) => <EvaluationResult record={record} />,
  },
  {
    key: 'evidence',
    header: 'Evidence source',
    render: (record) => <EvaluationSource record={record} />,
  },
]

export default function ConfigPageEvaluationDetails({
  group,
}: {
  group: EffectiveEvaluationGroup
}) {
  const [search, setSearch] = useState('')
  const records = useMemo(
    () => filterEvaluationDetails(group.records, search),
    [group.records, search],
  )
  return (
    <div>
      <TableHeader
        title="Evidence"
        count={records.length}
        searchPlaceholder="Search benchmark, effort, status, or source..."
        searchValue={search}
        onSearchChange={setSearch}
        variant="embedded"
      />
      <DataTable
        columns={columns}
        data={records}
        keyExtractor={(record) => record.key}
        readonly
        emptyMessage={
          search
            ? 'No evidence matches this search.'
            : 'No built-in or configured evaluation evidence is available for this model.'
        }
        pagination={{
          pageSize: 5,
          pageSizeOptions: [5, 10, 25, 50],
          itemLabel: 'records',
          resetKey: search,
        }}
      />
    </div>
  )
}
