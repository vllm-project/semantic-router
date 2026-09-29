import { useId } from 'react'
import ProductIcon from '../ProductIcon'
import { profileTitle } from './datasetPresentation'
import type { ExperimentMember } from './experimentApi'
import { experimentQuestionCount, experimentRoleLabels } from './experimentEvidencePresentation'
import { RunStatus } from './RunList'
import type { Run } from './types'
import styles from './ExperimentEvidence.module.css'

const primaryGroups = [
  { role: 'baseline', title: 'Single-model reference', icon: 'model' },
  { role: 'initial', title: 'Starting recipe', icon: 'mixture' },
  { role: 'candidate', title: 'Recipe versions', icon: 'edit' },
  { role: 'validation', title: 'Final validation', icon: 'check' },
] as const

function EvidenceRow({
  member,
  run,
  onOpenRun,
  showRole,
}: {
  member: ExperimentMember
  showRole: boolean
  run?: Run
  onOpenRun: (id: string) => void
}) {
  const questions = run ? experimentQuestionCount(run) : undefined
  return (
    <li className={styles.row}>
      <div className={styles.identity}>
        {showRole && <span className={styles.role}>{experimentRoleLabels[member.role]}</span>}
        <button className={styles.runLink} type="button" onClick={() => onOpenRun(member.run_id)}>
          {run?.manifest.name || 'Saved run'}
          <ProductIcon name="arrow-right" />
        </button>
        {run ? (
          <dl className={styles.context}>
            <div>
              <dt>Mode</dt>
              <dd>
                {run.manifest.mode === 'live'
                  ? 'Live'
                  : run.manifest.mode === 'preview'
                    ? 'Preview'
                    : 'Replay'}
              </dd>
            </div>
            <div>
              <dt>Profile</dt>
              <dd>{profileTitle(run.manifest.profile)}</dd>
            </div>
            {questions !== undefined && (
              <div>
                <dt>Questions</dt>
                <dd>{questions.toLocaleString()}</dd>
              </div>
            )}
          </dl>
        ) : (
          <p className={styles.unavailable}>Open this run to view its status and progress.</p>
        )}
      </div>
      {run && (
        <div className={styles.outcome}>
          <RunStatus status={run.status} />
          <span className={styles.progress}>
            <strong>
              {run.progress.completed.toLocaleString()} / {run.progress.total.toLocaleString()}
            </strong>{' '}
            attempts completed
          </span>
          <span className={run.progress.failed > 0 ? styles.failures : styles.noFailures}>
            {run.progress.failed.toLocaleString()} failed
          </span>
        </div>
      )}
      {member.hypothesis.trim() && (
        <p className={styles.note}>
          <span>Change being tested</span>
          {member.hypothesis}
        </p>
      )}
    </li>
  )
}

export default function ExperimentEvidence({
  members,
  runs,
  onOpenRun,
  hasMore,
}: {
  members: ExperimentMember[]
  runs: Run[]
  onOpenRun: (id: string) => void
  hasMore: boolean
}) {
  const headingID = useId()
  const byID = new Map(runs.map((run) => [run.id, run]))
  const supporting = members.filter(
    (member) => !primaryGroups.some((group) => group.role === member.role),
  )
  const rows = (items: ExperimentMember[], showRoles = false) => (
    <ul className={styles.rows}>
      {items.map((member) => (
        <EvidenceRow
          key={member.run_id}
          member={member}
          showRole={showRoles}
          run={byID.get(member.run_id)}
          onOpenRun={onOpenRun}
        />
      ))}
    </ul>
  )
  return (
    <section className={styles.evidence} aria-label="Saved runs">
      {!members.length && (
        <p className={styles.empty}>
          {hasMore
            ? 'No saved runs on this page yet. Continue to the next page to see more.'
            : 'No saved runs on this page yet. Start with a single-model reference, or add a run you have already saved.'}
        </p>
      )}
      {primaryGroups.map((group) => {
        const items = members.filter((member) => member.role === group.role)
        if (!items.length) return null
        const groupID = `${headingID}-${group.role}`
        return (
          <section key={group.role} className={styles.group} aria-labelledby={groupID}>
            <div className={styles.groupHeading}>
              <ProductIcon name={group.icon} />
              <h4 id={groupID}>{group.title}</h4>
              <span>{items.length} on this page</span>
            </div>
            {rows(items)}
          </section>
        )
      })}
      {supporting.length > 0 && (
        <details className={styles.supporting}>
          <summary>
            <ProductIcon name="chevron-right" />
            <strong>Supporting checks</strong>
            <span>{supporting.length} on this page</span>
          </summary>
          <p className={styles.supportingDescription}>
            Routing previews, pipeline checks, offline estimates and recovery attempts.
          </p>
          {rows(supporting, true)}
        </details>
      )}
      {hasMore && members.length > 0 && (
        <p className={styles.pageNotice}>More saved runs are available on the next page.</p>
      )}
    </section>
  )
}
