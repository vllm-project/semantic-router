import type { EvaluationCatalog } from '../../types/evaluationPlane'
import { compatibleSuiteEmptyReason } from './evaluationExperimentValidation'
import { evaluationResultScopeLabel } from './evaluationPresentation'
import { TRACK_PRESENTATION } from './evaluationTrackPresentation'
import type { EvaluationExperimentFormModel } from './useEvaluationExperimentForm'
import EvaluationExperimentSectionHeading from './EvaluationExperimentSectionHeading'
import styles from './EvaluationExperimentBenchmarkScope.module.css'
import noticeStyles from './EvaluationExperimentNotice.module.css'
import sectionStyles from './EvaluationExperimentSection.module.css'

interface EvaluationExperimentBenchmarkScopeProps {
  catalog: EvaluationCatalog
  form: EvaluationExperimentFormModel
}

export default function EvaluationExperimentBenchmarkScope({
  catalog,
  form,
}: EvaluationExperimentBenchmarkScopeProps) {
  return (
    <>
      <section className={sectionStyles.formSection}>
        <EvaluationExperimentSectionHeading
          index="03"
          title="Benchmarks"
          description="Select versioned workloads, then choose what this run should measure."
        />
        {form.compatibleSuites.length ? (
          <div className={styles.catalogGrid} role="group" aria-label="Benchmark selection">
            {form.compatibleSuites.map((suite) => (
              <label
                key={suite.id}
                className={`${styles.catalogCard} ${form.suiteIDs.includes(suite.id) ? styles.active : ''}`}
              >
                <input
                  type="checkbox"
                  checked={form.suiteIDs.includes(suite.id)}
                  disabled={form.baselineLocked}
                  onChange={() => form.toggleSuite(suite.id)}
                />
                <span>
                  <strong>{suite.name}</strong>
                  <small>{suite.description}</small>
                  <em>
                    {evaluationResultScopeLabel(suite.evidence_level)}
                    {suite.case_count ? ` · ${suite.case_count} cases` : ''}
                  </em>
                </span>
              </label>
            ))}
          </div>
        ) : (
          <div className={noticeStyles.contractWarning} role="status">
            {compatibleSuiteEmptyReason(catalog, form.targetID, form.mode)}
          </div>
        )}
      </section>

      <section className={sectionStyles.formSection}>
        <EvaluationExperimentSectionHeading
          index="04"
          title="Evaluation areas"
          description="Each area reports its own status, measurements, coverage, and release checks."
        />
        {form.selectableTrackIDs.length === 0 ? (
          <div className={noticeStyles.contractWarning} role="status">
            {form.suiteIDs.length === 0
              ? 'Select a compatible benchmark to see the areas it can measure.'
              : 'The selected benchmarks cannot measure any area for this Mixture and run type.'}
          </div>
        ) : null}
        <div className={styles.trackGrid} role="group" aria-label="Evaluation area selection">
          {catalog.tracks.map((track) => {
            const targetSupportsTrack = form.availableTrackIDs.includes(track.id)
            const available = form.selectableTrackIDs.includes(track.id)
            return (
              <label
                key={track.id}
                className={`${styles.trackCard} ${form.trackIDs.includes(track.id) ? styles.active : ''} ${!available ? styles.disabled : ''}`}
              >
                <input
                  type="checkbox"
                  checked={form.trackIDs.includes(track.id)}
                  disabled={form.baselineLocked || !available}
                  onChange={() => form.toggleTrack(track.id)}
                />
                <span>
                  <strong>{TRACK_PRESENTATION[track.id].label}</strong>
                  <small>{track.description}</small>
                  <em data-evaluation-unavailable-reason={!available ? 'true' : undefined}>
                    {available
                      ? `${track.metrics.length} ${track.metrics.length === 1 ? 'measurement' : 'measurements'}`
                      : !targetSupportsTrack
                        ? `Not supported for ${form.mode} on this target`
                        : 'Not included by the selected benchmarks'}
                  </em>
                </span>
              </label>
            )
          })}
        </div>
      </section>
    </>
  )
}
