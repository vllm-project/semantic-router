import { useCallback } from 'react';
import type { CreateTaskRequest, DatasetInfo, EvaluationDimension } from '../../types/evaluation';
import { DIMENSION_INFO } from '../../types/evaluation';
import { useTaskCreationForm, useDatasets } from '../../hooks/useEvaluation';
import styles from './TaskCreationForm.module.css';

interface TaskCreationFormProps {
  onSubmit: (request: CreateTaskRequest) => void;
  onCancel: () => void;
  loading?: boolean;
}

export function TaskCreationForm({ onSubmit, onCancel, loading }: TaskCreationFormProps) {
  const form = useTaskCreationForm();
  const { datasets: availableDatasets, loading: datasetsLoading } = useDatasets();

  const handleSubmit = useCallback(() => {
    const config = form.getConfig();
    onSubmit(config);
  }, [form, onSubmit]);

  const renderStepIndicator = () => (
    <div className={styles.stepIndicator}>
      {[1, 2, 3, 4].map((s) => (
        <button
          key={s}
          className={`${styles.step} ${form.step === s ? styles.active : ''} ${form.step > s ? styles.completed : ''}`}
          onClick={() => form.goToStep(s)}
          disabled={s > form.step && !form.isStepValid(form.step)}
        >
          <span className={styles.stepNumber}>{s}</span>
          <span className={styles.stepLabel}>
            {s === 1 && 'Basic Info'}
            {s === 2 && 'Dimensions'}
            {s === 3 && 'Datasets'}
            {s === 4 && 'Review'}
          </span>
        </button>
      ))}
    </div>
  );

  const renderStep1 = () => (
    <div className={styles.stepContent}>
      <h3>Basic Information</h3>
      <p className={styles.stepDescription}>Enter a name and description for your evaluation task.</p>

      <div className={styles.formGroup}>
        <label htmlFor="name">Task Name *</label>
        <input
          id="name"
          type="text"
          value={form.name}
          onChange={(e) => form.setName(e.target.value)}
          placeholder="e.g., Weekly MoM Evaluation"
          className={styles.input}
        />
      </div>

      <div className={styles.formGroup}>
        <label htmlFor="description">Description</label>
        <textarea
          id="description"
          value={form.description}
          onChange={(e) => form.setDescription(e.target.value)}
          placeholder="Describe the purpose of this evaluation..."
          className={styles.textarea}
          rows={3}
        />
      </div>

      <div className={styles.formRow}>
        <div className={styles.formGroup}>
          <label htmlFor="maxSamples">Max Samples</label>
          <input
            id="maxSamples"
            type="number"
            value={form.maxSamples}
            onChange={(e) => form.setMaxSamples(parseInt(e.target.value) || 50)}
            min={1}
            max={1000}
            className={styles.input}
          />
        </div>
        <div className={styles.formGroup}>
          <label htmlFor="samplesPerCat">Samples per Category</label>
          <input
            id="samplesPerCat"
            type="number"
            value={form.samplesPerCat}
            onChange={(e) => form.setSamplesPerCat(parseInt(e.target.value) || 10)}
            min={1}
            max={100}
            className={styles.input}
          />
        </div>
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className={styles.stepContent}>
      <h3>Select Evaluation Dimensions</h3>
      <p className={styles.stepDescription}>Choose which aspects of the MoM system to evaluate.</p>

      <div className={styles.dimensionGrid}>
        {(Object.entries(DIMENSION_INFO) as [EvaluationDimension, typeof DIMENSION_INFO[EvaluationDimension]][]).map(
          ([dim, info]) => (
            <button
              key={dim}
              className={`${styles.dimensionCard} ${form.dimensions.includes(dim) ? styles.selected : ''}`}
              onClick={() => form.toggleDimension(dim)}
              style={{ '--dim-color': info.color } as React.CSSProperties}
            >
              <div className={styles.dimensionHeader}>
                <span className={styles.dimensionIndicator} style={{ backgroundColor: info.color }} />
                <span className={styles.dimensionLabel}>{info.label}</span>
              </div>
              <p className={styles.dimensionDescription}>{info.description}</p>
            </button>
          )
        )}
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className={styles.stepContent}>
      <h3>Select Datasets</h3>
      <p className={styles.stepDescription}>Choose which datasets to use for each dimension.</p>

      {datasetsLoading ? (
        <div className={styles.loading}>Loading datasets...</div>
      ) : (
        <div className={styles.datasetGroups}>
          {form.dimensions.map((dim) => {
            const datasets = availableDatasets[dim] || [];
            return (
              <div key={dim} className={styles.datasetGroup}>
                <h4 style={{ color: DIMENSION_INFO[dim].color }}>{DIMENSION_INFO[dim].label}</h4>
                <div className={styles.datasetList}>
                  {datasets.length === 0 ? (
                    <p className={styles.noDatasets}>No datasets available for this dimension.</p>
                  ) : (
                    datasets.map((ds: DatasetInfo) => (
                      <label key={ds.name} className={styles.datasetItem}>
                        <input
                          type="checkbox"
                          checked={form.selectedDatasets[dim]?.includes(ds.name) || false}
                          onChange={() => form.toggleDataset(dim, ds.name)}
                        />
                        <div className={styles.datasetInfo}>
                          <span className={styles.datasetName}>{ds.name}</span>
                          <span className={styles.datasetDesc}>{ds.description}</span>
                        </div>
                      </label>
                    ))
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  const renderStep4 = () => {
    const config = form.getConfig();
    return (
      <div className={styles.stepContent}>
        <h3>Review & Submit</h3>
        <p className={styles.stepDescription}>Review your evaluation configuration before submitting.</p>

        <div className={styles.reviewSection}>
          <h4>Task Details</h4>
          <dl className={styles.reviewList}>
            <dt>Name</dt>
            <dd>{config.name}</dd>
            <dt>Description</dt>
            <dd>{config.description || '-'}</dd>
            <dt>Endpoint</dt>
            <dd>{config.config.endpoint}</dd>
            <dt>Max Samples</dt>
            <dd>{config.config.max_samples}</dd>
          </dl>
        </div>

        <div className={styles.reviewSection}>
          <h4>Dimensions</h4>
          <div className={styles.dimensionTags}>
            {config.config.dimensions.map((dim) => (
              <span
                key={dim}
                className={styles.dimensionTag}
                style={{ backgroundColor: `${DIMENSION_INFO[dim].color}20`, color: DIMENSION_INFO[dim].color }}
              >
                {DIMENSION_INFO[dim].label}
              </span>
            ))}
          </div>
        </div>

        <div className={styles.reviewSection}>
          <h4>Datasets</h4>
          <dl className={styles.reviewList}>
            {Object.entries(config.config.datasets).map(([dim, datasets]) => (
              <div key={dim}>
                <dt>{DIMENSION_INFO[dim as EvaluationDimension]?.label || dim}</dt>
                <dd>{(datasets as string[]).join(', ') || 'default'}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    );
  };

  return (
    <div className={styles.container}>
      {renderStepIndicator()}

      <div className={styles.content}>
        {form.step === 1 && renderStep1()}
        {form.step === 2 && renderStep2()}
        {form.step === 3 && renderStep3()}
        {form.step === 4 && renderStep4()}
      </div>

      <div className={styles.footer}>
        <button className={styles.cancelButton} onClick={onCancel} disabled={loading}>
          Cancel
        </button>
        <div className={styles.navButtons}>
          {form.step > 1 && (
            <button className={styles.prevButton} onClick={form.prevStep} disabled={loading}>
              Previous
            </button>
          )}
          {form.step < 4 ? (
            <button
              className={styles.nextButton}
              onClick={form.nextStep}
              disabled={!form.isStepValid(form.step) || loading}
            >
              Next
            </button>
          ) : (
            <button className={styles.submitButton} onClick={handleSubmit} disabled={loading}>
              {loading ? 'Creating...' : 'Create Task'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export default TaskCreationForm;
